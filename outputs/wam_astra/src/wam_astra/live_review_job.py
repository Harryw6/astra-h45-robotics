"""Measured H45 frames -> native demo memory -> real GPT-Policy Astra review.

This adapter has no robot command client. Timestamp/units provenance comes from
the verified controller transport and clock calibration supplied by its caller.
"""
from copy import deepcopy
import base64
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time
import uuid

from PIL import Image

from .astra import run_review
from .contexts import _under_wam
from .data import write_json,file_sha256
from .memory_request import prepare_memory_review
from .runtime_contract import validate_request


def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid finite '+name)
    return float(value)


class LiveMemoryReviewJob:
    def __init__(self,memory,task_id,output_root,*,clock_snapshot,max_capture_age_s,max_clock_age_s,
                 model_timeout_s,effort,episode_id=None,clock=time.monotonic,recovery_reserve_s=5.,
                 codex_bin=None,multiview_inspection_root=None,reference_strategy='retrieved_block'):
        self.memory=memory;self.task_id=task_id;self.task=memory.task(task_id)
        self.root=Path(output_root).resolve()
        raw_root=json.loads((memory.library_root/'manifest.json').read_text(encoding='utf-8-sig')).get('source_root')
        if (_under_wam(self.root) or self.root.is_relative_to(memory.library_root)
                or memory.library_root.is_relative_to(self.root)
                or (raw_root and self.root.is_relative_to(Path(raw_root).resolve()))):
            raise ValueError('Live job output must be separate from raw data and demo library')
        for name,value in [('max_capture_age_s',max_capture_age_s),('max_clock_age_s',max_clock_age_s),
                           ('model_timeout_s',model_timeout_s),('recovery_reserve_s',recovery_reserve_s)]:
            if number(value,name)<=0:raise ValueError('Positive '+name+' required')
        if effort not in ('low','medium','high','xhigh','max'):raise ValueError('Unsupported Astra effort')
        self.clock=clock;self.clock_snapshot=clock_snapshot;self.max_age=max_capture_age_s
        self.max_clock_age=max_clock_age_s;self.model_timeout=model_timeout_s;self.reserve=recovery_reserve_s
        self.effort=effort;self.codex_bin=codex_bin;self.multiview_root=multiview_inspection_root
        self.reference_strategy=reference_strategy
        self.episode_id=episode_id or 'live_'+uuid.uuid4().hex
        if self.episode_id in {e['episode_id'] for e in memory.catalog()['episodes']}:
            raise ValueError('Live episode identity cannot be a recorded-library episode')

    def _clocks(self,now):
        values=deepcopy(self.clock_snapshot())
        for key in ('unitree','p450'):
            value=values[key]
            for name in ('offset_s','uncertainty_s','checked_host_s'):value[name]=number(value[name],name)
            if (value['uncertainty_s']<0 or not 0<=now-value['checked_host_s']<=self.max_clock_age):
                raise ValueError('Clock calibration is stale or invalid')
        if sum(values[k]['uncertainty_s'] for k in ('unitree','p450'))>.025:
            raise ValueError('Combined clock uncertainty exceeds H45 25 ms contract')
        return values

    def _bounds(self,stamp,device,clocks,now,*,fresh):
        stamp=number(stamp,'capture timestamp');cal=clocks[device]
        centre=stamp-cal['offset_s'];lower=centre-cal['uncertainty_s'];upper=centre+cal['uncertainty_s']
        if upper>now or (fresh and now-lower>self.max_age):
            raise ValueError('Current capture is stale or possibly in the future')
        return lower,upper

    def prepare(self,*,packet,control,event,deadline_s,history,candidate):
        control=validate_request(control);now=number(self.clock(),'host now')
        if number(deadline_s,'deadline')-now<=self.reserve:
            raise RuntimeError('No review budget remains before recovery reserve')
        if packet.get('session')!=control['session_id']:
            raise ValueError('Capture session does not match pending review')
        clocks=self._clocks(now)
        state=packet.get('state')
        if not isinstance(state,list) or len(state)!=17:raise ValueError('Expected measured Unitree state17')
        state=[number(v,'measured state') for v in state]
        self._bounds(packet['stamp'],'unitree',clocks,now,fresh=True)
        self._bounds(packet['p450_packet']['stamp'],'p450',clocks,now,fresh=True)
        entries=[];omitted=[]
        # At most one earlier policy observation plus the held analysis frame.
        samples=[(p,False) for p in history[-1:]]+[(packet,True)]
        for index,(sample,fresh) in enumerate(samples):
            if sample.get('session')!=control['session_id']:
                if fresh:raise ValueError('Foreign current capture session')
                omitted.append('foreign historical live session');continue
            for device,cameras,payload in (('unitree',('front','wrist'),sample),
                                           ('p450',('uav',),sample['p450_packet'])):
                for camera in cameras:
                    lower,upper=self._bounds(payload['image_stamps'][camera],device,clocks,now,fresh=fresh)
                    if not fresh and now-lower>5.:
                        omitted.append(camera+': past live frame older than five seconds');continue
                    raw=base64.b64decode(payload['images'][camera],validate=True)
                    with Image.open(BytesIO(raw)) as picture:
                        if picture.size!=(640,480) or picture.format not in ('PNG','JPEG'):
                            raise ValueError('Review capture must be actual PNG/JPEG 640x480')
                        extension='.png' if picture.format=='PNG' else '.jpg'
                        picture.verify()
                    entries.append(dict(frame_id='live_'+str(index)+'_'+camera,camera=camera,raw=raw,
                        extension=extension,time_s=upper,capture_lower_s=lower,capture_upper_s=upper,
                        source_timestamp_s=payload['image_stamps'][camera],device=device))
        identifier=control['request_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',identifier):raise ValueError('Unsafe review request path identity')
        run=self.root/identifier
        run.mkdir(parents=True,exist_ok=False)
        case=run/'case';case.mkdir()
        frames={};content=[
            'LIVE CONTROLLER REVIEW. The UAV provides elevated target observation; low front/wrist target occlusion '
            'can be normal. Source times use separate device monotonic clocks mapped to host bounds. '
            'Historical demonstration outcomes remain unlabeled. Only explicit runtime policy instructions may be returned; '
            'no numeric joint proposal or motor command can be executed by this model session.']
        for entry in entries:
            image=case/(entry['frame_id']+entry['extension']);image.write_bytes(entry['raw'])
            frame={k:v for k,v in entry.items() if k not in ('raw','extension','frame_id')}
            frame.update(path=str(image),image_sha256=file_sha256(image),evidence_role='current_observation',
                time_s_semantics='latest_possible_capture_in_host_monotonic',
                source_clock_domain=entry['device']+'_monotonic',clock_mapping=clocks[entry['device']])
            frames[entry['frame_id']]=frame
            content.extend([json.dumps(dict(frame_id=entry['frame_id'],**frame),ensure_ascii=False),
                            dict(image=str(image),label=entry['frame_id'])])
        request=dict(schema_version=1,case_id=identifier,current_episode_id=self.episode_id,
            phase='causal_review',decision_time_s=now,clock_domain='host_monotonic_capture_bounds',
            task_id=self.task_id,task_instruction=self.task['instruction'],input_manifest='input.json',
            available_frames=frames,runtime_control=control,
            observation=dict(source_episode=self.episode_id,observation_id=packet['observation_id'],
                state=state[10:17],state_joint_units='rad',state_gripper_units='m',
                state17=state,p450_state4=packet['p450_packet']['state4'],
                event=deepcopy(event),candidate_action_stream=deepcopy(candidate),physical_limits=None,
                task_ground_truth='unknown',clock_calibration=clocks,omitted_history=omitted,
                state_quality_notice='Measured transport state; readiness/limits are verified separately by the device. '
                    'Captured feedback and model judgments do not establish a physical grasp.'))
        if 'checkpoint_kinds' in event:
            from .action_checkpoints import make_request
            request['action_checkpoint']=make_request(event,candidate)
            content.append('This checkpoint additionally requests raw H45 numerical correction candidates. '
                'They are suggestions bound to the original candidate anchor, never executable motor commands. '
                'The legacy normalized joint_proposal must remain null.')
        write_json(case/'input.json',dict(model='gpt-6-astra',instruction=self.task['instruction'],content=content))
        write_json(case/'request.json',request)
        prepared=prepare_memory_review(self.memory,self.task_id,case/'request.json',run/'prepared',
                                       k=1,multiview_inspection_root=self.multiview_root,
                                       reference_strategy=self.reference_strategy)
        write_json(run/'JOB_STATUS.json',dict(status='prepared',request_id=identifier,session_id=control['session_id'],
            generation=control['generation'],deadline_host_s=deadline_s,actuator_calls=0,
            physical_grasp_verified=False))
        return {**prepared,'job_root':str(run)}

    def __call__(self,**kwargs):
        prepared=self.prepare(**kwargs);root=Path(prepared['job_root'])
        try:
            remaining=kwargs['deadline_s']-self.clock()-self.reserve
            if remaining<=0:raise RuntimeError('Preparation consumed remaining review budget')
            result=run_review(prepared['request_json'],root/'native_model',codex_bin=self.codex_bin,
                              effort=self.effort,timeout_s=min(self.model_timeout,remaining))
            if 'checkpoint_action_review' in result['validation']:
                write_json(root/'ACTION_REVIEW.json',result['validation']['checkpoint_action_review'])
            note=self.memory.append_review(root/'native_model',task_id=self.task_id)
            write_json(root/'JOB_STATUS.json',dict(status='completed',native_run=str(root/'native_model'),
                memory_note_id=note['note_id'],deadline_host_s=kwargs['deadline_s'],actuator_calls=0,
                physical_grasp_verified=False))
            if self.clock()+self.reserve>kwargs['deadline_s']:
                raise RuntimeError('Astra result arrived after policy recovery reserve')
            return {**result['validation'],'native_run':str(root/'native_model'),
                    'native_receipt':result['receipt'],'memory_note_id':note['note_id']}
        except Exception as exc:
            write_json(root/'JOB_STATUS.json',dict(status='failed',error=str(exc),native_run=str(root/'native_model'),
                usage_notice='Inspect actual native receipt and usage; failed or interrupted usage may be unknown',
                actuator_calls=0,physical_grasp_verified=False))
            raise
