"""Live three-view H45 input and the existing paired-ACK action transport."""
import argparse
import base64
import concurrent.futures
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .controller import Device, StreamHeartbeat
from .goal_distance_realtime import RemotePolicy, SCHEMA, control_anchor, save
from .rtc_h45 import ArrivalPrefixEstimator, empty_context, pipeline_prefix_context, PrefixCoverageError
from .three_robot import AircraftClient, ThreeRobotDevice, GROUND_SITE_CONFIG, P450_ACTIONS
from .review_errors import ReviewSuperseded

SLOTS = {'front':'base_0_rgb', 'wrist':'left_wrist_0_rgb', 'uav':'right_wrist_0_rgb'}


class OperatorCompleted(Exception):
    """An explicit local operator completion request, not model success detection."""


def model_observation(dog, plane, prompt):
    """Use measured heading velocities and measured joint angles, never recordings."""
    state = np.asarray(dog['state'], dtype=float)
    uav = np.asarray(plane['state4'], dtype=float)
    if state.shape != (17,) or uav.shape != (4,) or not np.isfinite(np.r_[state,uav]).all():
        raise ValueError('Invalid live robot state dimensions or values')
    # Sensors.History.motion already rotates velocity into current heading.
    model_state = np.asarray([*state[7:10],*state[10:17],*uav],dtype=np.float32)
    images = {}
    for packet, names in ((dog,('front','wrist')),(plane,('uav',))):
        for name in names:
            raw = base64.b64decode(packet['images'][name], validate=True)
            bgr = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (480,640,3):
                raise ValueError('Live image must be 640x480: '+name)
            images[SLOTS[name]] = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return dict(state=model_state,image=images,
                image_mask={k:np.bool_(True) for k in images},prompt=prompt)


class LiveDevice(ThreeRobotDevice):
    live_input = True

    def capture_plane(self):
        announced = False
        while True:
            plane = self.aircraft.call('observe_live')
            if not plane.get('waiting_for_image'):
                return plane
            if not announced:
                print(json.dumps(dict(event='waiting_for_p450_image'), ensure_ascii=False), flush=True)
                announced = True
            time.sleep(.1)

    def capture(self, diagnostic=False):
        # Wait for both packets. Do not re-anchor at the time replies arrive.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            dog = pool.submit(Device.call,self,'/diagnostic' if diagnostic else '/observation')
            plane = pool.submit(self.capture_plane)
            dog,plane = dog.result(),plane.result()
        self.snapshot = plane
        return {**dog,'p450_anchor':plane['pose'],'p450_packet':plane}


def save_capture(folder, packet, obs):
    folder.mkdir(parents=True,exist_ok=True)
    for slot,rgb in obs['image'].items():
        if not cv2.imwrite(str(folder/(slot+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):
            raise RuntimeError('Could not save live image')
    plane=packet['p450_packet']
    save(folder/'states.json',dict(input_source='live_three_view_and_measured_state',
        model_state14=obs['state'].tolist(),prompt=obs['prompt'],
        go2_state=packet['state'][:10],piper_state=packet['state'][10:17],
        p450_state={k:v for k,v in plane.items() if k!='images'},
        unitree_stamp=packet['stamp'],image_stamps=packet.get('image_stamps'),
        state_stamps=packet.get('state_stamps'),observation_id=packet.get('observation_id')))


def execute_live(device, policy, prompt, duration, output, *, review_runtime=None):
    if not math.isfinite(duration) or not 0 <= duration <= 300:
        raise ValueError('Live duration must be 0 (operator completion) or in (0,300]')
    continuous = duration == 0
    if review_runtime is not None and not continuous:
        raise ValueError('Review runtime currently requires continuous duration=0; finite review budgets are not implemented')
    site = device.aircraft.call('status')
    if continuous and not site.get('live_continuous_supported'):
        raise RuntimeError('Deploy continuous live support to P450 first')
    if not continuous and duration > site.get('max_duration_s', 0):
        raise ValueError('Live duration exceeds approved site duration')
    if continuous and 'h45_live_continuous_v1' not in device.call('/status').get('capabilities', []):
        raise RuntimeError('Deploy continuous live support to Go2 first')
    device.wait_for_operator_ready()
    # Live execution starts only after manual takeoff; no preloaded recorded takeoff.
    status=device.aircraft.call('ground_wait_status')
    if status.get('landed') != 2:
        raise RuntimeError('Live mode requires the aircraft already airborne')
    reset=device.call('/reset',{})
    heart=StreamHeartbeat({'unitree':device},{'unitree':reset['session']})
    plans=[]; estimator=ArrivalPrefixEstimator(2.); end=None; n=0
    completion_source = 'duration_budget'
    policy_prompt=prompt
    def check_wait():
        if (output/'operator_completed.request').is_file():
            raise OperatorCompleted()
        heart.check()
        if review_runtime is not None:review_runtime.check()
    try:
        heart.start()
        if review_runtime is not None:review_runtime.start()
        while True:
            check_wait()
            recovered=None
            if review_runtime is not None and review_runtime.pending:
                recovered=review_runtime.recover(check_wait)
                if recovered is not None:
                    plans=[];end=None;estimator=ArrivalPrefixEstimator(2.)
                    policy_prompt=prompt+'\nAstra correction for the next fresh policy plan: '+recovered['policy_instruction']
                    checkpoint=recovered['review'].get('checkpoint_action_review')
                    if checkpoint and checkpoint.get('replacements'):
                        policy_prompt+='\nAstra suggested changes to the previous H45 proposal: '+json.dumps(
                            checkpoint['replacements'],ensure_ascii=False,separators=(',',':'))
                        policy_prompt+=('\nThese vectors refer to the previous observation anchor; '
                            'plan again from the new observation rather than treating them as new-anchor commands.')
            status=device.call('/status')
            if status.get('fault') or status.get('hardware_fault'):
                raise RuntimeError(str(status.get('fault') or status.get('hardware_fault')))
            stream=status.get('stream') or {}
            if stream.get('finished'):
                if continuous:
                    raise RuntimeError('Continuous stream ended without operator completion')
                device.call('/stream/finish',{});break
            # Atomic positioning is step-and-observe. Waiting is handled by the
            # existing native stream hold, never by emergency-stop/reset.
            if (plans and getattr(policy,'wait_for_chunk_end',False)
                    and not stream.get('waiting_for_prediction')):
                time.sleep(.02)
                continue
            generation=(recovered['generation'] if recovered is not None else
                        device.review_generation if review_runtime is not None else None)
            packet=recovered['packet'] if recovered is not None else device.capture()
            if review_runtime is not None and review_runtime.superseded(generation):continue
            obs=model_observation(packet,packet['p450_packet'],policy_prompt)
            if hasattr(policy,'observe_packet'):policy.observe_packet(packet)
            if review_runtime is not None:review_runtime.remember(packet)
            resume=bool(packet.get('waiting_for_prediction')) and bool(plans)
            context=empty_context()
            if plans and not resume:
                if packet['stamp'] <= plans[-1]['origin_at']:
                    time.sleep(.01);continue
                try:
                    context=pipeline_prefix_context(plans,packet['stamp'],control_anchor(packet),
                          obs['state'],estimator.steps,allow_short_prefix=True)
                except PrefixCoverageError:
                    time.sleep(.01);continue
            folder=output/('observation_%06d'%n)
            n+=1
            save_capture(folder,packet,obs)
            feedback=getattr(policy,'last_execution_feedback',None)
            if feedback is not None:save(folder/'execution_feedback.json',feedback)
            inference_started=time.perf_counter()
            corrected=recovered.get('recovery_actions') if recovered is not None else None
            if corrected is not None:
                actions=np.asarray(corrected,dtype=float)
                action_source='astra_reviewed_reanchored'
            else:
                actions=policy.infer_prepared(policy.prepare_input(obs,rtc=context),timeout=.25,check_wait=check_wait)
                action_source=getattr(policy,'last_action_source','h45_policy')
            inference_finished=time.perf_counter()
            check_wait()
            if actions is None:
                save(folder/'decision.json',dict(decision=getattr(policy,'last_decision',None),
                     phase=getattr(policy,'phase',None),actuator_commands_sent=False,
                     handoff_assessment=getattr(policy,'last_handoff_assessment',None),
                     elapsed_s=inference_finished-inference_started))
                continue
            if review_runtime is not None and review_runtime.superseded(generation):
                save(folder/'discarded.json',dict(reason='review_superseded_policy_reply',generation=generation))
                continue
            if review_runtime is not None and review_runtime.on_prediction(packet,actions,generation):
                save(folder/'discarded.json',dict(reason='review_event_before_prepare',generation=generation))
                continue
            body=dict(schema_id=SCHEMA,session=reset['session'],observation_id=packet['observation_id'],
                      origin_at=packet['stamp'],actions=actions.tolist(),model_state=obs['state'].tolist(),
                      execution_scope='unitree',rtc_arrival=True,live_continuous=continuous)
            if review_runtime is not None:body['review_generation']=generation
            if not plans:
                body.update(bootstrap=True,remaining_video_s=duration,wait_on_exhaustion=True,rtc_pipeline=True)
            else:
                body['end_at']=end
                if resume: body['resume_from_wait']=True
                else: body['rtc_source_plan_id']=plans[-1]['plan_id']
            try:
                transport_started=time.perf_counter()
                prepared=device.call('/stream/prepare',body)
                check_wait()
                if review_runtime is not None and review_runtime.superseded(generation):
                    save(folder/'discarded.json',dict(reason='review_superseded_preparation',generation=generation))
                    continue
                commit_body={'plan_id':prepared['plan_id']} if prepared.get('accepted') else None
                if commit_body is not None and review_runtime is not None:commit_body['review_generation']=generation
                committed=device.call('/stream/commit',commit_body) if commit_body is not None else None
                transport_finished=time.perf_counter()
            except ReviewSuperseded:
                if review_runtime is None or not review_runtime.superseded(generation):raise
                save(folder/'discarded.json',dict(reason='review_cancelled_old_transport',generation=generation))
                continue
            save(folder/'prediction.json',dict(actions=actions.tolist(),prepared=prepared,committed=committed,
                                              input_source='live',action_source=action_source,
                                              inference_or_correction_s=inference_finished-inference_started,
                                              prepare_commit_s=transport_finished-transport_started,
                                              rtc_prefix_steps=context['delay_steps']))
            if hasattr(policy,'record_execution'):
                policy.record_execution(packet=packet,action_source=action_source,
                    decision=getattr(policy,'last_decision',None),prepared=prepared,committed=committed,
                    inference_s=inference_finished-inference_started,transport_s=transport_finished-transport_started)
            if committed and committed.get('accepted'):
                if review_runtime is not None and not review_runtime.note_commit(generation):continue
                end=committed['end_at']
                plans.append(dict(actions=actions,state=obs['state'],anchor=control_anchor(packet),
                                  origin_at=committed['origin_at'],switch_at=committed['scheduled_start'],
                                  plan_id=committed['plan_id']))
                plans=plans[-12:]
                # Astra positioning waits for exhaustion before observing again;
                # its cloud latency is not a sample of Pi's pipelined inference.
                if action_source != 'astra_atomic':
                    estimator.observe(max(1/15,committed['scheduled_start']-packet['stamp']+1/15))
            print(json.dumps(dict(live=True,predictions=n,paired=device.paired)),flush=True)
    except OperatorCompleted:
        completion_source = 'operator_confirmed'
        device.abort()
    except BaseException:
        device.abort()
        raise
    finally:
        if review_runtime is not None:review_runtime.close()
        heart.close()
    return dict(complete=completion_source == 'operator_confirmed', session_ended=True,
                completion_source=completion_source, paired_commits=device.paired,
                observations=n,p450_stop=device.plane_stop)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('observe','infer','execute'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--duration-s',type=float,default=0.,help='0: run until operator confirms completion')
    p.add_argument('--checkpoint-step',type=int,default=20000);p.add_argument('--policy-url')
    p.add_argument('--p450-host',default='P450_HOST_REQUIRED');p.add_argument('--p450-known-host');p.add_argument('--p450-key')
    p.add_argument('--ask-p450-password',action='store_true');p.add_argument('--site-confirmed',action='store_true')
    p.add_argument('--site-config',default=GROUND_SITE_CONFIG);p.add_argument('--tracking-error-m',type=float,default=.5)
    a=p.parse_args()
    if a.mode=='execute' and not a.site_confirmed:p.error('Execution requires --site-confirmed')
    if not 0<a.tracking_error_m<=.5:p.error('Tracking error must be in (0,.5]')
    a.ground_start=True;a.no_speed_limit=True;a.live_observations=True
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));a.output.mkdir(parents=True,exist_ok=False)
    aircraft=AircraftClient(a,a.output/'p450.jsonl')
    device=LiveDevice(cfg['unitree'],a.token_file.read_text().strip(),aircraft,ground_start=True,no_speed_limit=True)
    policy=None
    try:
        status=device.calibrate()
        if not status.get('live_camera_capture'):raise RuntimeError('Enable h45_live_observations on dog gateway first')
        if a.mode != 'execute' and status.get('execution_enabled'):
            raise RuntimeError('Observe/infer requires an observe-only dog gateway')
        if a.mode=='execute':
            policy=RemotePolicy(a.policy_url or cfg['policy'],checkpoint_step=a.checkpoint_step)
            if (policy.metadata.get('action_names',[])[10:]!=P450_ACTIONS
                    or policy.metadata.get('uav_encoding')!='target_pose_relative_to_measured_chunk_start'
                    or policy.metadata.get('uav_xy_frame')!='chunk_start_yaw_forward_left'
                    or policy.metadata.get('fps')!=15):
                raise ValueError('UAV action schema mismatch')
            save(a.output/'summary.json',execute_live(device,policy,a.prompt,a.duration_s,a.output))
        else:
            # Observe-only gateway: initialize coordinates without actuator commands.
            Device.call(device, '/reset', {})
            packet=device.capture(diagnostic=True);obs=model_observation(packet,packet['p450_packet'],a.prompt)
            save_capture(a.output,packet,obs)
            if a.mode=='infer':
                policy=RemotePolicy(a.policy_url or cfg['policy'],checkpoint_step=a.checkpoint_step)
                actions=policy.infer_prepared(policy.prepare_input(obs,rtc=empty_context()),timeout=None)
                save(a.output/'prediction.json',dict(actions=actions.tolist(),robot_commands_sent=False))
            print(json.dumps(dict(output=str(a.output),robot_commands_sent=False)),flush=True)
    finally:
        device.stop_plane();aircraft.close()
        if policy:policy.close()

if __name__=='__main__':main()
