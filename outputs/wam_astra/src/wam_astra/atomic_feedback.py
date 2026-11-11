"""Separate submitted intentions from measured state changes; no motor calls."""
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from .navigation_memory import displacement


def snapshot(packet):
    poses={};stamps={}
    try:
        # Same anchor implementation as the existing H45/RTC controller.
        from real_robot.goal_distance_realtime import control_anchor
        x,y,z,yaw=map(float,control_anchor(packet))
        poses['go2']=dict(position_x=x,position_y=y,position_z=z,yaw=yaw)
        # Observation/camera time may advance while the same DDS sample is reused.
        stamps['go2']=float(packet['state_stamps'][0])
    except (ValueError,KeyError,TypeError,IndexError):poses['go2']=None
    try:
        plane=packet['p450_packet'];pose=plane['pose']
        vals=[float(pose[k]) for k in ('x','y','z','yaw')]
        if not all(math.isfinite(v) for v in vals):raise ValueError('Invalid UAV pose')
        poses['uav']=dict(zip(['position.0','position.1','position.2','attitude.2'],vals))
        stamps['uav']=float(plane['stamp'])
    except (ValueError,KeyError,TypeError):poses['uav']=None
    return dict(observation_id=packet.get('observation_id'),poses=poses,stamps=stamps,
                waiting_for_prediction=packet.get('waiting_for_prediction'))


class ActionFeedback:
    def __init__(self,journal=None):
        self.journal=Path(journal) if journal else None
        self.pending=None;self.latest=None;self.sequence=0

    def _save(self,value):
        self.latest=deepcopy(value)
        if self.journal:
            self.journal.parent.mkdir(parents=True,exist_ok=True)
            with self.journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

    def submission(self,*,packet,action_source,decision,prepared,committed,inference_s,transport_s):
        if action_source!='astra_atomic':return
        self.sequence+=1
        accepted=bool(prepared.get('accepted') and committed and committed.get('accepted'))
        value=dict(sequence=self.sequence,decision=deepcopy(decision),
            plan_id=(committed or prepared).get('plan_id'),
            status='submitted' if accepted else 'not_submitted',
            prepared=deepcopy(prepared),committed=deepcopy(committed),
            arrival_confirmed=None,measured_change=None,before=snapshot(packet),
            latency_s=dict(inference=inference_s,prepare_commit=transport_s),
            interpretation='Submission is not arrival. Measured change can include external/manual motion or odometry resets.')
        self.pending=(deepcopy(value),time.monotonic()) if accepted else None
        self._save(value)

    def observe(self,packet):
        if self.pending is None:return
        value,submitted_at=self.pending
        value=deepcopy(value);after=snapshot(packet);before=value['before'];changes={}
        for robot in ('go2','uav'):
            a=before['stamps'].get(robot);b=after['stamps'].get(robot)
            fresh=(a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and b>a)
            changes[robot]=displacement(before['poses'].get(robot),after['poses'].get(robot),robot) if fresh else None
        selected=(value['decision'] or {}).get('robot')
        ready=changes.get(selected) is not None and after['waiting_for_prediction'] is True
        value.update(after=after,measured_change=changes,
                     status='measured_after_submission' if ready else 'awaiting_new_measurement')
        value['latency_s']['submit_to_observation']=time.monotonic()-submitted_at
        self._save(value)
        if ready:self.pending=None


def reference_identifiers(path):
    """Allow only IDs actually present in loaded historical manifest content."""
    manifest=json.loads(Path(path).read_text(encoding='utf-8-sig'));result=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('window_id','episode_id','title') and isinstance(v,str):result.add(v)
                elif isinstance(v,(dict,list)):collect(v)
        elif isinstance(value,list):
            for item in value:collect(item)
    for part in manifest.get('content',[]):
        if isinstance(part,str):
            try:collect(json.loads(part))
            except json.JSONDecodeError:pass
    return result
