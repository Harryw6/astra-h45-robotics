"""Explicit opt-in wiring into H45's existing runtime callbacks.

No camera anchors, model sockets or actuator commands are created here.
"""
from copy import deepcopy
import time
import threading
import numpy as np
from .action_checkpoints import CheckpointMonitor,make_request,validate_answer,candidate_digest,number


class CheckpointHooks:
    def __init__(self,device,config,*,clock=time.monotonic):
        if not isinstance(config,dict) or not (config.get('calibration_verified') is True
                or config.get('configuration_source') == 'h45_runtime_v1'):
            raise ValueError('Explicit checkpoint runtime thresholds required')
        self.device=device
        self.monitor=CheckpointMonitor(**{k:v for k,v in config.items() if k not in ('calibration_verified','configuration_source','recovery')})
        self.clock=clock;self.lock=threading.RLock();self.recovery_permit=None
        self.recovery=deepcopy(config.get('recovery'))
        if self.recovery is not None:
            if number(self.recovery['lifetime_s'],'permit lifetime')<=0:raise ValueError('Positive permit lifetime required')
            for key,size in [('action_tolerance',14),('state_tolerance',17),('pose_tolerance',4)]:
                values=self.recovery[key]
                if not isinstance(values,list) or len(values)!=size or any(number(v,key)<0 for v in values):
                    raise ValueError('Explicit finite nonnegative '+key+' required')

    def reviewed(self,result,*,candidate,event,generation):
        with self.lock:
            self.recovery_permit=None
            value=result.get('checkpoint_action_review')
            if self.recovery is None or not value or value.get('decision')=='uncertain':return
            keys=('checkpoint_id','candidate_sha256','decision','reason','evidence_frame_ids','replacements','cannot_correct_reason')
            answer=validate_answer({k:value[k] for k in keys},make_request(event,candidate))
            if answer['decision']=='invalid' and answer['corrected_actions'] is None:return
            target=answer['corrected_actions'] if answer['decision']=='invalid' else candidate['model_actions']
            self.recovery_permit=dict(target=deepcopy(target),anchor=deepcopy(candidate['checkpoint_anchor']),
                generation=generation,expires=self.clock()+self.recovery['lifetime_s'],checked_digest=None,
                committed=candidate.get('committed',True))

    def recovery_actions(self,packet,generation):
        """Re-anchor an unexecuted reviewed chunk; never replay committed history.

        H45's native align_plan handles the base and Piper. UAV targets use
        the same yaw-relative displacement convention as the aircraft bridge.
        Review latency is a pause, not elapsed trajectory playback time.
        """
        from importlib import import_module
        with self.lock:
            p=self.recovery_permit
            if p is None or p['committed']:return None
            package=self.device.__class__.__module__.rsplit('.',1)[0]
            align=import_module(package+'.rtc_h45').align_plan
            anchor_of=import_module(package+'.goal_distance_realtime').control_anchor
            if generation!=p['generation'] or self.clock()>=p['expires']:
                raise RuntimeError('Reviewed correction expired or changed generation')
            old=p['anchor'];old_pose=old['p450_packet']['pose'];pose=packet['p450_packet']['pose']
            old_state=np.asarray(old['state'],float);state=np.asarray(packet['state'],float)
            drift=state-old_state
            pose_drift=np.array([pose[k]-old_pose[k] for k in ('x','y','z','yaw')])
            pose_drift[3]=np.arctan2(np.sin(pose_drift[3]),np.cos(pose_drift[3]))
            if (not np.isfinite(drift).all() or not np.isfinite(pose_drift).all()
                    or np.any(np.abs(drift)>self.recovery['state_tolerance'])
                    or np.any(np.abs(pose_drift)>self.recovery['pose_tolerance'])):
                raise RuntimeError('Scene moved beyond the reviewed correction context')
            target=np.asarray(p['target'],float)
            actions,_=align(target,0.,0.,anchor_of(old),anchor_of(packet),
                            old_state[7:17],state[7:17])
            c,s=np.cos(old_pose['yaw']),np.sin(old_pose['yaw'])
            dx=old_pose['x']+c*target[:,10]-s*target[:,11]-pose['x']
            dy=old_pose['y']+s*target[:,10]+c*target[:,11]-pose['y']
            c,s=np.cos(pose['yaw']),np.sin(pose['yaw'])
            actions[:,10]=c*dx+s*dy;actions[:,11]=-s*dx+c*dy
            actions[:,12]=old_pose['z']+target[:,12]-pose['z']
            yaw=old_pose['yaw']+target[:,13]-pose['yaw']
            actions[:,13]=np.arctan2(np.sin(yaw),np.cos(yaw))
            p['target']=actions.tolist();p['anchor']=deepcopy(dict(state=packet['state'],p450_packet=packet['p450_packet']))
            return actions

    def _permits(self,packet,candidate):
        with self.lock:
            p=self.recovery_permit
            if p is None:return False
            p['checked_digest']=None
            if self.clock()>=p['expires'] or candidate['generation']!=p['generation']:return False
            old=p['anchor'];pose=packet['p450_packet']['pose'];old_pose=old['p450_packet']['pose']
            for actual,expected,key in [(candidate['model_actions'],p['target'],'action_tolerance'),
                (packet['state'],old['state'],'state_tolerance'),
                ([pose[k] for k in ('x','y','z','yaw')],[old_pose[k] for k in ('x','y','z','yaw')],'pose_tolerance')]:
                a,b=np.asarray(actual,float),np.asarray(expected,float)
                if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(np.abs(a-b)>self.recovery[key]):return False
            p['checked_digest']=candidate_digest(candidate)
            return True

    def check(self):
        with self.lock:
            p=self.recovery_permit
            if p is not None and p['checked_digest'] is not None and self.clock()>=p['expires']:
                raise RuntimeError('Checkpoint recovery permit expired before paired commit')

    def __call__(self,packet,actions,generation):
        candidate=dict(observation_id=packet['observation_id'],generation=generation,
                       origin_at=packet['stamp'],model_actions=actions.tolist())
        if self._permits(packet,candidate):return None
        return self.monitor.prediction(candidate,gripper_m=packet['state'][16])

    def monitor_source(self):
        sample=self.device.monitor_status()
        plane=sample.get('p450',{})
        if self.monitor.rotation is not None and not self.monitor.rotation['finished']:
            if 'feedback' not in plane:raise RuntimeError('P450 measured rotation feedback capability missing')
            event=self.monitor.feedback(plane['feedback'],now_s=plane['monotonic'])
            if event is not None:sample['event']=event
        return sample

    def committed(self,candidate):
        self.check()
        with self.lock:
            p=self.recovery_permit
            if p is not None and p['checked_digest']==candidate_digest(candidate):self.recovery_permit=None
        anchor=candidate['checkpoint_anchor'];plane=anchor['p450_packet'];pose=plane['pose']
        self.monitor.track_commit(dict(stamp=plane['stamp'],source_stamp=plane['state_ros_stamp'],
            pose=[pose[k] for k in ('x','y','z','yaw')],velocity=plane['state4']),
            candidate['model_actions'],plan_id=candidate['observation_id'])


def checkpoint_runtime(device,job,config,**runtime_options):
    """Create runtime via the already-imported device's native client package.

    Caller must still perform normal device/site/deadline preflight. No start,
    reset, ROS publisher or enabling action occurs in this function.
    """
    from importlib import import_module
    hooks=CheckpointHooks(device,config)
    package=device.__class__.__module__.rsplit('.',1)[0]
    runtime=import_module(package+'.review_runtime').ReviewRuntime
    return runtime(device,job,monitor_source=hooks.monitor_source,prediction_gate=hooks,**runtime_options)
