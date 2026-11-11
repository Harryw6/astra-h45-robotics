"""Atomic body-frame displacement adapter for the existing H45 paired SDK bridge.

H45 uses 45 samples at 15 Hz. Go2 deltas are differentiated by the existing
goal_distance_gateway.decode_goals; UAV deltas go to the existing P450 bridge.
This module does not start hardware, replace SDKs, or infer metric calibration.
"""
import numpy as np

from .atomic_handoff import evaluate_handoff_evidence


class ObservationRequired(RuntimeError):
    """Planner has not selected movement; caller must keep waiting/observing."""


def atomic_chunk(action, state):
    if action['decision'] in ('observe','hold'):
        return None
    if action['decision'] != 'move':
        raise ValueError('Unknown atomic decision')
    state=np.asarray(state,dtype=float)
    if state.shape!=(14,) or not np.isfinite(state).all():
        raise ValueError('Expected current H45 state14')
    distance=action['distance_m']
    if isinstance(distance,bool) or not isinstance(distance,(int,float)) or not np.isfinite(distance) or distance<=0:
        raise ValueError('Movement requires a positive finite distance in metres')
    axis,sign={'forward':(0,1),'backward':(0,-1),'left':(1,1),'right':(1,-1),'up':(2,1),'down':(2,-1)}[action['direction']]
    if axis==2 and action['robot']!='uav':
        raise ValueError('Vertical movement is only defined for UAV')
    offset={'go2':0,'uav':10}[action['robot']]
    chunk=np.zeros((45,14),dtype=float)
    chunk[:,9]=state[9]  # absolute aperture; joint deltas remain zero
    chunk[:,offset+axis]=np.arange(1,46)*distance*sign/45
    return chunk


class HybridPolicy:
    """RemotePolicy-compatible policy for native live_h45.execute_live.

    planner receives current image/state observation and returns atomic JSON.
    begin_grasp is an explicit phase transition; no fabricated vision threshold.
    Original Pi transport and paired prepare/commit remain owned by H45.
    """
    def __init__(self,planner,pi):
        self.planner=planner
        self.pi=pi
        self.phase='positioning'
        self.last_decision=None
        self.last_action_source='astra_atomic'
        self.last_handoff_assessment=None

    @property
    def wait_for_chunk_end(self):
        return self.phase=='positioning'

    def observe_packet(self,packet):
        if hasattr(self.planner,'observe_packet'):
            self.planner.observe_packet(packet)

    def record_execution(self,**receipt):
        if hasattr(self.planner,'record_execution'):
            self.planner.record_execution(**receipt)

    @property
    def last_execution_feedback(self):
        return getattr(self.planner,'last_execution_feedback',None)

    def begin_grasp(self):
        self.phase='grasp'

    def prepare_input(self,observation,rtc=None):
        pi_input=None
        if self.phase=='grasp':
            # Astra provenance/telemetry are outside the original Pi serving contract.
            pi_observation={key:observation[key] for key in ('state','image','image_mask','prompt')
                            if key in observation}
            pi_input=self.pi.prepare_input(pi_observation,rtc=rtc)
        return self.phase,observation,pi_input

    def infer_prepared(self,prepared,timeout=120,*,check_wait=None):
        phase,observation,pi_input=prepared
        if phase!=self.phase:
            raise RuntimeError('Discard input prepared before phase transition')
        if check_wait:check_wait()
        if phase=='grasp':
            self.last_action_source='pi_original_h45'
            chunk=np.asarray(self.pi.infer_prepared(pi_input,timeout=timeout,check_wait=check_wait),dtype=float).copy()
            if chunk.shape!=(45,14) or not np.isfinite(chunk).all():
                raise ValueError('Invalid Pi action chunk')
            return chunk
        action=(self.planner.decide(observation,check_wait=check_wait)
                if hasattr(self.planner,'decide') else self.planner(observation))
        self.last_decision=action
        self.last_action_source='astra_atomic'
        if check_wait:check_wait()
        if action['decision']=='handoff_pi':
            self.last_handoff_assessment=evaluate_handoff_evidence(
                action.get('handoff_evidence'),
                allowed_reference_ids=getattr(self.planner,'reference_ids',None))
            if hasattr(self.planner,'record_handoff_assessment'):
                self.planner.record_handoff_assessment(self.last_handoff_assessment)
            if self.last_handoff_assessment['ready']:
                self.begin_grasp()
            return None  # Pi must receive a new measured observation next iteration.
        chunk=atomic_chunk(action,observation['state'])
        return chunk
