"""Local review-hold feedback checks. No actuator, enable or model calls here.

Configuration values must be supplied from verified device evidence. Passing
these software checks is not proof of physical safety, grasp or task outcome.
Live action units remain base velocities, joint radians and gripper metres.
"""
import math

import numpy as np

from .core import angle_delta, finite


def settings(value):
    if not isinstance(value, dict) or not (value.get('calibration_verified') is True
            or value.get('configuration_source') == 'h45_runtime_v1'):
        raise ValueError('Review pause requires explicit runtime settings and device limits')
    result = dict(value)
    required=value.get('require_deadline_binding',True)
    if type(required) is not bool:
        raise ValueError('require_deadline_binding must be boolean')
    result['require_deadline_binding']=required
    for name in ('max_feedback_age_s', 'monitor_period_s', 'worker_join_timeout_s',
                 'max_review_s', 'go2_min_body_height_m'):
        v = value.get(name)
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Explicit positive finite '+name+' required')
    if value['monitor_period_s'] >= value['max_feedback_age_s']:
        raise ValueError('monitor_period_s must be shorter than max_feedback_age_s')
    codes = value.get('go2_normal_state_codes')
    if (not isinstance(codes, list) or not codes
            or any(type(code) is not int or not 0 <= code <= 0xffffffff for code in codes)):
        raise ValueError('Explicit go2_normal_state_codes required')
    duration = value.get('min_hold_duration_s')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError('Explicit nonnegative min_hold_duration_s required')
    if type(value.get('min_hold_samples')) is not int or value['min_hold_samples'] < 2:
        raise ValueError('min_hold_samples must be at least two distinct feedback samples')
    result.pop('go2_pose_drift', None)  # Retired adapter-only stop threshold.
    result.pop('arm_drift', None)
    result.pop('arm_rate', None)
    for name, size in [('go2_velocity_tolerance', 3)]:
        arr = finite(value.get(name), (size,))
        if np.any(arr <= 0):
            raise ValueError('Explicit positive '+name+' required')
        result[name] = arr
    result['joint_limits'] = finite(value.get('joint_limits'), (6, 2))
    result['gripper_limits'] = finite(value.get('gripper_limits'), (2,))
    if (np.any(result['joint_limits'][:, 0] >= result['joint_limits'][:, 1])
            or result['gripper_limits'][0] >= result['gripper_limits'][1]):
        raise ValueError('Invalid explicit physical limits')
    return result


def check_arm(arm, config):
    arm = finite(arm, (7,))
    limits = config['joint_limits']
    if (np.any(arm[:6] < limits[:, 0]) or np.any(arm[:6] > limits[:, 1])
            or not config['gripper_limits'][0] <= arm[6] <= config['gripper_limits'][1]):
        raise RuntimeError('Measured/target arm is outside verified limits; no clipping allowed')
    return arm


def _stamp(value, now, config):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RuntimeError('Invalid feedback timestamp')
    if not 0 <= now-value <= config['max_feedback_age_s']:
        raise RuntimeError('Missing, future or stale review feedback')
    return value


def base_sample(base, now, config):
    if base.get('valid') is not True:
        raise RuntimeError('Invalid Go2 measured state')
    stamp_ns = base.get('monotonic_ns')
    if type(stamp_ns) is not int or stamp_ns <= 0:
        raise RuntimeError('Invalid Go2 monotonic_ns')
    stamp = _stamp(stamp_ns/1e9, now, config)
    pose = finite([*base['position'], base['yaw']], (4,))
    velocity = finite([*base['velocity'][:2], base['yaw_speed']], (3,))
    if (config.get('configuration_source') != 'h45_runtime_v1'
            and np.any(np.abs(velocity) > config['go2_velocity_tolerance'])):
        raise RuntimeError('Go2 measured velocity exceeds hold tolerance')
    return stamp, pose


class HoldWindow:
    def __init__(self, config, stop_response, initial, now):
        self.config = config
        self.stop_response = stop_response
        # A STOP acknowledgement is only a request receipt. It is not a sample
        # counted toward the post-stop stable window.
        self.stop_stamp, _ = base_sample(stop_response, now, config)
        self.initial = self._sample(initial, now)
        self.previous = self.initial
        self.latest = self.initial
        self.since = None
        self.count = 0
        self.ready = False
        self.last_feedback = initial
        self.reason = 'Awaiting distinct post-STOP measured feedback'

    def _sample(self, frame, now):
        if frame.get('readiness_verified') is not True:
            raise RuntimeError('Arm enable/calibration readiness not verified')
        stamp, pose = base_sample(frame['base'], now, self.config)
        snapshot_stamp, snapshot_pose = base_sample(frame['go2_snapshot'], now, self.config)
        arm = check_arm(frame['arm'], self.config)
        arm_stamp = _stamp(frame['arm_stamp'], now, self.config)
        return dict(stamps=(stamp, snapshot_stamp, arm_stamp), pose=pose,
                    snapshot_pose=snapshot_pose, arm=arm)

    def update(self, frame, now):
        sample = self._sample(frame, now)
        self.last_feedback = frame
        if any(a < b for a, b in zip(sample['stamps'], self.latest['stamps'])):
            raise RuntimeError('Feedback timestamp moved backwards')
        for index,key in enumerate(('pose','snapshot_pose','arm')):
            if (sample['stamps'][index] == self.latest['stamps'][index]
                    and not np.array_equal(sample[key],self.latest[key])):
                raise RuntimeError('Feedback values changed without a new timestamp')
        if min(sample['stamps']) <= self.stop_stamp:
            self.ready = False
            self.reason = 'Awaiting feedback newer than STOP state'
            return
        for key in ('pose', 'snapshot_pose'):
            delta = sample[key]-self.initial[key]
            delta[3] = angle_delta(self.initial[key][3], sample[key][3])
            # Retain measured feedback; do not stop on the retired drift threshold.
        self.latest=sample
        if self.config.get('configuration_source') == 'h45_runtime_v1':
            stationary=all(np.all(np.abs([*frame[key]['velocity'][:2],frame[key]['yaw_speed']])
                <= self.config['go2_velocity_tolerance']) for key in ('base','go2_snapshot'))
            if not stationary:
                self.ready=False;self.since=None;self.count=0
                self.reason='STOP acknowledged; waiting for measured deceleration'
                return
        if not all(a>b for a,b in zip(sample['stamps'],self.previous['stamps'])):
            # Checking the same still-fresh sample never extends the measured
            # window, but must not erase a valid window at the commit boundary.
            self.reason='Checked fresh feedback; waiting for all channels to advance'
            return
        self.previous = sample
        stamp = min(sample['stamps'])
        if self.since is None:
            self.since = stamp
        self.count += 1
        self.ready = (self.count >= self.config['min_hold_samples']
                      and stamp-self.since >= self.config['min_hold_duration_s'])
        self.reason = 'Measured window passed configured checks' if self.ready else 'Collecting stable feedback'

    def summary(self):
        return dict(ready=self.ready, reason=self.reason, samples=self.count,
                    stable_since=self.since, stop_response=self.stop_response,
                    latest_feedback=self.last_feedback, physical_hold_proven=False,
                    task_outcome='unknown')
