"""Explicit H45 goal-distance adapter for the existing Unitree prepare/commit service."""
import argparse
import json
import math
from pathlib import Path
import signal
import threading
import time
import uuid
from http.server import ThreadingHTTPServer

import numpy as np

from .core import DT, action_plan, angle_delta, command_limits, finite, heading, piper_sdk_target
from .gateway import Gateway, handler

SCHEMA_ID = 'heterovla_v3_goal_distance_h45_masked_v1'
TRANSFORM = 'goal_h45_displacement_and_direct_piper_targets_v2'
CAPABILITY = 'goal_distance_h45_stream_v1'
EXPIRY_CAPABILITY = 'stale_prediction_drop_v1'
BOOTSTRAP_CAPABILITY = 'preloaded_initial_h45_v1'
REUSE_CAPABILITY = 'normal_completion_reuse_v1'


def decode_goals(actions, arm_anchor, velocity_limits=None, execute_piper=True):
    """Decode cumulative targets, never treating their values as velocities.

    Difference consecutive cumulative targets and divide by DT. Execute the
    predicted XY displacement as given, without rescaling. XY velocities remain
    in the chunk-start frame until each dispatch rotates them by measured yaw.
    Piper joint deltas are restored against the state used by the model. Live
    feedback independently gates execution and initializes the joint tracker.
    """
    actions = finite(actions, (45, 14))
    anchor = finite(arm_anchor, (7,))
    previous = np.vstack((np.zeros((1, 3)), actions[:-1, :3]))
    chunk_velocity = (actions[:, :3] - previous) / DT
    chunk_velocity[:, 2] = np.arctan2(np.sin(actions[:, 2] - previous[:, 2]),
                                     np.cos(actions[:, 2] - previous[:, 2])) / DT
    rows = np.column_stack((chunk_velocity, actions[:, 3:9] + anchor[:6], actions[:, 9]))
    # Rotation can change component limits; validate both a nominal path here
    # and the measured-heading command immediately before every hardware send.
    nominal = rows.copy()
    for i in range(45):
        nominal[i, :2] = heading(chunk_velocity[i, :2], previous[i, 2])
    if np.any(np.abs(nominal[:, :3]) > command_limits('unitree', velocity_limits)):
        raise ValueError('Go2 decoded velocity exceeds execution limits')
    if execute_piper:
        for row in rows:
            validate_piper_encoding(row[3:])
    return rows


def validate_piper_encoding(arm):
    """Preserve targets; check only finite values and signed CAN integer encoding."""
    joints, gripper = piper_sdk_target(arm)
    if any(not -(2**31) <= value < 2**31 for value in joints+[gripper]):
        raise ValueError('Piper target exceeds signed 32-bit CAN encoding; raw target was not changed')


def validate_h45_dispatch(row, velocity_limits):
    action_plan(row[None, :3], 'go2', velocity_limits=velocity_limits)
    validate_piper_encoding(row[3:])


class GoalGateway(Gateway):
    def __init__(self, *args, execution_scope='unitree', prediction_max_age=2.5, **kwargs):
        if execution_scope not in ('go2', 'unitree'):
            raise ValueError('Unknown execution scope')
        self.execution_scope = execution_scope
        if not math.isfinite(prediction_max_age) or not 0 < prediction_max_age < 45 * DT:
            raise ValueError('H45 prediction age must leave a positive future-action window')
        self.prediction_max_age = prediction_max_age
        super().__init__(*args, **kwargs)
        self.keep_review_monitor_until_commit = True

    def status(self):
        result = super().status()
        if result['execution'] is not None:
            execution = result['execution']
            execution.pop('goal_actions', None)
            dispatches = execution.pop('dispatches', [])
            execution['dispatch_count'] = len(dispatches)
            execution['last_dispatch'] = dispatches[-1] if dispatches else None
        result.update(action_transform=TRANSFORM, capabilities=[CAPABILITY, EXPIRY_CAPABILITY, BOOTSTRAP_CAPABILITY, REUSE_CAPABILITY, "h45_rtc_scheduled_switch_v1", "h45_wait_resume_v1", "h45_rtc_pipeline_v1", "h45_rtc_arrival_switch_v1", "h45_piper_no_clip_v1", "h45_piper_raw_targets_v1"],
                      accepted_policy_schema=SCHEMA_ID, action_horizon=45,
                      immediate_commit=True,
                      observation_transport='execution_feedback_only',
                      prediction_expiry_policy='drop_prediction_keep_unexpired_buffer',
                      prediction_max_age_s=self.prediction_max_age, prediction_horizon_s=45 * DT,
                      discarded_predictions=getattr(self, 'dropped_predictions', 0),
                      piper_can_prepare_on_new_session=True,
                      new_prediction_policy='replace_remaining_at_next_control_tick',
                      go2_translation_scale_undone=1.0, go2_translation_scaling='none',
                      piper_anchor='paired_recorded_model_joint_state',
                      piper_joint_tracking_rate_rad_s=None, gripper_tracking_rate_m_s=None,
                      piper_move_speed_percent=getattr(self.hardware, 'piper_move_speed_percent', 100),
                      piper_target_policy='raw_targets_sdk_firmware_limits_unchanged',
                      actuator_setpoints_rate_limited=False)
        result['capabilities'].append('h45_coordinated_start_v1')
        result['capabilities'].append('h45_coordinated_arrival_v1')
        result['capabilities'].append('h45_live_continuous_v1')
        if self.review_config is not None:
            result['capabilities'].append('h45_review_pause_v1')
            result['capabilities'].append('h45_review_deadline_v1')
        result['review_resume_pending']=self.resume_monitoring
        result.update(execution_scope=self.execution_scope,
                      action_kind='go2' if self.execution_scope == 'go2' else 'unitree',
                      piper_execution_enabled=self.execute and self.execution_scope == 'unitree',
                      actuator_setpoints_rate_limited=False,
                      ignored_action_indices=list(range(3 if self.execution_scope == 'go2' else 10, 14)))
        result['go2_feedback_code_checked'] = getattr(self.hardware, 'check_go2_error_code', True)
        result['go2_feedback_code'] = getattr(self.hardware, 'go2_feedback_code', None)
        result['live_camera_capture'] = bool(getattr(self.hardware, 'cameras', {}))
        diagnostics = getattr(self.hardware, 'sensor_diagnostics', None)
        result['sensor_diagnostics'] = dict(diagnostics) if diagnostics is not None else None
        piper = getattr(self.hardware, 'piper', None)
        if piper is not None:
            try:
                arm = piper.GetArmStatus().arm_status
                result['piper_feedback'] = dict(ctrl_mode=int(arm.ctrl_mode), arm_status=int(arm.arm_status),
                    error_code=int(arm.err_code), enabled=[bool(x) for x in piper.GetArmEnableStatus()])
            except Exception as exc:
                result['piper_feedback_error'] = str(exc)
        result.update(execution_ready=False, waiting_for_standing=False, readiness_error=None)
        result['go2_standing'] = None
        if hasattr(self.hardware.sensors, 'streams'):
            try:
                with self.hardware.sensors.lock:
                    _, go2 = self.hardware.sensors.streams['go2'].past(time.monotonic(), .075)
                result['go2_body_height_m'] = float(go2[4])
                result['go2_standing'] = bool(math.isfinite(go2[4]) and go2[4] >= .23)
            except Exception:
                result['go2_standing'] = False
        motion_socket = getattr(self.hardware, 'motion_socket', None)
        result['motion_bridge_ready'] = motion_socket is not None and motion_socket.exists()
        if self.execute and not self.fault:
            try:
                self.hardware.check_execution()
                result['execution_ready'] = True
            except Exception as exc:
                result['readiness_error'] = str(exc)
                result['waiting_for_standing'] = 'not standing' in str(exc)
        elif self.fault:
            result['readiness_error'] = self.fault
        return result

    def reset(self):
        """A new explicit client execution may select CAN, without clearing faults."""
        with self.lock:
            if self.review_phase != 'open' or self.review_requested.is_set() or self.resume_monitoring:
                raise RuntimeError('Cannot reset during review pause or fresh-plan preparation')
            if self.active or self.fault:
                raise RuntimeError('Active/faulted gateway cannot reset')
            piper = getattr(self.hardware, 'piper', None)
            changed = None
            if self.execute and self.execution_scope == 'unitree' and piper is not None:
                guard = getattr(self.hardware, 'guard', None)
                if guard is None or guard.inhibited or getattr(self.hardware, 'fault', None):
                    raise RuntimeError('Piper safety fault must be inspected before a new session')
                from .piper_mode import switch_to_can
                command_started = False
                def send(method, *args):
                    nonlocal command_started
                    command_started = True
                    return guard.call(method, *args)
                deadline = time.monotonic() + .75
                while True:
                    try:
                        with self.hardware.io_lock, guard.lock:
                            changed = switch_to_can(piper, send)
                        break
                    except Exception as exc:
                        missing_feedback = ('Missing/stale Piper feedback' in str(exc)
                                            or 'missing or stale Piper CAN feedback' in str(exc))
                        if not command_started and missing_feedback and not guard.inhibited:
                            if time.monotonic() < deadline:
                                time.sleep(.02)
                                continue
                            # No actuator command was sent. Keep this standby
                            # attempt retryable instead of creating an E-stop.
                            raise RuntimeError('Piper standby feedback is not fresh; no session commands sent') from exc
                        self.stop('Piper session preparation failed: ' + str(exc))
                        raise
            result = super().reset()
            self._held_wait_epoch = None
            if changed is not None:
                result['piper_session_preparation'] = changed
            return result

    def finish_stream(self):
        """Release only a normally stopped recording; never clear an error latch."""
        with self.lock:
            if self.fault or getattr(self.hardware, 'fault', None):
                raise RuntimeError('A faulted executor cannot acknowledge normal completion')
            if (self.active or not self.stream_mode or not self.stream.finished
                    or self.plan is None or self.plan.get('status') != 'done'):
                raise RuntimeError('The recording has not finished normally')
            if self.session is None and hasattr(self, '_last_completion_ack'):
                return dict(self._last_completion_ack)
            history_path = None
            if hasattr(self.hardware, 'run'):
                history_path = self.hardware.run / ('h45_completed_' + self.session + '.json')
                history = [{key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in plan.items() if key != 'snapshot'}
                           for plan in self.stream.history]
                history_path.write_text(json.dumps(history, indent=2) + '\n')
            self.session = None
            self.snapshots.clear()
            self.stream_prepared = None
            self._last_completion_ack = dict(released=True, normal_completion=True, motion_started=False,
                executor_reusable=True, history_path=str(history_path) if history_path else None)
            return dict(self._last_completion_ack)

    def prepare(self, request):
        raise ValueError('H45 goals require stream prepare/commit')

    def observe(self):
        """Recorded images need a fresh control anchor, not a live JPEG upload."""
        with self.lock:
            if self.fault or self.session is None:
                raise RuntimeError(self.fault or 'Reset the episode first')
            self.hardware.check_execution()
            live = bool(getattr(self.hardware, 'cameras', {}))
            snapshot = (self.hardware.sensors.snapshot() if live else
                        getattr(self.hardware.sensors, 'execution_snapshot', self.hardware.sensors.snapshot)())
            snapshot = dict(snapshot)
            if not live:
                snapshot.update(images={}, image_stamps={})
            snapshot.update(observation_id=uuid.uuid4().hex, session=self.session,
                            review_epoch=self.review_epoch, review_phase=self.review_phase,
                            observation_transport='live_images_and_state' if live else 'execution_feedback_only',
                            waiting_for_prediction=self.stream.waiting_since is not None,
                            wait_epoch=self.stream.wait_epoch)
            self.snapshots[snapshot['observation_id']] = {key: value for key, value in snapshot.items() if key != 'images'}
            while len(self.snapshots) > 32:
                self.snapshots.popitem(last=False)
            return snapshot

    def _discard(self, reason, **details):
        self.dropped_predictions = getattr(self, 'dropped_predictions', 0) + 1
        return dict(accepted=False, discarded=True, reason=reason,
                    previous_prediction_continues=self.active and self.stream.waiting_since is None,
                    waiting_for_prediction=self.stream.waiting_since is not None, **details)

    def _resume_valid(self, snapshot):
        """A stopped-state prediction cannot be reused across wait episodes or motion."""
        self.resume_check = dict(wait_epoch=self.stream.wait_epoch,
                                 observation_wait_epoch=snapshot.get('wait_epoch'))
        if (not self.stream.wait_on_exhaustion or self.stream.waiting_since is None
                or self.stream.pending is not None or not snapshot.get('waiting_for_prediction')
                or snapshot.get('wait_epoch') != self.stream.wait_epoch):
            self.resume_check['reason'] = 'wait_episode_or_buffer_changed'
            return False
        current = getattr(self.hardware.sensors, 'execution_snapshot', self.hardware.sensors.snapshot)()
        old, new = np.asarray(snapshot['state']), np.asarray(current['state'])
        yaw_change = angle_delta(math.atan2(old[5], old[6]), math.atan2(new[5], new[6]))
        errors = dict(translation_m=float(np.linalg.norm(new[:2]-old[:2])), yaw_rad=abs(yaw_change))
        limits = dict(translation_m=.02, yaw_rad=.05)
        if self.execution_scope == 'unitree':
            errors.update(joint_rad=float(np.max(np.abs(new[10:16]-old[10:16]))),
                          gripper_m=float(abs(new[16]-old[16])))
            limits.update(joint_rad=.05, gripper_m=.005)
        failed = [k for k in errors if not math.isfinite(errors[k]) or errors[k] > limits[k]]
        self.resume_check.update(measured_change=errors, limits=limits, failed_checks=failed)
        return not failed

    def heartbeat(self, request):
        with self.lock:
            fault = self.fault or getattr(self.hardware, 'fault', None)
            if fault:
                raise RuntimeError('H45 execution fault: ' + str(fault))
            return super().heartbeat(request)

    def prepare_stream(self, request):
        with self.lock:
            self._check_review_credentials(request)
            if self.resume_monitoring and request.get('bootstrap') is not True:
                raise ValueError('Review resume requires a fresh bootstrap at row zero')
            if self.fault:
                raise RuntimeError('H45 execution fault: ' + self.fault)
            if not self.execute or (self.active and not self.stream_mode):
                raise RuntimeError('H45 gateway is observe-only, faulted, or busy')
            if request.get('schema_id') != SCHEMA_ID:
                raise ValueError('Expected the explicit H45 displacement schema')
            if request.get('execution_scope', 'unitree') != self.execution_scope:
                raise ValueError('Client and gateway execution scopes differ')
            if request.get('session') != self.session or self.session is None:
                raise ValueError('Wrong episode session')
            snapshot = self.snapshots.get(request.get('observation_id'))
            if snapshot is None:
                raise ValueError('Unknown/consumed live observation id')
            now = time.monotonic()
            bootstrap = request.get('bootstrap', False)
            continuous = request.get('live_continuous', False)
            if type(continuous) is not bool:
                raise ValueError('live_continuous must be a boolean')
            if continuous and not getattr(self.hardware, 'cameras', {}):
                raise ValueError('Continuous live execution requires live cameras')
            if self.stream.history and continuous != self.stream.history[0].get('live_continuous', False):
                raise ValueError('Live/recorded session mode cannot change')
            if type(bootstrap) is not bool:
                raise ValueError('bootstrap must be a boolean')
            if bootstrap and (self.stream_mode or self.active or self.stream.history):
                raise ValueError('Only the first preloaded chunk may start a new timeline')
            resume = request.get('resume_from_wait', False)
            wait_on_exhaustion = request.get('wait_on_exhaustion', False)
            pipeline = request.get('rtc_pipeline', False)
            arrival = request.get('rtc_arrival', False)
            if type(arrival) is not bool:
                raise ValueError('rtc_arrival must be a boolean')
            if type(pipeline) is not bool or (pipeline and (not bootstrap or not wait_on_exhaustion)):
                raise ValueError('RTC pipeline must be selected on the initial waiting-enabled chunk')
            if bootstrap:
                if arrival and not pipeline:
                    raise ValueError('Arrival handoff requires the RTC pipeline')
            elif arrival != self.stream.arrival_handoff:
                raise ValueError('RTC handoff mode cannot change within a session')
            if arrival and request.get('rtc_switch_at') is not None:
                raise ValueError('Arrival handoff cannot specify a fixed RTC switch time')
            if type(resume) is not bool or type(wait_on_exhaustion) is not bool:
                raise ValueError('Wait/resume options must be booleans')
            if wait_on_exhaustion and not bootstrap:
                raise ValueError('Waiting mode must be selected on the initial chunk')
            if resume and (bootstrap or request.get('rtc_switch_at') is not None):
                raise ValueError('Resume requires a new unconstrained stopped-state prediction')
            if resume and not self._resume_valid(snapshot):
                self.snapshots.pop(request['observation_id'])
                return self._discard('wait_state_changed', resume_check=self.resume_check)
            if not resume and self.stream.waiting_since is not None:
                self.snapshots.pop(request['observation_id'])
                return self._discard('fresh_stopped_state_prediction_required')
            origin = float(request['origin_at'])
            age = now - snapshot['stamp']
            if (not math.isfinite(origin) or origin > snapshot['stamp'] + .001
                    or snapshot['stamp'] - origin > .15
                    or age < 0):
                raise RuntimeError('H45 prediction origin is invalid: origin=%r snapshot=%r age=%r' %
                                   (origin, snapshot['stamp'], age))
            max_age = self.max_action_age if bootstrap else self.prediction_max_age
            if not arrival and age > max_age:
                self.snapshots.pop(request['observation_id'])
                return self._discard('live_observation_expired', observation_age_s=age,
                                     max_action_age_s=max_age)
            first = 0 if bootstrap or resume else max(0, int((now - origin) / DT))
            if first >= 45:
                if arrival:
                    self.snapshots.pop(request['observation_id'])
                    return self._discard('prediction_horizon_exhausted')
                raise RuntimeError('H45 prediction already expired')
            self.hardware.check_execution()
            raw = finite(request['actions'], (45, 14))
            model_state = finite(request['model_state'], (14,))
            rows = decode_goals(raw, model_state[3:10], self.velocity_limits,
                                execute_piper=self.execution_scope == 'unitree')
            original_arm_goals = np.column_stack((raw[:, 3:9] + model_state[3:9], raw[:, 9]))
            projected_entries = int(np.count_nonzero(rows[:, 3:] != original_arm_goals)) if self.execution_scope == 'unitree' else 0
            current = getattr(self.hardware.sensors, 'execution_snapshot', self.hardware.sensors.snapshot)()
            candidate = rows[first].copy()
            anchor_yaw = math.atan2(snapshot['state'][5], snapshot['state'][6])
            current_yaw = math.atan2(current['state'][5], current['state'][6])
            candidate[:2] = heading(candidate[:2], angle_delta(anchor_yaw, current_yaw))
            if self.execution_scope == 'unitree':
                validate_h45_dispatch(candidate, self.velocity_limits)
            else:
                action_plan(candidate[None, :3], 'go2', velocity_limits=self.velocity_limits)
            remaining = float(request['remaining_video_s']) if bootstrap and not continuous else None
            if bootstrap and not continuous and (not math.isfinite(remaining) or not 0 < remaining <= 300):
                raise ValueError('Invalid preloaded-video duration')
            end = None if continuous else (now + remaining if bootstrap else float(request['end_at']))
            if end is not None and (not math.isfinite(end) or end > now + 300):
                raise ValueError('Invalid recorded-episode end time')
            if end is not None and end + self.stream.pause_duration(now) <= now:
                self.snapshots.pop(request['observation_id'])
                return self._discard('recorded_episode_ended')
            self.snapshots.pop(request['observation_id'])
            self.stream.hold_last = False
            rtc_switch = request.get('rtc_switch_at')
            rtc_source = request.get('rtc_source_plan_id')
            if arrival and not bootstrap and not resume:
                old = self.stream.latest_plan()
                if old is None or rtc_source != old['plan_id']:
                    raise ValueError('Arrival handoff requires the latest source action buffer')
                if now >= old['origin_at']+len(old['rows'])*DT:
                    return self._discard('fresh_stopped_state_prediction_required')
            if rtc_switch is not None:
                old = self.stream.latest_plan()
                if (bootstrap or old is None or rtc_source != old['plan_id']
                        or not math.isfinite(float(rtc_switch))
                        or not origin < rtc_switch < origin+45*DT
                        or rtc_switch >= old['origin_at']+len(old['rows'])*DT):
                    raise ValueError('Invalid RTC switch or source action buffer')
                if now >= rtc_switch:
                    return self._discard('rtc_switch_deadline_missed')
                if self.stream.pipeline and rtc_switch <= old.get('scheduled_start', old['origin_at']):
                    raise ValueError('RTC pipeline switch must follow its source plan')
            plan_id = request.get('plan_id', uuid.uuid4().hex)
            if not isinstance(plan_id, str) or len(plan_id) != 32 or uuid.UUID(hex=plan_id).hex != plan_id:
                raise ValueError('Invalid client chunk ID')
            self._check_review_deadline(request)
            self.stream_prepared = dict(plan_id=plan_id, rows=rows,
                resume_from_wait=resume, wait_on_exhaustion=wait_on_exhaustion,
                rtc_pipeline=pipeline,
                rtc_arrival=arrival,
                rtc_switch_at=rtc_switch, rtc_source_plan_id=rtc_source,
                goal_actions=raw.copy(), snapshot=snapshot, origin_at=origin,
                model_state=model_state.tolist(),
                projected_piper_entries=projected_entries,
                bootstrap=bootstrap, remaining_video_s=remaining, live_continuous=continuous,
                anchor_yaw=anchor_yaw, end_at=end, status='prepared', executed_steps=0)
            return dict(accepted=True, plan_id=self.stream_prepared['plan_id'], status='prepared',
                        received_goal_actions=raw.tolist(), applied_actions=rows.tolist(),
                        applied_xy_frame='chunk_start_heading_until_dispatch',
                        live_continuous=continuous,
                        remaining_execution_s=None if continuous else (remaining if bootstrap else end + self.stream.pause_duration(now) - now),
                        expired_prefix_steps=first, expires_at=None if bootstrap else origin + 45 * DT,
                        observation_age_s=age,
                        action_transform=TRANSFORM)

    def commit_stream(self, request):
        """Use the existing immediate path or a future cross-device start."""
        with self.review_lock, self.lock:
            if self.review_config is not None:self._check_review_credentials(request)
            if self.resume_monitoring:
                if self.hold_window is None or not self.hold_window.ready:
                    raise RuntimeError('Review hold is not ready for commit')
                self.poll_review()
                if self.fault or not self.hold_window.ready:
                    raise RuntimeError('Review hold changed before commit')
                if not self.stream_prepared or self.stream_prepared.get('bootstrap') is not True:
                    raise ValueError('Review resume commit requires a fresh bootstrap')
            plan = self.stream_prepared
            if self.fault or plan is None or request.get('plan_id') != plan['plan_id']:
                raise ValueError('No matching H45 prediction')
            now = time.monotonic()
            coordinated = request.get('coordinated', False)
            requested_start = request.get('coordinated_start_at')
            fresh_start = plan.get('bootstrap') or plan.get('resume_from_wait')
            if type(coordinated) is not bool or (requested_start is not None and not coordinated):
                raise ValueError('Invalid coordinated commit options')
            start = now
            coordinated_arrival = coordinated and plan.get('rtc_arrival')
            if coordinated:
                if not (fresh_start or coordinated_arrival) and requested_start is not None:
                    raise ValueError('Scheduled RTC already carries its switch time')
                if (fresh_start or coordinated_arrival) and requested_start is None:
                    raise ValueError('Coordinated arrival requires a switch time')
                start = float(requested_start if fresh_start or coordinated_arrival else plan['rtc_switch_at'])
                lead = start - now
                if not math.isfinite(lead) or lead > 2.5 or (fresh_start and lead > .75):
                    raise ValueError('Invalid coordinated switch time')
                min_lead = .018 if coordinated_arrival and not fresh_start else .12
                if coordinated_arrival and not fresh_start and lead > .12:
                    raise ValueError('Arrival switch is too far in the future')
                if lead < min_lead:
                    self.stream_prepared = None
                    return self._discard('coordinated_switch_lead_too_short')
            max_age = self.max_action_age if plan.get('bootstrap') else self.prediction_max_age
            if not plan.get('rtc_arrival') and now - plan['snapshot']['stamp'] > max_age:
                self.stream_prepared = None
                return self._discard('live_observation_expired_before_commit',
                    observation_age_s=now - plan['snapshot']['stamp'], max_action_age_s=max_age)
            if plan.get('bootstrap'):
                if self.stream.history or self.active:
                    raise RuntimeError('The initial preloaded chunk has already started')
                plan['origin_at'] = start
                plan['end_at'] = None if plan.get('live_continuous') else start + plan['remaining_video_s']
            if plan.get('resume_from_wait'):
                if not self._resume_valid(plan['snapshot']):
                    self.stream_prepared = None
                    return self._discard('wait_state_changed_before_commit', resume_check=self.resume_check)
                # The robot held still during inference; start the fresh chunk at row zero.
                plan['origin_at'] = start
            elif self.stream.waiting_since is not None:
                self.stream_prepared = None
                return self._discard('fresh_stopped_state_prediction_required_before_commit')
            sampled_at = start if coordinated_arrival else now
            first = 0 if fresh_start else int((sampled_at - plan['origin_at']) / DT)
            if not 0 <= first < 45:
                if coordinated_arrival:
                    self.stream_prepared = None
                    return self._discard('prediction_horizon_exhausted')
                raise RuntimeError('No unexpired H45 action remains')
            end = plan['end_at']
            if self.stream_deadline is not None and end != self.stream_deadline:
                raise ValueError('A recorded-episode stream cannot change its end time')
            if end is not None and now >= end + self.stream.pause_duration(now):
                self.stream_prepared = None
                return self._discard('recorded_episode_ended')
            if end is not None and coordinated_arrival and not fresh_start and start >= end + self.stream.pause_duration(now):
                self.stream_prepared = None
                return self._discard('recorded_episode_ended')
            self.hardware.check_execution()
            self.stream_deadline = end
            switch = plan.get('rtc_switch_at')
            if plan.get('rtc_arrival') and not plan.get('bootstrap') and not plan.get('resume_from_wait'):
                old = self.stream.latest_plan()
                if old is None or plan.get('rtc_source_plan_id') != old['plan_id']:
                    self.stream_prepared = None
                    return self._discard('rtc_source_changed_before_commit')
                if sampled_at >= old['origin_at']+len(old['rows'])*DT:
                    self.stream_prepared = None
                    return self._discard('fresh_stopped_state_prediction_required_before_commit')
            if switch is not None and now >= switch:
                self.stream_prepared = None
                return self._discard('rtc_switch_deadline_missed_before_commit')
            self._check_review_deadline(request)
            if self.review_deadline is not None and self.resume_monitoring:
                self.review_deadline.check_start(start if switch is None else switch)
            plan.update(status='scheduled', scheduled_start=start if switch is None else switch)
            if plan.get('bootstrap'):
                self.stream.wait_on_exhaustion = plan['wait_on_exhaustion']
                self.stream.pipeline = plan['rtc_pipeline']
                self.stream.arrival_handoff = plan['rtc_arrival']
            self.stream.schedule(plan, replace_future=plan.get('rtc_arrival', False))
            if self.review_deadline is not None and self.resume_monitoring:self.review_deadline.release()
            self.stream_prepared = None
            if self.resume_monitoring:
                self.resume_monitoring = False
                self.review_monitor_stop.set()
            self.last_controller_at = now
            if not self.active:
                self.stream_mode = True
                self.active = True
                self.cancel.clear()
                self.last_tick = now
                self.worker = threading.Thread(target=self._run_stream, daemon=True)
                self.worker.start()
            # Resume adds the wait through the common switch time. Subsequent
            # commits report the accumulated pause from the running stream.
            effective_end = None if end is None else end + self.stream.pause_duration(start if plan.get('resume_from_wait') else now)
            return dict(accepted=True, plan_id=plan['plan_id'], status='scheduled',
                        origin_at=plan['origin_at'], end_at=end, scheduled_start=plan['scheduled_start'],
                        effective_end_at=effective_end,
                        remaining_prediction_s=plan['origin_at'] + 45 * DT - now,
                        initial_chunk_preloaded=plan.get('bootstrap', False),
                        first_unexpired_action=first, replacement='next_control_tick')

    def _run_stream(self):
        due = None
        epoch = self.review_epoch
        try:
            while not self.cancel.is_set():
                now = time.monotonic()
                finish_for_review = False
                with self.lock:
                    if self.review_requested.is_set() or epoch != self.review_epoch:
                        break
                    if now - self.last_controller_at > 1.:
                        raise RuntimeError('Local inference-client heartbeat expired')
                    if self.stream_deadline is not None and now >= self.stream_deadline+self.stream.pause_duration(now):
                        self.stream.finished = True
                        break
                    sampled = self.stream.sample(now)
                    if self.stream.finished:
                        break
                    if sampled is None and self.stream.waiting_since is not None:
                        self.hardware.check_execution()
                        if self.review_config is not None:
                            finish_for_review = True
                        elif getattr(self, '_held_wait_epoch', None) != self.stream.wait_epoch:
                            self.hardware.hold()
                            self._held_wait_epoch = self.stream.wait_epoch
                        # Waiting is not a missed actuator tick. Restart the clock on resume.
                        due = None
                    if sampled is not None and due is None:
                        due = now
                    self.last_tick = now
                if finish_for_review:
                    self._finish_hold()
                    return
                if sampled is None or now < due:
                    self.cancel.wait(.002)
                    continue
                plan, index = sampled
                current = getattr(self.hardware.sensors, 'execution_snapshot', self.hardware.sensors.snapshot)()
                yaw = math.atan2(current['state'][5], current['state'][6])
                row = plan['rows'][index].copy()
                row[:2] = heading(row[:2], angle_delta(plan['anchor_yaw'], yaw))
                joint_goal = row[3:].copy()
                if self.execution_scope == 'unitree':
                    validate_h45_dispatch(row, self.velocity_limits)
                else:
                    row = row[:3]
                    action_plan(row[None], 'go2', velocity_limits=self.velocity_limits)
                if not self._dispatch_row(row, epoch):
                    break
                dispatched = time.monotonic()
                with self.lock:
                    self.stream.record_dispatch(plan, index, dispatched, sampled_at=now)
                    plan.setdefault('dispatches', []).append(dict(
                        monotonic=dispatched, sampled_at=now, action_index=index, command=row.tolist(),
                        predicted_joint_goal=joint_goal.tolist(),
                        raw_model_joint_goal=[*(plan['goal_actions'][index, 3:9] + np.asarray(plan['model_state'][3:9])),
                                              float(plan['goal_actions'][index, 9])],
                        measured_state=current['state']))
                    self.plan = plan
                    self.last_tick = dispatched
                # Skip elapsed ticks instead of replaying stale commands in a burst.
                due += (math.floor((now - due) / DT) + 1) * DT
            if not self.cancel.is_set():
                self._finish_hold()
                with self.lock:
                    if self.plan is not None:
                        self.plan.update(status='done', finished_at=time.monotonic())
        except Exception as exc:
            self.stop(str(exc))
        finally:
            with self.lock:
                self.active = False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--scope', choices=('go2', 'unitree'), default='unitree')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    enforce_velocity_limits = config.get('goal_distance_go2_velocity_limits_enabled', True)
    if type(enforce_velocity_limits) is not bool:
        raise ValueError('goal_distance_go2_velocity_limits_enabled must be a boolean')
    if not enforce_velocity_limits:
        config['velocity_limits'] = False
    token = args.token_file.read_text().strip()
    if len(token) < 24:
        raise ValueError('A configured gateway token is required')
    if args.execute and args.scope == 'go2':
        from .go2_stream_gateway import Go2Hardware
        hardware = Go2Hardware(config)
    else:
        from .unitree import UnitreeHardware
        hardware = UnitreeHardware(dict(config, capture_live_cameras=config.get('h45_live_observations', False)), execute=args.execute)
    service = GoalGateway(hardware, execute=args.execute,
                          max_action_age=config.get('max_action_age_s', .75),
                          prediction_max_age=config.get('goal_prediction_max_age_s', 2.5),
                          velocity_limits=config.get('velocity_limits'), execution_scope=args.scope,
                          review_pause=config.get('review_pause'))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', config['port']), handler(service, token))
    except BaseException:
        service.close()
        raise
    def terminate(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(json.dumps(dict(port=config['port'], execute=args.execute, capability=CAPABILITY)), flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        run = hardware.run
        service.close()
        history = []
        for plan in service.stream.history:
            history.append({k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in plan.items() if k != 'snapshot'})
        (run / 'h45_execution_history.json').write_text(json.dumps(history, indent=2) + '\n')


if __name__ == '__main__':
    main()
