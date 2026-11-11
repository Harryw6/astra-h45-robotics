"""Windows-local recorded-video -> remote H45 policy -> immediate Unitree execution."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import cv2
import numpy as np

from .controller import Device, StreamHeartbeat

SLOTS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
SCHEMA = 'heterovla_v3_goal_distance_h45_masked_v1'
CAPABILITY = 'goal_distance_h45_stream_v1'
EXPIRY_CAPABILITY = 'stale_prediction_drop_v1'
BOOTSTRAP_CAPABILITY = 'preloaded_initial_h45_v1'
REUSE_CAPABILITY = 'normal_completion_reuse_v1'
IMAGE_PREPROCESSING = 'openpi_pil_bilinear_pad_224_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wait_for_idle_can_feedback(device, status, timeout=3.):
    """A new command may wait for fresh standby data; it never clears faults."""
    deadline = time.monotonic() + timeout
    waiting_reported = False
    while not status.get('fault') and not status.get('hardware_fault'):
        messages = str(status.get('sensor_error') or '') + ' ' + str(status.get('readiness_error') or '')
        transient = ('missing or stale Piper CAN feedback' in messages
                     or 'Missing Piper feedback/enable status' in messages)
        if not transient:
            break
        if not waiting_reported:
            print(json.dumps(dict(waiting_for_fresh_piper_feedback=True, robot_commands_sent=False)), flush=True)
            waiting_reported = True
        if time.monotonic() >= deadline:
            raise RuntimeError('Piper feedback did not become fresh within %.1f seconds; no motion started' % timeout)
        time.sleep(.05)
        status = device.call('/status')
    return status


def require_unitree_execution_ready(status):
    reason = status.get('fault') or status.get('hardware_fault') or status.get('sensor_error')
    if not status['execution_enabled'] or reason:
        raise RuntimeError('Unitree execution is not ready: ' + str(reason or 'execution is disabled'))
    if status.get('go2_standing') is False:
        raise RuntimeError('Go2 must be standing before starting the recording')
    if status.get('execution_ready') is False:
        piper = status.get('piper_feedback', {})
        can_prepare = (status.get('piper_can_prepare_on_new_session')
            and piper.get('ctrl_mode') in (0, 2) and piper.get('arm_status') == 0
            and piper.get('error_code') == 0 and len(piper.get('enabled', [])) == 6
            and all(piper['enabled']))
        if not can_prepare:
            raise RuntimeError('Executor remains in standby: ' + str(status.get('readiness_error')))


class RecordedEpisode:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'episode.json').read_text(encoding='utf-8'))
        for name, expected in self.metadata['sha256'].items():
            if sha256(self.directory / name) != expected:
                raise ValueError('Recorded episode asset changed: ' + name)
        with np.load(self.directory / 'timeline.npz', allow_pickle=False) as archive:
            self.times = archive['source_times'].copy()
            self.rows = archive['source_rows'].copy()
            self.state = archive['state14'].copy()
            self.valid = archive['state14_valid'].all(axis=1) & np.isfinite(self.state).all(axis=1)
            self.camera_frames = archive['camera_frames'].copy()
        self.duration = float(self.metadata['duration_s'])
        if (not self.valid.any() or self.state.shape != (len(self.times), 14)
                or np.any(np.diff(self.times) <= 0) or not np.isfinite(self.times).all()
                or self.camera_frames.shape != (len(self.times), 3)
                or np.any(self.camera_frames < 0) or not 0 < self.duration <= 300):
            raise ValueError('Invalid synchronized validation episode')
        self.caps, self.last = {}, {}

    def index_at(self, elapsed):
        index = int(np.searchsorted(self.times, elapsed, side='right') - 1)
        if index < 0 or index >= len(self.times) or not self.valid[index]:
            return None
        return index

    def observation(self, index):
        if not self.valid[index]:
            raise ValueError('Recorded state is missing; model inference must wait for a valid frame')
        images = {}
        for column, slot in enumerate(SLOTS):
            target = int(self.camera_frames[index, column])
            if slot not in self.caps:
                cap = cv2.VideoCapture(str(self.directory / self.metadata['videos'][slot]))
                if not cap.isOpened():
                    raise RuntimeError('Could not open recorded camera: ' + slot)
                self.caps[slot] = cap
                self.last[slot] = (-1, None)
            cap = self.caps[slot]
            previous, frame = self.last[slot]
            if target < previous or target - previous > 60:
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                    raise RuntimeError('Could not seek recorded camera: ' + slot)
                previous = target - 1
            while previous < target:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Recorded camera decode failed: ' + slot)
                previous += 1
            if frame is None or frame.shape != (480, 640, 3):
                raise ValueError('Expected original recorded RGB frame 640x480')
            self.last[slot] = (previous, frame)
            images[slot] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return dict(state=self.state[index].copy(), image=images,
                    image_mask={slot: np.bool_(True) for slot in SLOTS},
                    prompt=self.metadata['prompt'])

    def close(self):
        for cap in self.caps.values():
            cap.release()


def validate_policy_metadata(metadata, checkpoint_step):
    if type(checkpoint_step) is not int or checkpoint_step <= 0:
        raise ValueError('Expected a positive integer checkpoint step')
    if (metadata.get('schema_id') != SCHEMA or metadata.get('action_horizon') != 45
            or metadata.get('physical_state_dim') != 14
            or metadata.get('checkpoint_step') != checkpoint_step):
        raise ValueError('Expected H45 ckpt%d policy; received checkpoint step %r' %
                         (checkpoint_step, metadata.get('checkpoint_step')))


class RemotePolicy:
    def __init__(self, url, checkpoint_step=25000):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.codec, self.packer = msgpack_numpy, msgpack_numpy.Packer()
        self.connection = connect(url, compression=None, proxy=None, ping_interval=None,
                                  open_timeout=15, max_size=8 * 1024 * 1024)
        self.metadata = self._receive(20)
        try:
            validate_policy_metadata(self.metadata, checkpoint_step)
        except ValueError:
            self.close()
            raise

    def _receive(self, timeout):
        value = self.connection.recv(timeout=timeout)
        if isinstance(value, str):
            raise RuntimeError(value)
        return self.codec.unpackb(value)

    def prepare_input(self, observation, rtc=None):
        payload = dict(observation)
        if rtc is not None:
            from .rtc_h45 import CAPABILITY as RTC_CAPABILITY
            if self.metadata.get('rtc_capability') != RTC_CAPABILITY:
                raise ValueError('Server does not support H45 RTC')
            payload['rtc'] = rtc
        preprocessing = self.metadata.get('client_image_preprocessing')
        if preprocessing:
            if (preprocessing.get('id') != IMAGE_PREPROCESSING
                    or preprocessing.get('height') != 224 or preprocessing.get('width') != 224):
                raise ValueError('Unsupported server image preprocessing contract')
            try:
                from PIL import __version__ as pillow_version
                from openpi_client.image_tools import resize_with_pad
            except ImportError as exc:
                raise RuntimeError('Install Pillow==' + preprocessing['pillow_version']
                                   + ' and apply the H45 transport update before execution') from exc
            if pillow_version != preprocessing['pillow_version']:
                raise RuntimeError('Exact model image preprocessing requires Pillow=='
                                   + preprocessing['pillow_version'])
            payload['image'] = {name: resize_with_pad(image, 224, 224)
                                for name, image in payload['image'].items()}
            payload['image_preprocessing'] = IMAGE_PREPROCESSING
        expected = {name: hashlib.sha256(image.tobytes()).hexdigest()
                    for name, image in payload['image'].items()}
        if self.metadata.get('image_transport') == 'lossless_png':
            images = {}
            for name, image in payload['image'].items():
                ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise RuntimeError('Lossless recorded-image encoding failed')
                images[name] = {'lossless_png': encoded.tobytes()}
            payload['image'] = images
        # Fixed noise makes changes reflect current recorded observations.
        payload['sampling_noise'] = np.random.default_rng(42).standard_normal((45, 32)).astype(np.float32)
        wire = self.packer.pack(payload)
        return dict(wire=wire, wire_bytes=len(wire), image_sha256=expected,
                    state_sha256=hashlib.sha256(observation['state'].tobytes()).hexdigest())

    def infer_prepared(self, prepared, timeout=120, *, check_wait=None):
        self.connection.send(prepared['wire'])
        while True:
            try:
                result = self._receive(timeout)
                break
            except TimeoutError:
                if check_wait is None:
                    raise
                check_wait()
        self.last_server_timing = result.get('server_timing', {})
        self.last_rtc = result.get('rtc')
        if result['received_input_sha256']['images'] != prepared['image_sha256']:
            raise ValueError('Realtime server received different validation-image pixels')
        if result['received_input_sha256']['state'] != prepared['state_sha256']:
            raise ValueError('Realtime server received a different recorded state')
        actions = np.asarray(result['actions'], dtype=np.float64)
        if actions.shape != (45, 14) or not np.isfinite(actions).all():
            raise ValueError('Invalid H45 prediction')
        return actions

    def infer(self, observation, timeout=120):
        return self.infer_prepared(self.prepare_input(observation), timeout)

    def close(self):
        self.connection.close()


def control_anchor(packet):
    state = np.asarray(packet['state'], dtype=float)
    anchor = np.array([state[0], state[1], state[2], np.arctan2(state[5], state[6])])
    if not np.isfinite(anchor).all():
        raise ValueError('RTC requires finite current Go2 pose feedback')
    return anchor


def run(source, policy, device, output, execute, execution_scope='unitree', rtc_delay_s=None,
        rtc_handoff='arrival'):
    rtc_enabled = rtc_delay_s is not None
    if rtc_enabled and (not execute or not isinstance(device, Device)):
        raise ValueError('RTC requires an explicit hardware-free loopback or validated Device execution entry')
    from .rtc_h45 import (empty_context, pipeline_prefix_context, PrefixCoverageError,
                         ArrivalPrefixEstimator, ARRIVAL_CAPABILITY)
    from .core import DT
    rtc_plan = None
    rtc_plans = []
    if rtc_handoff not in ('arrival', 'scheduled'):
        raise ValueError('Unknown RTC handoff mode')
    arrival_handoff = rtc_enabled and rtc_handoff == 'arrival'
    prefix_estimator = ArrivalPrefixEstimator(rtc_delay_s) if arrival_handoff else None
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'model.json', policy.metadata)
    first_index = int(np.flatnonzero(source.valid)[0])
    preparation_started = time.monotonic()
    initial_observation = source.observation(first_index)
    initial_input = (policy.prepare_input(initial_observation, rtc=empty_context()) if rtc_enabled
                     else policy.prepare_input(initial_observation))
    initial_preparation_s = time.monotonic() - preparation_started
    started = time.monotonic()
    initial_actions = policy.infer_prepared(initial_input, timeout=None if arrival_handoff else 120)
    initial_inference_s = time.monotonic() - started
    if not rtc_enabled and getattr(device, 'rtc_loopback_only', False):
        # Matched non-RTC benchmark warmup; RTC uses its first valid result directly.
        started = time.monotonic()
        initial_actions = policy.infer_prepared(initial_input)
        initial_inference_s = time.monotonic() - started
    if rtc_enabled:
        if not np.isfinite(rtc_delay_s) or not 0 < rtc_delay_s < 44*DT:
            raise ValueError('RTC prefix must leave future actions inside the H45 horizon')
        rtc_delay_steps = int(np.ceil(rtc_delay_s/DT))
    print(json.dumps(dict(warmup_complete=True, seconds=time.monotonic() - started,
                          first_action_chunk_preloaded=True,
                          source_episode=source.metadata['source_episode'],
                          duration_s=source.duration, real_execution=execute and not getattr(device, 'rtc_loopback_only', False),
                          simulated_execution=bool(getattr(device, 'rtc_loopback_only', False)))), flush=True)
    records, error, final, heartbeats = [], None, None, None
    controller_error = None
    stage = 'control_preflight'
    completed_normally = False
    session = None
    try:
        if execute:
            status = device.calibrate()
            status = wait_for_idle_can_feedback(device, status)
            if (CAPABILITY not in status.get('capabilities', [])
                    or EXPIRY_CAPABILITY not in status.get('capabilities', [])
                    or BOOTSTRAP_CAPABILITY not in status.get('capabilities', [])
                    or REUSE_CAPABILITY not in status.get('capabilities', [])
                    or status.get('accepted_policy_schema') != SCHEMA
                    or not status.get('immediate_commit')
                    or status.get('execution_scope') != execution_scope):
                raise RuntimeError('Deploy the H45 immediate-replacement gateway first')
            if rtc_enabled and 'h45_rtc_scheduled_switch_v1' not in status.get('capabilities', []):
                raise RuntimeError('Deploy the RTC scheduled-switch gateway before real execution')
            if rtc_enabled and 'h45_wait_resume_v1' not in status.get('capabilities', []):
                raise RuntimeError('Deploy the RTC wait/resume gateway before execution')
            if rtc_enabled and 'h45_rtc_pipeline_v1' not in status.get('capabilities', []):
                raise RuntimeError('Deploy/reload the RTC pipeline gateway before execution')
            if arrival_handoff and ARRIVAL_CAPABILITY not in status.get('capabilities', []):
                raise RuntimeError('Deploy/reload the RTC arrival-switch gateway before execution')
            if execution_scope == 'unitree' and 'h45_piper_no_clip_v1' not in status.get('capabilities', []):
                raise RuntimeError('Deploy/reload the Piper no-clipping gateway before execution')
            if execution_scope == 'unitree' and 'h45_piper_raw_targets_v1' not in status.get('capabilities', []):
                raise RuntimeError('Deploy/reload the Piper raw-target gateway before execution')
            # A three-robot ground start can validate the preloaded aircraft
            # trajectory before the pilot changes mode or a device session starts.
            if callable(getattr(device, 'validate_initial_plan', None)):
                stage = 'initial_plan_preflight'
                device.validate_initial_plan(initial_actions, source.duration)
            if callable(getattr(device, 'wait_for_operator_ready', None)):
                stage = 'operator_readiness'
                status = device.wait_for_operator_ready()
            require_unitree_execution_ready(status)
            stage = 'session_reset'
            reset = device.call('/reset', {})
            session = reset['session']
            heartbeats = StreamHeartbeat({'unitree': device}, {'unitree': session})
            heartbeats.start()
        first_time = float(source.times[first_index])
        timeline = time.monotonic() - first_time
        end = timeline + source.duration
        remote_end = None
        last = -1
        bootstrap_pending = execute
        initial_deadline = time.monotonic() + 10
        while rtc_enabled or time.monotonic() < end:
            if heartbeats is not None:
                heartbeats.check()
            elapsed = time.monotonic() - timeline
            if rtc_enabled and not bootstrap_pending:
                scheduling_status = device.call('/status')
                if scheduling_status.get('fault') or scheduling_status.get('hardware_fault'):
                    raise RuntimeError('RTC execution fault: '+str(scheduling_status.get('fault') or scheduling_status.get('hardware_fault')))
                clock = scheduling_status.get('stream') or {}
                elapsed = first_time+clock['source_elapsed_s']
                if clock.get('finished') or elapsed >= source.duration:
                    break
                tail = DT if arrival_handoff else rtc_delay_steps*DT
                if not clock.get('waiting_for_prediction') and elapsed+tail >= source.duration:
                    time.sleep(.01)
                    continue
            if bootstrap_pending and not arrival_handoff and time.monotonic() > initial_deadline:
                raise RuntimeError('Initial fresh execution handshake did not complete within 10 seconds')
            index = first_index if bootstrap_pending or not records else source.index_at(elapsed)
            if (rtc_enabled and not bootstrap_pending and index is None
                    and clock.get('waiting_for_prediction')):
                # A paused clock may land on an invalid recorded frame. Use the
                # latest valid causal input, not a future frame or an endless wait.
                causal = np.flatnonzero(source.valid & (source.times <= elapsed))
                if len(causal):
                    index = int(causal[-1])
            # During a hold the video clock is frozen, but a failed inference may
            # be retried with fresh control feedback from the same recorded frame.
            if index is None or (index == last and not bootstrap_pending and not rtc_enabled):
                time.sleep(.002)
                continue
            if execute and time.monotonic() - device.clock_checked > 10:
                device.calibrate()
            # Video seeking, RGB conversion, PNG encoding and serialization do
            # not consume the live control observation's freshness budget.
            if bootstrap_pending:
                observation, prepared_input = initial_observation, initial_input
                input_preparation_s = initial_preparation_s
            else:
                stage = 'recorded_image_preparation'
                preparation_started = time.monotonic()
                observation = source.observation(index)
                prepared_input = None if rtc_enabled else policy.prepare_input(observation)
                input_preparation_s = time.monotonic() - preparation_started
            observation_requested = time.monotonic()
            stage = 'live_control_observation'
            packet = device.call('/observation') if execute else None
            rtc_context = None
            resuming = False
            if rtc_enabled and not bootstrap_pending:
                if packet['stamp'] <= rtc_plan['origin_at']:
                    # The feedback clock can return the same timestamp on two
                    # adjacent polls. A new plan requires a newer observation.
                    continue
                resuming = bool(packet.get('waiting_for_prediction'))
                if resuming:
                    rtc_context = empty_context()
                else:
                    try:
                        rtc_context = pipeline_prefix_context(rtc_plans, packet['stamp'],
                            control_anchor(packet), observation['state'],
                            prefix_estimator.steps if arrival_handoff else rtc_delay_steps,
                            allow_short_prefix=arrival_handoff)
                    except PrefixCoverageError:
                        # An actual late request may have consumed the remaining
                        # coverage. Continue executing it; the gateway holds only
                        # at exhaustion, then permits a fresh stopped-state request.
                        time.sleep(.02)
                        continue
                prepared_input = policy.prepare_input(observation, rtc=rtc_context)
            observation_roundtrip_s = time.monotonic() - observation_requested
            before = time.monotonic()
            stage = 'model_upload_inference_response'
            if bootstrap_pending:
                actions = initial_actions
            elif arrival_handoff:
                # Poll only to surface hardware faults while the devices drain
                # their current buffers and hold. The model wait has no deadline.
                actions = policy.infer_prepared(prepared_input, timeout=.25,
                                                check_wait=heartbeats.check)
            else:
                actions = policy.infer_prepared(prepared_input,
                                                timeout=(10. if rtc_enabled else 2.) if execute else 120.)
            returned = time.monotonic()
            if returned >= end and not rtc_enabled:
                break
            record = dict(prediction_id=str(len(records)), source_frame_index=index,
                          source_row=int(source.rows[index]), source_time_s=float(source.times[index]),
                          inference_s=initial_inference_s if bootstrap_pending else returned - before,
                          initial_chunk_preloaded=bootstrap_pending,
                          predicted_goals=actions.tolist(),
                          input_preparation_s=input_preparation_s,
                          input_wire_bytes=prepared_input.get('wire_bytes'),
                          server_inference_s=getattr(policy, 'last_server_timing', {}).get('infer_ms', 0.) / 1000.,
                          observation_roundtrip_s=observation_roundtrip_s,
                          model_state_source='paired_recorded_validation_state14',
                          piper_execution_anchor='paired_recorded_model_joint_state',
                          ground_truth_actions_used=False, submitted=False)
            if rtc_enabled:
                record['rtc'] = getattr(policy, 'last_rtc', None)
                record['resume_from_wait'] = resuming
                record['rtc_requested_prefix_s'] = rtc_delay_s
                record['rtc_prefix_steps'] = rtc_context['delay_steps'] if rtc_context is not None else 0
                record['rtc_handoff'] = rtc_handoff
                record['rtc_prefix_estimate_s'] = prefix_estimator.estimate_s if arrival_handoff else rtc_delay_s
                record['request_started_at'] = before+device.offset
                record['request_returned_at'] = returned+device.offset
                record['rtc_source_plan_id'] = rtc_plan['plan_id'] if rtc_plan else None
            if execute:
                stage = 'control_submission'
                heartbeats.check()
                submission_started = time.monotonic()
                body = dict(
                    schema_id=SCHEMA, session=session, observation_id=packet['observation_id'],
                    origin_at=packet['stamp'], actions=actions.tolist(),
                    model_state=observation['state'].tolist(), execution_scope=execution_scope)
                if arrival_handoff:
                    body['rtc_arrival'] = True
                if bootstrap_pending:
                    body.update(bootstrap=True, remaining_video_s=source.duration - first_time)
                    if rtc_enabled:
                        body['wait_on_exhaustion'] = True
                        body['rtc_pipeline'] = True
                else:
                    body['end_at'] = remote_end
                    if resuming:
                        body['resume_from_wait'] = True
                    elif rtc_enabled:
                        body['rtc_source_plan_id'] = rtc_plan['plan_id']
                        if not arrival_handoff:
                            body['rtc_switch_at'] = packet['stamp']+rtc_context['delay_steps']*DT
                prepared = device.call('/stream/prepare', body)
                record.update(prepared=prepared,
                              live_observation={key: value for key, value in packet.items() if key != 'images'})
                if prepared.get('accepted') is False:
                    record.update(discarded=True, discard_stage='prepare', discard_reason=prepared['reason'])
                else:
                    if not np.array_equal(np.asarray(prepared['received_goal_actions']), actions):
                        raise RuntimeError('Gateway did not receive the complete raw H45 prediction')
                    committed = device.call('/stream/commit', dict(plan_id=prepared['plan_id']))
                    record['committed'] = committed
                    if committed.get('accepted') is False:
                        record.update(discarded=True, discard_stage='commit', discard_reason=committed['reason'])
                    else:
                        record['submitted'] = True
                        if rtc_enabled:
                            rtc_plan = dict(actions=actions.copy(), state=observation['state'].copy(),
                                anchor=control_anchor(packet), origin_at=committed['origin_at'],
                                switch_at=committed.get('scheduled_start', committed['origin_at']),
                                plan_id=committed['plan_id'])
                            if arrival_handoff:
                                rtc_plans = [p for p in rtc_plans if p['switch_at'] < rtc_plan['switch_at']]
                            rtc_plans.append(rtc_plan)
                            active = [i for i, p in enumerate(rtc_plans) if p['switch_at'] <= packet['stamp']]
                            if active:
                                rtc_plans = rtc_plans[active[-1]:]
                        record['remaining_prediction_s'] = committed.get('remaining_prediction_s')
                        if bootstrap_pending:
                            timeline = committed['origin_at'] - device.offset - first_time
                            remote_end = committed['end_at']
                            end = remote_end - device.offset
                            bootstrap_pending = False
                record['control_submission_s'] = time.monotonic() - submission_started
                if arrival_handoff and not record['initial_chunk_preloaded']:
                    # Estimate to the dog-side commit, not merely WebSocket
                    # return; include a control period until its next sample.
                    commit = record.get('committed', {})
                    eligible_at = (commit['scheduled_start'] if commit.get('accepted')
                                   else time.monotonic()+device.offset)
                    measured = max(DT, eligible_at-packet['stamp']+DT)
                    prefix_estimator.observe(measured)
                    record['rtc_observed_handoff_delay_s'] = measured
                    record['rtc_next_prefix_estimate_s'] = prefix_estimator.estimate_s
                if arrival_handoff and record.get('submitted'):
                    record['rtc_commit_action_index'] = record['committed']['first_unexpired_action']
                    record['rtc_commit_age_s'] = record['committed']['scheduled_start']-packet['stamp']
            records.append(record)
            with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            last = index
            print(json.dumps(dict(video_s=round(elapsed if rtc_enabled else time.monotonic() - timeline, 2), predictions=len(records),
                                  inference_s=round(record['inference_s'], 3), submitted=record['submitted'],
                                  server_inference_s=round(record['server_inference_s'], 3),
                                  input_preparation_s=round(input_preparation_s, 3),
                                  input_wire_bytes=record['input_wire_bytes'],
                                  rtc_prefix_steps=record.get('rtc_prefix_steps'),
                                  rtc_handoff=record.get('rtc_handoff'),
                                  rtc_commit_action_index=record.get('rtc_commit_action_index'),
                                  resume_from_wait=record.get('resume_from_wait'),
                                  remaining_prediction_s=record.get('remaining_prediction_s'),
                                  discarded=record.get('discarded', False), reason=record.get('discard_reason'))), flush=True)
        stage = 'normal_completion'
        if execute:
            if not any(record['submitted'] for record in records):
                raise RuntimeError('No fresh prediction was accepted; no motion was started')
            deadline = time.monotonic() + 2
            while True:
                final = device.call('/status')
                if final.get('fault'):
                    raise RuntimeError('H45 execution fault: ' + str(final['fault']))
                if not final['active'] and (final.get('stream') or {}).get('finished'):
                    completed_normally = True
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('Recorded-video end stop was not acknowledged')
                time.sleep(.01)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        controller_error = error
        if execute and session is not None:
            try:
                final = device.call('/status')
                root_fault = final.get('fault') or final.get('hardware_fault')
                if root_fault:
                    error = 'RuntimeError: H45 execution fault: ' + str(root_fault)
            except Exception:
                pass
    finally:
        if execute and session is not None:
            try:
                if completed_normally and error is None:
                    stopped = device.call('/stream/finish', {})
                    if not stopped.get('released'):
                        raise RuntimeError('Normal completion was not released')
                else:
                    stopped = device.call('/stop', {})
            except Exception as exc:
                stopped = {'error': str(exc)}
                error = error or 'Completion cleanup failed: ' + str(exc)
                try:
                    stopped['fallback_stop'] = device.call('/stop', {})
                except Exception as stop_exc:
                    stopped['fallback_stop_error'] = str(stop_exc)
        else:
            stopped = None
        if heartbeats is not None:
            heartbeats.close()
    summary = dict(complete=error is None and bool(records), error=error,
                   failure_stage=stage if error else None, controller_error=controller_error,
                   source_episode=source.metadata['source_episode'], source_duration_s=source.duration,
                   checkpoint=policy.metadata['checkpoint'], predictions=len(records),
                   submitted_predictions=sum(record['submitted'] for record in records),
                   discarded_predictions=sum(record.get('discarded', False) for record in records),
                   normal_completion_released=bool(stopped and stopped.get('released')),
                   robot_commands_sent=any(r['submitted'] for r in records),
                   execution_targets=(['go2', 'piper'] if execution_scope == 'unitree' else ['go2']) if execute else [],
                   uav_commands=False, input_source='recorded_three_view_video_and_paired_state14',
                   cached_predictions_used=False, ground_truth_actions_used=False,
                   replacement='new predictions replace remaining actions on the next 15 Hz tick',
                   execution=final, stopped=stopped)
    if rtc_enabled:
        summary.update(rtc_enabled=True, rtc_prefix_steps=rtc_delay_steps,
            rtc_handoff=rtc_handoff,
            rtc_prefix_policy='adaptive_end_to_end_latency' if arrival_handoff else 'fixed',
            rtc_final_prefix_estimate_s=prefix_estimator.estimate_s if arrival_handoff else rtc_delay_s,
            rtc_guided_submissions=sum(r['submitted'] and r.get('rtc_prefix_steps', 0)>0 for r in records),
            resume_submissions=sum(r['submitted'] and r.get('resume_from_wait', False) for r in records),
            replacement=('RTC arrival handoff on the next control tick; indices use actual observation age'
                         if arrival_handoff else 'asynchronous RTC scheduled pipeline; video clock pauses only when action buffer is exhausted'))
    if getattr(device, 'rtc_loopback_only', False):
        summary.update(robot_commands_sent=False, real_hardware_commands_sent=False,
            execution_targets=[], simulated_execution_targets=['go2', 'piper'], rtc_enabled=rtc_enabled)
    save(output / 'summary.json', summary)
    print(json.dumps(dict(complete=summary['complete'], error=error, predictions=len(records),
                          failure_stage=summary['failure_stage'],
                          source_duration_s=source.duration, robot_commands_sent=summary['robot_commands_sent'],
                          summary=str(output / 'summary.json'))), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'infer', 'execute', 'stop'))
    parser.add_argument('--episode', type=Path, default=Path('validation_episode_h45'))
    parser.add_argument('--config', type=Path, default=Path('controller.json'))
    parser.add_argument('--token-file', type=Path, default=Path('token'))
    parser.add_argument('--checkpoint-step', type=int, default=25000)
    parser.add_argument('--rtc', action='store_true')
    parser.add_argument('--rtc-delay-s', type=float, default=1.5,
                        help='Initial guidance prefix estimate in arrival mode; fixed prefix in scheduled mode')
    parser.add_argument('--rtc-handoff', choices=('arrival', 'scheduled'), default='arrival',
                        help='arrival: replace on the next tick when ready; scheduled: legacy fixed-prefix switch')
    parser.add_argument('--scope', choices=('go2', 'unitree'), default='unitree')
    parser.add_argument('--output', type=Path, default=Path('outputs') / ('h45_video_' + time.strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    device = Device(config['unitree'], args.token_file.read_text().strip())
    if args.mode == 'stop':
        print(json.dumps(device.call('/stop', {})), flush=True)
        return
    source = RecordedEpisode(args.episode)
    policy = RemotePolicy(config['policy'], checkpoint_step=args.checkpoint_step)
    def interrupted(*_):
        raise KeyboardInterrupt('Operator stopped recorded-video execution')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'check':
            status = device.calibrate()
            print(json.dumps(dict(gateway=status, policy=policy.metadata,
                clock_uncertainty_ms=device.uncertainty * 1000,
                source_episode=source.metadata['source_episode'], duration_s=source.duration,
                robot_commands_sent=False), indent=2), flush=True)
        else:
            result = run(source, policy, device, args.output, args.mode == 'execute', args.scope,
                rtc_delay_s=args.rtc_delay_s if args.rtc else None, rtc_handoff=args.rtc_handoff)
            if not result['complete']:
                raise SystemExit(1)
    finally:
        source.close()
        policy.close()


if __name__ == '__main__':
    main()
