"""Authenticated device-local observation, prepare/commit, and deadman service."""
import argparse
import hmac
import json
import math
import os
import signal
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .core import DT, action_plan, angle_delta, finite, command_limits
from .action_stream import ActionBuffer
from .review_pause import settings as review_settings, HoldWindow
from .review_deadline import DeadlineGrant


class ReviewRejected(ValueError):
    """An obsolete or paused request has no dispatch authority."""


class Gateway:
    def __init__(self, hardware, execute=False, velocity_scale=1., max_action_age=.75,
                 velocity_limits=None, review_pause=None):
        if velocity_scale != 1. or not 0 < max_action_age <= 1:
            raise ValueError('Raw execution requires velocity_scale=1; action age must be (0,1] seconds')
        self.review_config = None if review_pause is None else review_settings(review_pause)
        self.review_epoch = 0
        self.review_phase = "open"
        self.review_requested = threading.Event()
        self.dispatch_lock = threading.RLock()
        self.review_lock = threading.RLock()
        self.review_monitor_stop = threading.Event()
        self.review_monitor = None
        self.hold_window = None
        self.review_started = None
        self.hold_receipt = None
        self.keep_review_monitor_until_commit = False
        self.resume_monitoring = False
        self.review_deadline = None
        self.hardware = hardware
        self.kind = hardware.sensors.kind
        self.action_kind = getattr(hardware, 'action_kind', self.kind)
        self.execute = execute
        self.velocity_scale = velocity_scale
        limits = command_limits(self.action_kind, velocity_limits)
        self.velocity_limits = False if velocity_limits is False else limits
        self.stream_deadline = None
        self.max_action_age = max_action_age
        self.session = None
        self.lock = threading.RLock()
        self.snapshots = OrderedDict()
        self.plan = None
        self.stream = ActionBuffer()
        self.stream_mode = False
        self.stream_prepared = None
        self.fault = None
        self.cancel = threading.Event()
        self.closed = threading.Event()
        self.active = False
        self.worker = None
        self.last_tick = time.monotonic()
        self.last_controller_at = self.last_tick
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _check_review_credentials(self, request, *, paused=False, new_pause=False):
        if request.get('session') != self.session or self.session is None:
            raise ReviewRejected('Wrong episode session')
        # Enabled deployments require the epoch on every action message.
        # Legacy deployments without review configuration retain epoch-zero API.
        if self.review_config is not None or self.review_requested.is_set():
            if type(request.get('review_epoch')) is not int or request['review_epoch'] != self.review_epoch:
                raise ReviewRejected('Wrong/missing review epoch')
        if not paused and self.review_phase != 'open':
            raise ReviewRejected('Gateway is review-paused; fresh resume is required')
        if not new_pause:self._check_review_deadline(request)

    def _check_review_deadline(self,request):
        if self.review_deadline is not None:
            try:self.review_deadline.validate(request,now=time.monotonic())
            except ValueError as exc:raise ReviewRejected(str(exc)) from exc
        elif (self.review_config is not None and self.review_config['require_deadline_binding']
                and (self.review_phase!='open' or self.resume_monitoring)):
            raise ReviewRejected('Missing review deadline grant')

    def _dispatch_row(self, row, epoch):
        # A sampled row is not permission to dispatch. Recheck after acquiring
        # the same barrier used by pause, including an already blocked sender.
        with self.dispatch_lock:
            with self.lock:
                if (self.cancel.is_set() or self.review_requested.is_set()
                        or self.fault or epoch != self.review_epoch):
                    return False
            self.hardware.send(row)
            return True

    def _finish_hold(self):
        # Review-enabled deployments never fall through to legacy hold(), which
        # writes an enable opcode to the gripper. A normal stream end is also a
        # review pause, not repetition of nonzero base velocity.
        if self.review_config is not None:
            def completed_pause():
                with self.review_lock:
                    if self.review_requested.is_set() or self.fault or self.closed.is_set():
                        return  # An external pause/stop already owns cancellation.
                    try:
                        request=dict(session=self.session,review_epoch=self.review_epoch)
                        if self.review_config['require_deadline_binding']:
                            request.update(review_transaction_id='device_hold_'+uuid.uuid4().hex,
                                review_deadline_s=time.monotonic()+self.review_config['max_review_s'])
                        self.pause_review(request)
                    except Exception as exc:
                        if not self.fault:self.stop('Automatic review hold failed: '+str(exc))
            if self.worker is threading.current_thread():
                # A worker cannot join itself. A separate coordinator joins it
                # before the measured hold and avoids joining a worker blocked
                # on a review_lock held by an external pause request.
                threading.Thread(target=completed_pause, daemon=True).start()
            else:
                completed_pause()
        else:
            with self.dispatch_lock:
                self.hardware.hold()

    def review_status(self):
        with self.lock:
            result = dict(configured=self.review_config is not None,
                          phase=self.review_phase, review_epoch=self.review_epoch,
                          session=self.session, ready=False, task_outcome='unknown',
                          physical_hold_proven=False)
            result['review_deadline_required']=bool(self.review_config and self.review_config['require_deadline_binding'])
            if self.review_deadline is not None:
                result.update(self.review_deadline.fields(),review_deadline_active=not self.review_deadline.released)
            if self.hold_window is not None:
                result.update(self.hold_window.summary())
            if self.fault or self.review_phase != 'holding':
                result['ready'] = False
            if self.review_deadline is not None and not self.review_deadline.released:
                if time.monotonic()>=self.review_deadline.expires_s:result['ready']=False
            return result

    def pause_review(self, request):
        if self.review_config is None or self.kind != 'unitree' or not self.execute:
            raise RuntimeError('Unitree execution review pause must be explicitly configured')
        with self.review_lock:
            with self.lock:
                self._check_review_credentials(request,new_pause=True)
                if self.fault:
                    raise RuntimeError(self.fault)
                entered=time.monotonic()
                grant=DeadlineGrant.from_pause(request,now=entered,max_review_s=self.review_config['max_review_s'],
                    previous=self.review_deadline,required=self.review_config['require_deadline_binding'])
                self.review_deadline=grant
                self.review_phase = 'pausing'
                self.review_epoch += 1
                self.review_requested.set()
                self.cancel.set()
                self.snapshots.clear()
                self.plan = None
                self.stream_prepared = None
                self.stream = ActionBuffer()
                self.stream_mode = False
                self.stream_deadline = None
                self.review_started = entered
                self.resume_monitoring = False
                worker = self.worker
            try:
                # Existing dispatch completes before STOP. Sampled but not yet
                # dispatched rows see the invalidated epoch and cannot follow it.
                acquired = self.dispatch_lock.acquire(timeout=self.review_config['worker_join_timeout_s'])
                if not acquired:
                    # No positive pause acknowledgement is possible while an
                    # existing hardware call is still in flight. Latch failure
                    # before scheduling a best-effort halt; never block this
                    # handler indefinitely behind that same call.
                    with self.lock:
                        self.fault = 'Review dispatch barrier timed out; physical halt unverified'
                    threading.Thread(target=self.stop, args=(self.fault,), daemon=True).start()
                    raise RuntimeError(self.fault)
                self.dispatch_lock.release()
                if worker is not None and worker is not threading.current_thread():
                    worker.join(timeout=self.review_config['worker_join_timeout_s'])
                    if worker.is_alive():
                        raise RuntimeError('Previous action worker did not stop')
                with self.dispatch_lock:
                    receipt = self.hardware.review_hold(self.review_config)
                now = time.monotonic()
                if self.review_deadline is not None:self.review_deadline.check_time(now)
                window = HoldWindow(self.review_config, receipt['stop_response'],
                                    receipt['initial_feedback'], now)
                with self.lock:
                    # A sender can finish bookkeeping after cancellation but
                    # before join. Invalidate those records after it exits too.
                    self.plan = None
                    self.stream_prepared = None
                    self.snapshots.clear()
                    self.hold_receipt = receipt
                    self.hold_window = window
                    self.active = False
                    self.review_phase = 'holding'
                    self.review_monitor_stop = threading.Event()
                    self.review_monitor = threading.Thread(target=self._monitor_review, daemon=True)
                    self.review_monitor.start()
            except Exception as exc:
                if self.fault is None:
                    self.stop('Review pause failed: '+str(exc))
                raise
            return self.status()

    def _monitor_review(self):
        stop = self.review_monitor_stop
        while not stop.wait(self.review_config['monitor_period_s']):
            self.poll_review()

    def poll_review(self):
        """One actual feedback poll; the background monitor uses this same path."""
        with self.review_lock:
            with self.lock:
                if (self.review_phase != 'holding' and not self.resume_monitoring) or self.fault:
                    return self.review_status()
            try:
                if self.review_deadline is not None:self.review_deadline.check_time(time.monotonic())
                if time.monotonic()-self.review_started > self.review_config['max_review_s']:
                    raise RuntimeError('Review deadline expired; heartbeat cannot extend it')
                with self.dispatch_lock:
                    frame = self.hardware.review_feedback(self.review_config)
                now = time.monotonic()
                if self.review_deadline is not None:self.review_deadline.check_time(now)
                with self.lock:
                    self.hold_window.update(frame, now)
            except Exception as exc:
                self.stop('Review feedback failed: '+str(exc))
            return self.review_status()

    def resume_review(self, request):
        with self.review_lock:
            with self.lock:
                self._check_review_credentials(request, paused=True)
                if self.review_phase != 'holding' or self.fault:
                    raise RuntimeError('No healthy review hold to resume')
                snapshot = self.snapshots.get(request.get('observation_id'))
                if (snapshot is None or snapshot.get('review_epoch') != self.review_epoch
                        or snapshot['stamp'] <= self.review_started
                        or not 0 <= time.monotonic()-snapshot['stamp'] <= self.max_action_age):
                    raise ValueError('Resume requires a fresh pause-era observation')
            result = self.poll_review()
            if not result['ready']:
                raise RuntimeError('Measured hold is not ready: '+result.get('reason', 'unknown'))
            with self.lock:
                # Fresh prepare/commit still required; no row is replayed here.
                self._check_review_deadline(request)
                if self.keep_review_monitor_until_commit:
                    self.resume_monitoring = True
                else:
                    self.review_monitor_stop.set()
                self.review_phase = 'open'
                self.review_requested.clear()
                self.cancel.clear()
                self.worker = None
            return self.status()

    def status(self):
        with self.lock:
            return dict(kind=self.kind, monotonic=time.monotonic(), session=self.session,
                        capabilities=['validate_actions_v1', 'stream_actions_v1',
                                      'stream_hold_heartbeat_v1', 'review_pause_v1'],
                        review_epoch=self.review_epoch, review=self.review_status(),
                        action_transform='identity',
                        velocity_limits=None if self.velocity_limits is False else self.velocity_limits.tolist(),
                        velocity_limits_enforced=self.velocity_limits is not False,
                        action_kind=self.action_kind,
                        execution_enabled=self.execute, fault=self.fault, active=self.active,
                        sensor_error=self.hardware.sensors.error,
                        hardware_fault=getattr(self.hardware, 'fault', None),
                        input_errors=getattr(self.hardware.sensors, 'input_errors', {}),
                        stream=self.stream.status() if self.stream_mode else None,
                        execution=None if self.plan is None else {
                            k: v for k, v in self.plan.items() if k not in ('rows', 'snapshot')})

    def reset(self):
        with self.lock:
            if self.active or self.fault:
                raise RuntimeError('Active/faulted gateway cannot reset; restart after inspecting the fault')
            if self.review_phase != "open":
                raise RuntimeError("Cannot reset during review pause")
            if self.review_deadline is not None and not self.review_deadline.released:
                raise RuntimeError('Cannot reset before review grant is consumed')
            self.review_epoch += 1
            self.review_deadline=None
            self.hardware.sensors.reset()
            self.session = uuid.uuid4().hex
            self.snapshots.clear()
            self.plan = None
            self.stream = ActionBuffer()
            self.stream_mode = False
            self.stream_prepared = None
            self.stream_deadline = None
            return self.status()

    def observe(self):
        with self.lock:
            if self.fault or self.session is None:
                raise RuntimeError(self.fault or 'Reset the episode first')
            snapshot = self.hardware.sensors.snapshot()
            snapshot.update(observation_id=uuid.uuid4().hex, session=self.session,
                            review_epoch=self.review_epoch)
            self.snapshots[snapshot['observation_id']] = {
                k: v for k, v in snapshot.items() if k != 'images'}
            while len(self.snapshots) > 32:
                self.snapshots.popitem(last=False)
            return snapshot

    def prepare(self, request):
        with self.lock:
            self._check_review_credentials(request)
            if not self.execute or self.fault or self.active or self.stream_mode:
                raise RuntimeError('Gateway is observe-only, faulted, or executing')
            if request.get('session') != self.session or self.session is None:
                raise ValueError('Wrong episode session')
            if self.plan and self.plan['status'] in ('prepared', 'scheduled', 'running'):
                raise RuntimeError('A plan is already pending')
            snapshot = self.snapshots.get(request.get('observation_id'))
            if snapshot is None:
                raise ValueError('Unknown/consumed observation id')
            if not 0 <= time.monotonic()-snapshot['stamp'] <= self.max_action_age:
                raise RuntimeError('Model actions expired; inference result will not execute')
            self.hardware.check_execution()
            state_snapshot = getattr(self.hardware.sensors, 'execution_snapshot', self.hardware.sensors.snapshot)
            current = state_snapshot()
            old_state, state = snapshot['state'], current['state']
            if sum((a-b)**2 for a, b in zip(state[:2], old_state[:2])) > .1**2:
                raise RuntimeError('Robot moved while inference was running')
            yaw_change = angle_delta(math.atan2(old_state[5], old_state[6]), math.atan2(state[5], state[6]))
            if abs(yaw_change) > .15 or (self.kind == 'p450' and abs(state[2]-old_state[2]) > .08):
                raise RuntimeError('Robot orientation/altitude changed while inference was running')
            rows = action_plan(request['actions'], self.kind,
                               current_arm=state[10:17] if self.kind == 'unitree' else None,
                               velocity_scale=self.velocity_scale, velocity_limits=self.velocity_limits)
            self.snapshots.pop(request['observation_id'])
            self.plan = dict(plan_id=uuid.uuid4().hex, status='prepared', rows=rows,
                             snapshot=snapshot, review_epoch=self.review_epoch, session=self.session,
                             prepared_at=time.monotonic(), executed_steps=0)
            return dict(plan_id=self.plan['plan_id'], status='prepared', applied_actions=rows.tolist())

    def prepare_stream(self, request):
        with self.lock:
            self._check_review_credentials(request)
            if not self.execute or self.fault or (self.active and not self.stream_mode):
                raise RuntimeError('Gateway cannot accept a streaming prediction')
            if request.get('session') != self.session or self.session is None:
                raise ValueError('Wrong episode session')
            snapshot = self.snapshots.get(request.get('observation_id'))
            if snapshot is None:
                raise ValueError('Unknown/consumed observation id')
            now = time.monotonic()
            origin = float(request['origin_at'])
            if (not math.isfinite(origin) or origin > snapshot['stamp']+.001
                    or snapshot['stamp']-origin > .15
                    or not 0 <= now-snapshot['stamp'] <= self.max_action_age):
                raise RuntimeError('Prediction origin or observation age is invalid')
            width = {'unitree': 10, 'go2': 3, 'p450': 4}[self.action_kind]
            rows = finite(request['actions'], (15, width))
            index = max(0, int((now-origin)/DT))
            if index >= len(rows):
                raise RuntimeError('Prediction horizon already expired')
            self.hardware.check_execution()
            current_arm = None
            if self.action_kind == 'unitree':
                if hasattr(self.hardware, 'arm_feedback'):
                    current_arm = self.hardware.arm_feedback()[0]
                else:
                    current_arm = self.hardware.sensors.snapshot()['state'][10:17]
            # Check every target without modifying it. Initial physical tracking
            # is checked against the first target that can still be executed.
            action_plan(rows, self.action_kind, rows[0, 3:] if self.action_kind == 'unitree' else None,
                        velocity_scale=self.velocity_scale, velocity_limits=self.velocity_limits)
            action_plan(rows[index:], self.action_kind,
                        current_arm,
                        velocity_scale=self.velocity_scale, velocity_limits=self.velocity_limits)
            self.snapshots.pop(request['observation_id'])
            self.stream_prepared = dict(plan_id=uuid.uuid4().hex, rows=rows.copy(),
                snapshot=snapshot, review_epoch=self.review_epoch, session=self.session,
                origin_at=origin, status='prepared', executed_steps=0)
            hold_last = request.get('hold_last_action', self.stream.hold_last)
            if not isinstance(hold_last, bool) or (hold_last and self.kind != 'unitree'):
                raise ValueError('Last-action holding is supported only for Unitree')
            if self.stream_mode and hold_last != self.stream.hold_last:
                raise ValueError('Cannot change the waiting policy during an action stream')
            self.stream.hold_last = hold_last
            if 'end_at' in request:
                end = float(request['end_at'])
                if not math.isfinite(end) or not now < end <= now+300:
                    raise ValueError('Invalid fixed-video end time')
                self.stream_prepared['end_at'] = end
            return dict(plan_id=self.stream_prepared['plan_id'], status='prepared',
                        applied_actions=rows.tolist(), expired_prefix_steps=index,
                        expires_at=origin+len(rows)*DT)

    def commit_stream(self, request):
        with self.lock:
            self._check_review_credentials(request)
            plan = self.stream_prepared
            if self.fault or plan is None or request.get('plan_id') != plan['plan_id']:
                raise ValueError('No matching streaming prediction')
            start = float(request['start_at'])
            now = time.monotonic()
            if not math.isfinite(start) or not .025 <= start-now <= .4:
                raise ValueError('Stream start must be 25..400 ms in the future')
            if start-plan['snapshot']['stamp'] > self.max_action_age:
                raise RuntimeError('Streaming prediction will be stale at scheduled start')
            index = int((start-plan['origin_at'])/DT)
            if index >= len(plan['rows']):
                raise RuntimeError('No unexpired prediction remains at scheduled start')
            self.hardware.check_execution()
            end = plan.get('end_at')
            if self.stream_deadline is not None and end != self.stream_deadline:
                raise ValueError('A fixed-video stream cannot change its end time')
            if end is not None:
                self.stream_deadline = end
            plan.update(status='scheduled', scheduled_start=start)
            self.stream.schedule(plan)
            self.last_controller_at = now
            self.stream_prepared = None
            if not self.active:
                self.stream_mode = True
                self.active = True
                self.cancel.clear()
                self.last_tick = now
                self.worker = threading.Thread(target=self._run_stream, daemon=True)
                self.worker.start()
            return dict(plan_id=plan['plan_id'], status='scheduled',
                        first_unexpired_action=index)

    def finish_stream(self):
        with self.lock:
            if not self.stream_mode or self.fault:
                raise RuntimeError('No healthy action stream to finish')
            self.stream.drain = True
            return self.status()

    def heartbeat(self, request):
        with self.lock:
            if (not self.execute or self.fault or self.session is None
                    or request.get('session') != self.session):
                raise RuntimeError('Heartbeat requires a healthy execution session')
            self.last_controller_at = time.monotonic()
            return dict(accepted=True, monotonic=self.last_controller_at)

    def _run_stream(self):
        due = None
        try:
            while not self.cancel.is_set():
                now = time.monotonic()
                with self.lock:
                    if self.stream.hold_last and now-self.last_controller_at > 1.:
                        raise RuntimeError('Inference controller heartbeat expired')
                    if self.stream_deadline is not None and now >= self.stream_deadline:
                        self.stream.finished = True
                        break
                    sampled = self.stream.sample(now)
                    if (self.review_config is not None and sampled is not None
                            and now >= sampled[0]['origin_at']+len(sampled[0]['rows'])*DT):
                        break  # Never dispatch an exhausted hold_last row.
                    if self.stream.finished:
                        break
                    if sampled is not None and due is None:
                        due = now
                    self.last_tick = now
                if sampled is None or now < due:
                    self.cancel.wait(.002)
                    continue
                if now-due > .03:
                    raise RuntimeError('Stream control loop missed an action tick')
                plan, index = sampled
                if not self._dispatch_row(plan['rows'][index], plan['review_epoch']):
                    break
                dispatched = time.monotonic()
                with self.lock:
                    self.stream.record_dispatch(plan, index, dispatched)
                    self.plan = plan
                    self.last_tick = dispatched
                due += DT
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

    def diagnose(self):
        with self.lock:
            if self.execute or self.active:
                raise RuntimeError('Diagnostic observations require an observe-only gateway')
            # No session/observation nonce is issued or stored: this data cannot prepare actions.
            return self.hardware.sensors.snapshot(diagnostic=True)

    def commit(self, request):
        with self.lock:
            self._check_review_credentials(request)
            if self.fault or self.active or self.plan is None or self.plan['status'] != 'prepared':
                raise RuntimeError('No committable plan')
            if request.get('plan_id') != self.plan['plan_id']:
                raise ValueError('Wrong plan id')
            start = float(request['start_at'])
            now = time.monotonic()
            if not math.isfinite(start) or not .025 <= start-now <= .4:
                raise ValueError('Start time must be 25..400 ms in the future')
            if start-self.plan['snapshot']['stamp'] > self.max_action_age:
                raise RuntimeError('Actions will be stale at scheduled execution time')
            self.hardware.check_execution()
            self.plan.update(status='scheduled', scheduled_start=start)
            self.cancel.clear()
            self.active = True
            self.last_tick = now
            self.worker = threading.Thread(target=self._run, args=(self.plan,), daemon=True)
            self.worker.start()
            return dict(plan_id=self.plan['plan_id'], status='scheduled')

    def validate_actions(self, request):
        """Receive and check a real action payload without creating an execution plan."""
        with self.lock:
            if self.execute or self.active:
                raise RuntimeError('Action validation requires an observe-only gateway')
            request_id = request.get('request_id')
            if not isinstance(request_id, str) or len(request_id) != 32:
                raise ValueError('Expected a 32-character request id')
            values = request.get('actions')
            if not isinstance(values, list) or not 1 <= len(values) <= 15:
                raise ValueError('Expected 1..15 action rows')
            rows = finite(values, (len(values), 10 if self.kind == 'unitree' else 4))
            result = dict(kind=self.kind, request_id=request_id, received=True,
                          received_actions=rows.tolist(), received_at=time.monotonic(),
                          executed=False, executable=False, validation_only=True,
                          validation_passed=False, candidate_actions=None,
                          execution_readiness_checked=False, error=None)
            try:
                # Diagnostic snapshot retains grounded/unarmed state. No actuator methods,
                # observation credential, plan id, or worker are created by this endpoint.
                snapshot = self.hardware.sensors.snapshot(diagnostic=True)
                result['readiness_issues'] = snapshot.get('readiness_issues', [])
                result['feedback_stamp'] = snapshot['stamp']
                candidate = action_plan(rows, self.kind,
                    current_arm=snapshot['state'][10:17] if self.kind == 'unitree' else None,
                    velocity_scale=self.velocity_scale, velocity_limits=self.velocity_limits)
                result.update(validation_passed=True, candidate_actions=candidate.tolist())
            except (ValueError, RuntimeError) as exc:
                result['error'] = str(exc)
            return result

    def _run(self, plan):
        try:
            start = plan['scheduled_start']
            while time.monotonic() < start:
                self.last_tick = time.monotonic()
                if self.cancel.wait(.005):
                    return
            if time.monotonic()-start > .03:
                raise RuntimeError('Missed execution start deadline')
            self.hardware.check_execution()
            plan.update(status='running', actual_start=time.monotonic())
            previous = -1
            while not self.cancel.is_set():
                now = time.monotonic()
                index = int((now-start)/DT)
                if index >= len(plan['rows']):
                    break
                if index > previous+1:
                    raise RuntimeError('Control loop missed an action interval')
                if not self._dispatch_row(plan['rows'][index], plan['review_epoch']):
                    break
                self.last_tick = time.monotonic()
                plan['executed_steps'] = index+1
                previous = index
                self.cancel.wait(.005)
            if self.cancel.is_set():
                return
            self._finish_hold()
            with self.lock:
                if not self.fault:
                    plan.update(status='done', finished_at=time.monotonic())
        except Exception as exc:
            self.stop(str(exc))
        finally:
            with self.lock:
                self.active = False

    def _watch(self):
        while not self.closed.wait(.02):
            if self.active and time.monotonic()-self.last_tick > .2:
                self.stop('Gateway control loop watchdog expired')

    def stop(self, reason='Operator/controller stop'):
        with self.lock:
            self.review_monitor_stop.set()
            if self.hold_window is not None:
                self.hold_window.ready = False
            self.fault = self.fault or str(reason)
            self.cancel.set()
            if self.plan and self.plan['status'] != 'done':
                self.plan.update(status='fault', error=self.fault)
        try:
            if self.execute:
                with self.dispatch_lock:
                    self.hardware.halt()
        except Exception as exc:
            with self.lock:
                self.fault += '; stop error: ' + str(exc)
        return self.status()

    def close(self):
        self.stop('Gateway shutdown')
        self.closed.set()
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=2)
        self.hardware.close()


def handler(service, token):
    class Handler(BaseHTTPRequestHandler):
        def dispatch(self):
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token):
                self.send_error(403)
                return
            parsed = urlparse(self.path)
            try:
                if self.command == 'GET' and parsed.path == '/status':
                    result = service.status()
                elif self.command == 'GET' and parsed.path == '/review/status':
                    result = service.review_status()
                elif self.command == 'GET' and parsed.path == '/observation':
                    result = service.observe()
                elif self.command == 'GET' and parsed.path == '/preview':
                    result = service.hardware.sensors.preview()
                    result['hardware_fault'] = getattr(service.hardware, 'fault', None)
                    result['mode'] = getattr(service.hardware, 'mode', None)
                    result['callback_errors'] = getattr(service.hardware, 'callback_errors', {})
                elif self.command == 'GET' and parsed.path == '/diagnostic':
                    result = service.diagnose()
                elif self.command == 'POST':
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 65536:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    if parsed.path == '/review/pause':
                        result = service.pause_review(body)
                    elif parsed.path == '/review/resume':
                        result = service.resume_review(body)
                    elif parsed.path == '/reset':
                        result = service.reset()
                    elif parsed.path == '/prepare':
                        result = service.prepare(body)
                    elif parsed.path == '/commit':
                        result = service.commit(body)
                    elif parsed.path == '/stream/prepare':
                        result = service.prepare_stream(body)
                    elif parsed.path == '/stream/commit':
                        result = service.commit_stream(body)
                    elif parsed.path == '/stream/finish':
                        result = service.finish_stream()
                    elif parsed.path == '/stream/heartbeat':
                        result = service.heartbeat(body)
                    elif parsed.path == '/validate_actions':
                        result = service.validate_actions(body)
                    elif parsed.path == '/stop':
                        result = service.stop()
                    else:
                        raise ValueError('Unknown endpoint')
                else:
                    raise ValueError('Unknown endpoint')
                encoded = json.dumps(result, allow_nan=False).encode()
                self.send_response(200)
            except Exception as exc:
                if (parsed.path in ('/prepare', '/commit', '/stream/prepare', '/stream/commit')
                        and not isinstance(exc, ReviewRejected)):
                    service.stop(str(exc))
                encoded = json.dumps(dict(error=str(exc),review_rejected=isinstance(exc,ReviewRejected))).encode()
                self.send_response(409)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch()

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--token-file', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    token = args.token_file.expanduser().read_text().strip() if args.token_file else os.environ.get('HETEROVLA_ROBOT_TOKEN', '')
    if len(token) < 24:
        parser.error('Set HETEROVLA_ROBOT_TOKEN to a shared random token of at least 24 characters')
    if config['kind'] == 'unitree':
        from .unitree import UnitreeHardware
        hardware = UnitreeHardware(config, args.execute)
    elif config['kind'] == 'p450':
        from .p450 import P450Hardware
        hardware = P450Hardware(config, args.execute)
    else:
        parser.error('Unknown hardware kind')
    service = Gateway(hardware, args.execute, config.get('velocity_scale', 1.),
                      config.get('max_action_age_s', .75), config.get('velocity_limits'),
                      review_pause=config.get('review_pause'))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', config['port']), handler(service, token))
    except Exception:
        service.close()
        raise
    def terminate(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(json.dumps(dict(listening='127.0.0.1:%s' % config['port'], kind=config['kind'],
                          execute=args.execute)), flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    finally:
        service.close()
        server.server_close()


if __name__ == '__main__':
    main()
