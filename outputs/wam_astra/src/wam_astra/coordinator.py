"""Caller-driven shadow review boundary around native monitors and RTC storage.

No provider, model, network, robot SDK or actuator is constructed or called.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from threading import RLock
import time
from uuid import uuid4

from .rtc import validate_proposal_timestamp, validate_trajectory


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Context must contain finite JSON data') from exc


@dataclass(frozen=True)
class CoordinatorConfig:
    max_observation_age_s: float = .5
    max_request_age_s: float = 2.
    max_future_start_s: float = .5
    max_horizon: int = 256
    joint_lower: tuple | None = None
    joint_upper: tuple | None = None
    max_velocity: tuple | None = None
    max_acceleration: tuple | None = None

    def __post_init__(self):
        for field in ('max_observation_age_s', 'max_request_age_s', 'max_future_start_s'):
            if _number(getattr(self, field), field) <= 0:
                raise ValueError(f'{field} must be positive')
        if type(self.max_horizon) is not int or self.max_horizon < 1:
            raise ValueError('max_horizon must be a positive integer')
        for field in ('joint_lower', 'joint_upper', 'max_velocity', 'max_acceleration'):
            value = getattr(self, field)
            if value is not None:
                if len(value) != 6 or any(not math.isfinite(_number(x, field)) for x in value):
                    raise ValueError(f'{field} must contain six finite values')
                if field in ('max_velocity', 'max_acceleration') and any(x <= 0 for x in value):
                    raise ValueError(f'{field} must be positive')
                object.__setattr__(self, field, tuple(value))
        if (self.joint_lower is None) != (self.joint_upper is None):
            raise ValueError('Both joint bounds must be configured')
        if self.joint_lower is not None and any(lo >= hi for lo, hi in zip(self.joint_lower, self.joint_upper)):
            raise ValueError('Joint lower bounds must be below upper bounds')


@dataclass(frozen=True)
class ShadowActionProvenance:
    task: str
    request_id: str
    observation_id: str
    queue_generation: int
    target_at_s: float
    plan_observed_at_s: float
    decision: str
    shadow: bool = True


class ShadowCoordinator:
    """Own review order around native queue CAS and adapter trajectory checks.

    The queue must be owned by this coordinator for proposal replacement.
    External consumers may pop; external preemption must invalidate generation.
    This object never consumes a queue action or sends one to any hardware.
    """

    def __init__(self, queue, *, config=None, retrieval_callback=None, monitor=None, clock=time.monotonic):
        self.queue = queue
        self.config = config or CoordinatorConfig()
        self.retrieval_callback = retrieval_callback
        self.monitor = monitor
        self.clock = clock
        self._lock = RLock()
        self._observation = None
        self._pending = None
        self._event = None

    def _now(self, now_s):
        return _number(self.clock() if now_s is None else now_s, 'now_s')

    def _fresh_observation(self, now):
        if self._observation is None:
            raise ValueError('No observation is available')
        age = now - self._observation['observed_at_s']
        if age < 0:
            raise ValueError('Observation is in the future')
        if age > self.config.max_observation_age_s:
            raise ValueError('Latest observation is stale')

    def observe(self, observation: dict, *, monitor_event=None, now_s=None):
        """Record explicit measured q6 + aperture metres, preserving unknowns.

        Observation ``state`` has seven values: six measured radians and one
        measured metre value. Optional ``previous_gripper_normalized`` is a
        separate prior command, not inferred from measured aperture.
        """
        with self._lock:
            now = self._now(now_s)
            obs = _json_copy(observation)
            if not isinstance(obs, dict) or not isinstance(obs.get('observation_id'), str) or not obs['observation_id'].strip():
                raise ValueError('observation_id must be nonempty text')
            timestamp = _number(obs.get('observed_at_s'), 'observed_at_s')
            if timestamp > now:
                raise ValueError('Observation is in the future')
            if obs.get('state_joint_units') != 'rad' or obs.get('state_gripper_units') != 'm':
                raise ValueError('Measured state units must be rad and m')
            state = obs.get('state')
            if not isinstance(state, list) or len(state) != 7:
                raise ValueError('Measured state must contain q6 and gripper metres')
            for value in state:
                _number(value, 'state')
            if 'previous_gripper_normalized' in obs and obs['previous_gripper_normalized'] is not None:
                if _number(obs['previous_gripper_normalized'], 'previous_gripper_normalized') not in (0., 1.):
                    raise ValueError('previous_gripper_normalized must be discrete 0 or 1')
            if self._observation is not None:
                if obs['observation_id'] == self._observation['observation_id']:
                    if obs != self._observation:
                        raise ValueError('An observation_id cannot be reused for different data')
                elif timestamp <= self._observation['observed_at_s']:
                    raise ValueError('Out-of-order observation time')
            event = None
            if monitor_event is not None:
                event = _json_copy(monitor_event)
                if not isinstance(event, dict):
                    raise ValueError('monitor_event must be a native monitor result object')
                if event.get('observation_id') not in (None, '', obs['observation_id']):
                    raise ValueError('Foreign monitor observation_id')
            self._observation = obs
            if event is not None:
                self._event = event
                if event.get('triggered') is True or event.get('status') == 'unavailable':
                    self.queue.invalidate()
                    self._pending = None
            return deepcopy(obs)

    def observe_tracking(self, observation, target, observed, *, now_s=None):
        """Run a supplied native TrackingMonitor, then record its event."""
        if self.monitor is None:
            raise ValueError('A native tracking monitor must be supplied')
        event = self.monitor.update(target, observed, observation_id=observation.get('observation_id', ''))
        self.observe(observation, monitor_event=event, now_s=now_s)
        return event

    def make_review_request(self, task: str, *, demonstrations=None, history=None, now_s=None):
        """Export data and native manifest references; never invoke a provider."""
        with self._lock:
            now = self._now(now_s)
            self._fresh_observation(now)
            if not isinstance(task, str) or not task.strip():
                raise ValueError('task must be nonempty text')
            obs = deepcopy(self._observation)
            demos = demonstrations
            if demos is None and self.retrieval_callback is not None:
                demos = self.retrieval_callback(deepcopy(obs))
            demos = _json_copy([] if demos is None else list(demos))
            past = _json_copy([] if history is None else list(history))
            snapshot = self.queue.snapshot()
            req = {
                'schema_version': 1, 'request_id': uuid4().hex,
                'observation_id': obs['observation_id'], 'observed_at_s': obs['observed_at_s'],
                'created_at_s': now, 'queue_generation': snapshot.generation, 'queue_cursor': snapshot.cursor,
                'task': task.strip(), 'observation': obs, 'demonstrations': demos, 'history': past,
                'monitor_event': deepcopy(self._event), 'shadow': True, 'shadow_only': True,
                'action_contract': {'joint_units': 'rad', 'gripper_units': 'normalized',
                    'gripper_values': [0., 1.], 'columns': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper'],
                    'recorded_targets_are_historical_only': True},
            }
            self._pending = (deepcopy(req), snapshot)
            return req

    def apply_review(self, response: dict, *, now_s=None):
        """Validate a caller-provided proposal and CAS the in-memory future."""
        with self._lock:
            try:
                return self._apply_review(response, self._now(now_s))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                return {'status': 'rejected', 'reason': str(exc), 'shadow': True, 'shadow_only': True}

    def _apply_review(self, response, now):
        if not isinstance(response, dict):
            raise ValueError('Review must be a JSON object')
        if type(response.get('schema_version')) is not int or response['schema_version'] != 1:
            raise ValueError('Unsupported review schema_version')
        common = {'schema_version', 'request_id', 'observation_id', 'queue_generation', 'queue_cursor', 'decision', 'reason'}
        proposal = {'actions', 'action_joint_units', 'action_gripper_units', 'plan_observed_at_s', 'target_start_at_s', 'step_s'}
        if not common.issubset(response) or set(response) - common - proposal - {'evidence'}:
            raise ValueError('Review has missing or unsupported schema fields')
        if self._pending is None:
            raise ValueError('No pending request; review is duplicate or obsolete')
        req, snapshot = self._pending
        if response['request_id'] != req['request_id']:
            raise ValueError('Foreign or out-of-order request_id')
        if response['observation_id'] != req['observation_id'] or self._observation['observation_id'] != req['observation_id']:
            raise ValueError('Foreign or superseded observation_id')
        for field in ('queue_generation', 'queue_cursor'):
            if type(response[field]) is not int or response[field] != req[field]:
                raise ValueError(f'Wrong {field}')
        current = self.queue.snapshot()
        if current.generation != snapshot.generation:
            raise ValueError('Native queue generation changed')
        if current.cursor != snapshot.cursor:
            raise ValueError('Native queue cursor changed')
        self._fresh_observation(now)
        if now < req['created_at_s']:
            raise ValueError('Request time is in the future')
        if now - req['created_at_s'] > self.config.max_request_age_s:
            raise ValueError('Review request is stale')
        decision = response['decision']
        if decision not in ('pass', 'edit', 'recover', 'hold'):
            raise ValueError('Unsupported decision')
        if not isinstance(response['reason'], str) or not response['reason'].strip():
            raise ValueError('Review reason must be nonempty text')
        if 'evidence' in response and (not isinstance(response['evidence'], list) or any(not isinstance(x, str) for x in response['evidence'])):
            raise ValueError('Review evidence must be strings')
        if decision == 'hold':
            if set(response) & proposal:
                raise ValueError('Hold must not contain action proposal fields')
            self.queue.invalidate()
            self._pending = None
            return {'status': 'hold', 'request_id': req['request_id'], 'reason': response['reason'], 'shadow': True, 'shadow_only': True}
        if not proposal.issubset(response):
            raise ValueError('Action proposal schema fields are missing')
        if response['action_joint_units'] != 'rad' or response['action_gripper_units'] != 'normalized':
            raise ValueError('Action units must be rad and normalized')
        if _number(response['plan_observed_at_s'], 'plan_observed_at_s') != req['observed_at_s']:
            raise ValueError('Plan origin does not match the reviewed observation')
        start = _number(response['target_start_at_s'], 'target_start_at_s')
        step = _number(response['step_s'], 'step_s')
        actions = response['actions']
        if not isinstance(actions, list) or not 1 <= len(actions) <= self.config.max_horizon:
            raise ValueError('Action horizon is missing or exceeds configured maximum')
        for row in actions:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError('Actions must have shape H x 7')
            for value in row:
                _number(value, 'Action target')
        expired = validate_proposal_timestamp(start, now, step, len(actions))
        if expired:
            raise ValueError('Proposal contains expired targets; re-dating is forbidden')
        if start > now + self.config.max_future_start_s:
            raise ValueError('Proposal starts too far in the future')
        cfg = self.config
        if any(value is None for value in (cfg.joint_lower, cfg.joint_upper, cfg.max_velocity, cfg.max_acceleration)):
            raise ValueError('All trajectory limits must be explicitly configured')
        prior = self._observation.get('previous_gripper_normalized')
        if prior is None:
            raise ValueError('previous_gripper_normalized is unknown; boundary validation unavailable')
        previous_action = self._observation['state'][:6] + [prior]
        validated = validate_trajectory(actions, step_s=step, joint_lower=cfg.joint_lower,
            joint_upper=cfg.joint_upper, max_velocity=cfg.max_velocity,
            max_acceleration=cfg.max_acceleration, previous_action=previous_action)
        sources = [ShadowActionProvenance(task=req['task'], request_id=req['request_id'],
            observation_id=req['observation_id'], queue_generation=snapshot.generation,
            target_at_s=start + i * step, plan_observed_at_s=req['observed_at_s'], decision=decision)
            for i in range(len(validated))]
        # No interpolation, smoothing, expiry trimming, archived-control replay,
        # or native model-space prefix is manufactured by this coordinator.
        if not self.queue.replace_future(validated, sources, snapshot=snapshot):
            raise ValueError('Native queue generation or cursor changed during atomic replacement')
        self._pending = None
        return {'status': 'accepted', 'request_id': req['request_id'], 'decision': decision,
                'action_count': len(validated), 'target_start_at_s': start, 'step_s': step,
                'shadow': True, 'shadow_only': True}
