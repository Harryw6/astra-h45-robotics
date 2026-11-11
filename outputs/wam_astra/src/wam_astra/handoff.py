"""Configured action-stream gates and symbolic handoff; no model or robot I/O.

This is integration glue around the native RTC queue and reviewer contract,
not a learned failure detector, grasp verifier, controller or paper reproduction.
All physical-state/hold assertions are explicitly caller attestations.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from threading import RLock
from uuid import uuid4

from .review_contract import validate_review_reply


def _copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("input must contain finite JSON values") from exc


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


class HandoffSupervisor:
    """One active review gate; caller exclusively owns queue submission.

    Freshness and the actual discrete gripper encoding must be configured.
    This wrapper never submits, consumes, retimes or executes queue actions.
    """

    def __init__(self, queue, *, gripper_open, gripper_closed, clock,
                 max_observation_age_s, max_hold_receipt_age_s,
                 action_joint_units="rad", action_gripper_units="normalized"):
        self.gripper_open = _number(gripper_open, "gripper_open")
        self.gripper_closed = _number(gripper_closed, "gripper_closed")
        if {self.gripper_open, self.gripper_closed} != {0., 1.}:
            raise ValueError("explicit normalized gripper encoding must be distinct 0 and 1")
        if action_joint_units != "rad" or action_gripper_units != "normalized":
            raise ValueError("canonical action units must be explicit rad and normalized")
        self.max_observation_age_s = _number(max_observation_age_s, "max_observation_age_s")
        self.max_hold_receipt_age_s = _number(max_hold_receipt_age_s, "max_hold_receipt_age_s")
        if min(self.max_observation_age_s, self.max_hold_receipt_age_s) <= 0 or not callable(clock):
            raise ValueError("freshness bounds must be positive and clock callable")
        self.queue, self.clock = queue, clock
        self._lock = RLock()
        self._last_event = self._last_observation = self._gate = None
        self._requested_gate_ids = set()

    def _now(self):
        now = _number(self.clock(), "clock")
        if now < 0:
            raise ValueError("clock must be nonnegative seconds")
        return now

    def _observation(self, value, now):
        obs = _copy(value)
        if not isinstance(obs, dict):
            raise ValueError("observation must be an object")
        for key in ("observation_id", "task_id", "attempt_id", "episode_id"):
            _text(obs.get(key), key)
        if "case_id" in obs:
            _text(obs["case_id"], "case_id")
        if obs.get("state_verified") is not True or obs.get("source") != "measured_controller_feedback":
            raise ValueError("fresh measured state requires explicit caller attestation")
        if obs.get("state_joint_units") != "rad" or obs.get("state_gripper_units") != "m":
            raise ValueError("measured state units must be rad and m")
        state = obs.get("state")
        if not isinstance(state, list) or len(state) != 7:
            raise ValueError("measured state requires six joints and aperture metres")
        for value in state:
            _number(value, "measured state")
        if state[6] < 0:
            raise ValueError("measured aperture metres cannot be negative")
        timestamp = _number(obs.get("observed_at_s"), "observed_at_s")
        if timestamp < 0 or not 0 <= now - timestamp <= self.max_observation_age_s:
            raise ValueError("observation is stale or in the future")
        if obs.get("task_status") in ("cancelled", "superseded"):
            raise ValueError("task is cancelled or superseded")
        return obs

    def _event(self, kind, reason, obs=None, **fields):
        value = {"schema_version": 1, "gate_id": uuid4().hex, "request_id": uuid4().hex,
                 "case_id": obs.get("case_id", uuid4().hex) if obs else uuid4().hex,
                 "kind": kind, "reason": reason, "review_required": kind != "no_gate",
                 "queue_committed": False, "executed_actions": 0, "physical_actions_validated": False,
                 "origin": "configured_action_stream_gate", "state_verified_is_caller_attestation": True,
                 **fields}
        if obs:
            value.update({key: obs[key] for key in ("observation_id", "task_id", "attempt_id", "episode_id")})
        self._last_event, self._last_observation = deepcopy(value), deepcopy(obs)
        return value

    def inspect_proposal(self, canonical_actions, observation, previous_gripper_command):
        """Find an explicit open-to-closed command edge anywhere in the chunk.

        This does not infer polarity from aperture, image semantics or motion.
        A returned gate must be requested before any closing action is committed.
        """
        with self._lock:
            obs = None
            try:
                obs = self._observation(observation, self._now())
                _text(obs.get("case_id", "generated_case"), "case_id")
                previous = _number(previous_gripper_command, "previous_gripper_command")
                encoding = (self.gripper_open, self.gripper_closed)
                if previous not in encoding:
                    raise ValueError("previous_gripper_command is unknown in configured command encoding")
                actions = _copy(canonical_actions)
                if not isinstance(actions, list) or not actions:
                    raise ValueError("canonical actions require a nonempty H x 7 JSON list")
                closing_index = None
                for index, action in enumerate(actions):
                    if not isinstance(action, list) or len(action) != 7:
                        raise ValueError("canonical actions require H x 7")
                    for value in action:
                        _number(value, "canonical action")
                    command = action[6]
                    if command not in encoding:
                        raise ValueError("gripper command does not match explicit configured encoding")
                    if closing_index is None and previous == self.gripper_open and command == self.gripper_closed:
                        closing_index = index
                    previous = command
                if closing_index is None:
                    return self._event("no_gate", "no_configured_open_to_closed_transition", obs)
                return self._event("pre_grasp", "configured_open_to_closed_transition", obs,
                                   closing_action_index=closing_index, grasp_success="unknown")
            except (ValueError, TypeError, KeyError) as exc:
                return self._event("hold", str(exc), obs)

    def _queue_unchanged(self, gate):
        current = self.queue.snapshot()
        if (current.generation != gate["binding"]["queue_generation"]
                or current.cursor != gate["binding"]["queue_cursor"] or not self.queue.empty()):
            raise ValueError("native queue changed or became nonempty during gate")

    def _hold_receipt(self, value, observation_id, requested_at, now):
        receipt = _copy(value)
        if not isinstance(receipt, dict):
            raise ValueError("local hold receipt is missing")
        if (receipt.get("confirmed") is not True or receipt.get("base_velocity_zero") is not True
                or receipt.get("source") != "measured_controller_feedback"):
            raise ValueError("confirmed joint/base hold caller attestation is missing")
        if receipt.get("observation_id") != observation_id or receipt.get("joint_units") != "rad":
            raise ValueError("local hold receipt has foreign observation or units")
        held = receipt.get("held_joint_positions_rad")
        if not isinstance(held, list) or len(held) != 6:
            raise ValueError("local hold receipt requires measured held q6")
        for value in held:
            _number(value, "held joint position")
        checked = _number(receipt.get("checked_at_s"), "hold checked_at_s")
        if checked < requested_at or not 0 <= now - checked <= self.max_hold_receipt_age_s:
            raise ValueError("local hold attestation is stale or in the future")
        return receipt

    def _gate_result(self):
        gate = self._gate
        return {"status": gate["status"], "reason": gate["reason"], "binding": deepcopy(gate["binding"]),
                "review_ready": gate["status"] == "review_ready",
                "local_hold_attestation_accepted": gate.get("hold_receipt") is not None,
                "physical_hold_verified_by_wrapper": False,
                "state_verified_is_caller_attestation": True, "queue_committed": False,
                "model_calls": 0, "actuator_calls": 0}

    def request_gate(self, event, obs, hold_callback=None):
        """Invalidate native future, then request an explicitly supplied local hold.

        The callback is caller code, not a transport implemented here. Its
        receipt is an attestation, not hardware verification by this wrapper.
        """
        with self._lock:
            request_started = False
            try:
                event = _copy(event)
                if not self._last_event or not _same(event, self._last_event) or not event["review_required"]:
                    raise ValueError("event is foreign, superseded or requires no review")
                if event["gate_id"] in self._requested_gate_ids:
                    raise ValueError("gate was already requested; inspect a fresh proposal for a new gate")
                self._requested_gate_ids.add(event["gate_id"])
                self.queue.invalidate()
                snapshot = self.queue.snapshot()
                binding = {key: event[key] for key in ("case_id", "gate_id", "request_id")}
                binding.update({"queue_generation": snapshot.generation, "queue_cursor": snapshot.cursor})
                self._gate = {"status": "pending_local_hold", "reason": "local hold callback/receipt required",
                              "binding": binding, "hold_receipt": None}
                request_started = True
                now = self._now()
                observation = self._observation(obs, now)
                if self._last_observation is None or not _same(observation, self._last_observation):
                    raise ValueError("gate observation differs from inspected observation")
                binding.update({key: observation[key] for key in ("observation_id", "task_id", "attempt_id", "episode_id")})
                self._gate["observation"] = observation
                self._gate["requested_at_s"] = now
                if not callable(hold_callback):
                    return self._gate_result()
                request = {"operation": "hold_current_measured_joints_and_stop_base", **deepcopy(binding),
                           "held_joint_positions_rad": observation["state"][:6], "joint_units": "rad",
                           "requested_at_s": now, "caller_transport_required": True}
                result = hold_callback(request)
                later = self._now()
                self._observation(observation, later)
                receipt = self._hold_receipt(result, observation["observation_id"], now, later)
                self._queue_unchanged(self._gate)
                self._gate["hold_receipt"] = receipt
                if event["kind"] == "hold":
                    raise ValueError("invalid proposal/observation requires corrected fresh input before review")
                self._gate.update(status="review_ready", reason="fresh local hold caller attestation accepted")
                return self._gate_result()
            except Exception as exc:
                if request_started and self._gate and self._gate["status"] == "pending_local_hold":
                    self._gate["reason"] = str(exc)
                    return self._gate_result()
                return {"status": "rejected", "reason": str(exc), "review_ready": False,
                        "queue_committed": False, "physical_hold_verified_by_wrapper": False}

    def accept_strategy(self, validated_review, latest_observation):
        """Return a symbolic fresh-follower request, never timed actions.

        Review duration is not a trajectory timing budget. Old evidence remains
        identified as old; latest sensing, local hold and a real VLA/solver must
        be re-established outside this wrapper before any action submission.
        """
        with self._lock:
            try:
                gate = self._gate
                if not gate or gate["status"] != "review_ready":
                    raise ValueError("no ready gate; cancelled, superseded or duplicate strategy")
                review = _copy(validated_review)
                if set(review) != {"binding", "validation"} or not _same(review["binding"], gate["binding"]):
                    raise ValueError("strategy binding differs from active case/request/gate/attempt")
                latest = self._observation(latest_observation, self._now())
                original = gate["observation"]
                if "case_id" in latest and latest["case_id"] != gate["binding"]["case_id"]:
                    raise ValueError("latest observation changed case_id")
                for key in ("task_id", "attempt_id", "episode_id"):
                    if latest[key] != original[key]:
                        raise ValueError(f"latest observation changed {key}")
                if latest["observation_id"] == original["observation_id"] or latest["observed_at_s"] <= original["observed_at_s"]:
                    raise ValueError("a new observation ID and later fresh timestamp are required")
                self._queue_unchanged(gate)
                validation = review["validation"]
                arguments = validation["arguments"]
                if arguments["correction"]["joint_proposal"] is not None:
                    raise ValueError("only nonnumeric symbolic corrections may cross this handoff")
                verified = validate_review_reply(
                    {"name": "review_action_stream", "arguments": arguments},
                    case_id=gate["binding"]["case_id"], decision_time_s=original["observed_at_s"],
                    available_frames=original.get("available_frames", {}), physical_limits=None,
                    current_episode_id=original["episode_id"])
                if not _same(validation, verified):
                    raise ValueError("validation must exactly match revalidated causal reviewer contract")
                correction = verified["arguments"]["correction"]
                if any(re.search(r"\d", text) for text in [correction["strategy"]] + [step["description"] for step in correction["steps"]]):
                    raise ValueError("numeric literals are not allowed in symbolic correction text")
                follower_id = uuid4().hex
                gate.update(status="strategy_handed_off", follower_request_id=follower_id,
                            follower_observation=latest)
                return {"status": "follower_required", "follower_request_id": follower_id,
                        "binding": deepcopy(gate["binding"]), "instruction": correction["strategy"],
                        "symbolic_steps": deepcopy(correction["steps"]),
                        "latest_observation_id": latest["observation_id"],
                        "latest_observed_at_s": latest["observed_at_s"],
                        "based_on_review_observation_id": original["observation_id"],
                        "requires_fresh_vla": True, "requires_fresh_local_hold_before_execution": True,
                        "requires_existing_coordinator_action_validation": True,
                        "queue_committed": False, "model_calls": 0, "actuator_calls": 0,
                        "effective_decision": "hold", "task_outcome": "unknown",
                        "physical_hold_verified_by_wrapper": False}
            except (ValueError, TypeError, KeyError) as exc:
                return {"status": "rejected", "reason": str(exc), "queue_committed": False,
                        "effective_decision": "hold", "model_calls": 0, "actuator_calls": 0}

    def cancel(self, gate_id, reason="caller_cancelled"):
        with self._lock:
            if not self._gate or self._gate["binding"]["gate_id"] != gate_id:
                raise ValueError("cannot cancel a foreign gate")
            _text(reason, "cancel reason")
            self.queue.invalidate()
            self._gate.update(status="cancelled", reason=reason)
            return self._gate_result()

    def mark_post_grasp(self, latest_observation, measured_receipt):
        """Require caller-attested measured closure before post-grasp review.

        A submitted close is insufficient. This does not declare object contact
        or success, infer a lift from q6, or permit any lifting action.
        """
        with self._lock:
            try:
                gate, now = self._gate, self._now()
                if not gate or gate["status"] != "strategy_handed_off" or gate.get("post_grasp_marked"):
                    raise ValueError("no unconsumed follower handoff for measured closure")
                if self.queue.snapshot().generation != gate["binding"]["queue_generation"]:
                    raise ValueError("native queue generation changed after follower handoff")
                obs, receipt = self._observation(latest_observation, now), _copy(measured_receipt)
                if (receipt.get("source") != "measured_controller_feedback" or receipt.get("confirmed") is not True
                        or receipt.get("closure_observed") is not True
                        or _number(receipt.get("gripper_command"), "gripper_command") != self.gripper_closed):
                    raise ValueError("measured closure caller attestation is required")
                for key in ("observation_id", "task_id", "attempt_id", "episode_id"):
                    if receipt.get(key) != obs[key]:
                        raise ValueError("closure receipt has foreign observation/task/attempt")
                    if key != "observation_id" and obs[key] != gate["binding"][key]:
                        raise ValueError("closure belongs to another task/attempt/episode")
                if receipt.get("follower_request_id") != gate["follower_request_id"]:
                    raise ValueError("closure receipt belongs to another follower")
                checked = _number(receipt.get("checked_at_s"), "closure checked_at_s")
                if checked < obs["observed_at_s"] or not 0 <= now - checked <= self.max_hold_receipt_age_s:
                    raise ValueError("closure receipt is stale or in the future")
                if (obs["observation_id"] == gate["follower_observation"]["observation_id"]
                        or obs["observed_at_s"] <= gate["follower_observation"]["observed_at_s"]):
                    raise ValueError("measured closure needs a newer observation than follower input")
                gate["post_grasp_marked"] = True
                return self._event("post_grasp_verify", "caller_attested_measured_closure", obs,
                                   lift_requires_review=True, grasp_success="unknown",
                                   physical_feedback_verified_by_wrapper=False)
            except (ValueError, TypeError, KeyError) as exc:
                return {"status": "rejected", "reason": str(exc), "queue_committed": False,
                        "grasp_success": "unknown", "physical_feedback_verified_by_wrapper": False}
