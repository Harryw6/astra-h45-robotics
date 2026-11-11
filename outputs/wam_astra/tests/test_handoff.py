"""Synthetic caller attestations exercise the real pinned native action queue."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "work/wam_demo_library/vendor"))


def api():
    assert importlib.util.find_spec("wam_astra.handoff") is not None, "handoff supervisor is missing"
    return importlib.import_module("wam_astra.handoff")


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def observation(clock, identifier="obs-1", **changes):
    result = {"observation_id": identifier, "observed_at_s": clock(),
              "task_id": "task-1", "attempt_id": "attempt-1", "episode_id": "episode-1",
              "state": [.1] * 6 + [.03], "state_joint_units": "rad", "state_gripper_units": "m",
              "state_verified": True, "source": "measured_controller_feedback",
              "available_frames": {"synthetic-front": {"camera": "front", "time_s": clock() - .1}},
              "fixture_type": "synthetic_test_only"}
    result.update(changes)
    return result


def actions(*commands):
    return [[.1] * 6 + [command] for command in commands]


@pytest.fixture
def environment():
    from wam_astra.rtc import ActionQueue
    queue, clock = ActionQueue(), Clock()
    return queue, clock


def supervisor(environment, **changes):
    queue, clock = environment
    config = dict(gripper_open=1., gripper_closed=0., clock=clock,
                  max_observation_age_s=.5, max_hold_receipt_age_s=.2)
    config.update(changes)
    return api().HandoffSupervisor(queue, **config)


def synthetic_hold_callback(clock, **changes):
    """Explicit synthetic test callback, not a transport or hardware proof."""
    def callback(request):
        result = {"observation_id": request["observation_id"],
                  "held_joint_positions_rad": [.1] * 6, "joint_units": "rad",
                  "base_velocity_zero": True, "confirmed": True,
                  "source": "measured_controller_feedback", "checked_at_s": clock(),
                  "fixture_type": "synthetic_test_only"}
        result.update(changes)
        return result
    return callback


def ready_gate(environment):
    queue, clock = environment
    sup = supervisor(environment)
    obs = observation(clock)
    event = sup.inspect_proposal(actions(1., 0.), obs, 1.)
    ready = sup.request_gate(event, obs, synthetic_hold_callback(clock))
    assert ready["review_ready"]
    return sup, obs, ready


def validated_envelope(ready, obs):
    from wam_astra.review_contract import validate_review_reply
    binding = ready["binding"]
    reply = {"name": "review_action_stream", "arguments": {
        "schema_version": 1, "case_id": binding["case_id"], "decision_time_s": obs["observed_at_s"],
        "assessment": {"status": "uncertain", "confidence": .3,
                       "evidence": [], "limitations": ["仅为合成测试，无真实抓取证据。"]},
        "correction": {"strategy": "先保持并补充观察，再由本地策略重新对齐。",
                       "steps": [{"kind": "observe", "description": "补充腕部画面确认目标位置。",
                                  "evidence_frame_ids": ["synthetic-front"]}],
                       "joint_proposal": None, "not_executable_reason": "需要新鲜本地观测和已验证的跟随策略。"}}}
    validation = validate_review_reply(reply, case_id=binding["case_id"], decision_time_s=obs["observed_at_s"],
                                       available_frames=obs["available_frames"])
    return {"binding": deepcopy(binding), "validation": validation}


def latest_observation(clock, **changes):
    clock.now += .1
    return observation(clock, "obs-latest", **changes)


def test_pre_grasp_detects_future_close_before_any_commit(environment):
    queue, clock = environment
    sup = supervisor(environment)
    snapshot = queue.snapshot()
    event = sup.inspect_proposal(actions(1., 1., 0., 0.), observation(clock), 1.)
    assert event["kind"] == "pre_grasp"
    assert event["closing_action_index"] == 2
    assert event["review_required"]
    assert event["queue_committed"] is False
    assert event["executed_actions"] == 0
    assert queue.empty()
    assert queue.snapshot().generation == snapshot.generation


def test_transition_uses_configured_polarity_and_internal_reopen(environment):
    _, clock = environment
    sup = supervisor(environment, gripper_open=0., gripper_closed=1.)
    event = sup.inspect_proposal(actions(1., 0., 1.), observation(clock), 1.)
    assert event["kind"] == "pre_grasp"
    assert event["closing_action_index"] == 2


def test_no_transition_does_not_invent_grasp_event(environment):
    _, clock = environment
    event = supervisor(environment).inspect_proposal(actions(1., 1.), observation(clock), 1.)
    assert event["kind"] == "no_gate"
    assert event["review_required"] is False
    assert event["physical_actions_validated"] is False


@pytest.mark.parametrize("changes", [
    {"state_verified": False}, {"source": "recorded_csv"}, {"state_joint_units": "degree"},
    {"state_gripper_units": "normalized"}, {"state": [float("nan")] * 7},
    {"state": [0.] * 6}, {"observed_at_s": 99.}, {"observed_at_s": 101.},
    {"attempt_id": ""}, {"extra": float("inf")},
])
def test_invalid_or_unknown_observation_holds_with_finite_json(environment, changes):
    _, clock = environment
    event = supervisor(environment).inspect_proposal(actions(0.), observation(clock, **changes), 1.)
    assert event["kind"] == "hold"
    assert event["review_required"]
    assert event["queue_committed"] is False
    json.dumps(event, allow_nan=False)


@pytest.mark.parametrize("proposal,previous", [([[0.] * 6], 1.), (actions(float("inf")), 1.),
                                             (actions(.3), 1.), (actions(0.), None), (actions(0.), .03)])
def test_invalid_actions_and_unknown_previous_command_hold(environment, proposal, previous):
    _, clock = environment
    result = supervisor(environment).inspect_proposal(proposal, observation(clock), previous)
    assert result["kind"] == "hold"


def test_gate_invalidates_native_future_and_preempts_old_queue_snapshot(environment):
    queue, clock = environment
    assert queue.merge(actions(1., 0.), actions(1., 0.), 0, snapshot=queue.snapshot())
    old = queue.snapshot()
    sup = supervisor(environment)
    obs = observation(clock)
    event = sup.inspect_proposal(actions(1., 0.), obs, 1.)
    result = sup.request_gate(event, obs)
    assert queue.empty()
    assert queue.snapshot().generation == old.generation + 1
    assert not queue.merge(actions(0.), actions(0.), 0, snapshot=old)
    assert result["status"] == "pending_local_hold"
    assert result["review_ready"] is False
    assert result["physical_hold_verified_by_wrapper"] is False


@pytest.mark.parametrize("changes", [
    {"confirmed": False}, {"base_velocity_zero": False}, {"source": "command_submitted"},
    {"observation_id": "foreign"}, {"held_joint_positions_rad": [float("nan")] * 6},
    {"checked_at_s": 99.}, {"checked_at_s": 101.}, {"joint_units": "degree"},
])
def test_missing_or_invalid_local_hold_never_becomes_review_ready(environment, changes):
    _, clock = environment
    sup = supervisor(environment)
    obs = observation(clock)
    event = sup.inspect_proposal(actions(0.), obs, 1.)
    result = sup.request_gate(event, obs, synthetic_hold_callback(clock, **changes))
    assert result["status"] == "pending_local_hold"
    assert result["review_ready"] is False


def test_confirmed_hold_is_caller_attestation_not_physical_proof(environment):
    _, _, ready = ready_gate(environment)
    assert ready["status"] == "review_ready"
    assert ready["local_hold_attestation_accepted"] is True
    assert ready["physical_hold_verified_by_wrapper"] is False
    assert ready["model_calls"] == ready["actuator_calls"] == 0


def test_153_second_strategy_requires_fresh_follower_and_never_retimes_queue(environment):
    queue, clock = environment
    sup, obs, ready = ready_gate(environment)
    review = validated_envelope(ready, obs)
    clock.now += 153.6
    latest = latest_observation(clock)
    result = sup.accept_strategy(review, latest)
    assert result["status"] == "follower_required"
    assert result["requires_fresh_vla"] is True
    assert result["requires_fresh_local_hold_before_execution"] is True
    assert result["latest_observation_id"] == latest["observation_id"]
    assert result["based_on_review_observation_id"] == obs["observation_id"]
    assert "actions" not in result
    assert "target_start_at_s" not in result
    assert queue.empty()
    assert result["queue_committed"] is False


@pytest.mark.parametrize("changes", [{"task_id": "other"}, {"attempt_id": "other"}, {"episode_id": "other"},
                                     {"observed_at_s": 99.}, {"state_verified": False}])
def test_strategy_rejects_stale_or_foreign_latest_state(environment, changes):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    result = sup.accept_strategy(validated_envelope(ready, obs), latest_observation(clock, **changes))
    assert result["status"] == "rejected"


def test_strategy_requires_new_observation_id_and_time(environment):
    sup, obs, ready = ready_gate(environment)
    result = sup.accept_strategy(validated_envelope(ready, obs), obs)
    assert result["status"] == "rejected"


def test_latest_explicit_case_change_rejects_strategy(environment):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    latest = latest_observation(clock, case_id="foreign-case")
    assert sup.accept_strategy(validated_envelope(ready, obs), latest)["status"] == "rejected"


def test_duplicate_strategy_acceptance_cannot_issue_another_follower(environment):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    review = validated_envelope(ready, obs)
    assert sup.accept_strategy(review, latest_observation(clock))["status"] == "follower_required"
    assert sup.accept_strategy(review, latest_observation(clock, observation_id="later"))["status"] == "rejected"


def test_hold_callback_returning_after_observation_expires_cannot_open_review(environment):
    _, clock = environment
    sup, obs = supervisor(environment), observation(clock)
    event = sup.inspect_proposal(actions(0.), obs, 1.)
    def delayed_synthetic_callback(request):
        clock.now += .6
        return synthetic_hold_callback(clock)(request)
    result = sup.request_gate(event, obs, delayed_synthetic_callback)
    assert result["status"] == "pending_local_hold"
    assert result["review_ready"] is False


def test_mutated_gate_cannot_invalidate_or_replace_a_ready_request(environment):
    queue, clock = environment
    sup, obs, ready = ready_gate(environment)
    generation = queue.snapshot().generation
    forged = sup.inspect_proposal(actions(0.), obs, 1.)
    forged["closing_action_index"] = 999
    assert sup.request_gate(forged, obs)["status"] == "rejected"
    assert queue.snapshot().generation == generation


@pytest.mark.parametrize("phase", ["pending_local_hold", "review_ready", "strategy_handed_off"])
def test_same_requested_event_cannot_reset_lifecycle_or_native_generation(environment, phase):
    queue, clock = environment
    sup, obs = supervisor(environment), observation(clock)
    event = sup.inspect_proposal(actions(0.), obs, 1.)
    callback = None if phase == "pending_local_hold" else synthetic_hold_callback(clock)
    first = sup.request_gate(event, obs, callback)
    review = None
    if phase == "strategy_handed_off":
        review = validated_envelope(first, obs)
        assert sup.accept_strategy(review, latest_observation(clock))["status"] == "follower_required"
    before = queue.snapshot()
    replay = sup.request_gate(event, obs, synthetic_hold_callback(clock))
    assert replay["status"] == "rejected"
    assert queue.snapshot().generation == before.generation
    assert queue.snapshot().cursor == before.cursor
    assert queue.empty()
    if phase == "review_ready":
        assert sup.accept_strategy(validated_envelope(first, obs), latest_observation(clock))["status"] == "follower_required"
    elif phase == "strategy_handed_off":
        assert sup.accept_strategy(review, latest_observation(clock, observation_id="newer"))["status"] == "rejected"
    else:
        newer = latest_observation(clock)
        new_event = sup.inspect_proposal(actions(0.), newer, 1.)
        assert new_event["gate_id"] != event["gate_id"]
        assert sup.request_gate(new_event, newer, synthetic_hold_callback(clock))["review_ready"]


@pytest.mark.parametrize("field", ["gate_id", "request_id", "case_id", "observation_id", "attempt_id"])
def test_strategy_rejects_foreign_request_binding(environment, field):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    review = validated_envelope(ready, obs)
    review["binding"][field] = "foreign"
    assert sup.accept_strategy(review, latest_observation(clock))["status"] == "rejected"


@pytest.mark.parametrize("mutation", [
    lambda review: review.update(validation={"executable": False}),
    lambda review: review["validation"].update(case_id="foreign"),
    lambda review: review["validation"].update(decision_time_s=1.),
    lambda review: review["validation"]["arguments"].update(case_id="foreign"),
    lambda review: review["validation"]["arguments"]["correction"].update(
        joint_proposal={"joint_units": "rad", "gripper_units": "normalized", "step_s": .1, "actions": actions(0.)}),
    lambda review: review["validation"]["arguments"]["correction"].update(strategy="前移0.2米"),
])
def test_strategy_revalidates_actual_contract_and_rejects_numeric_plans(environment, mutation):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    review = validated_envelope(ready, obs)
    mutation(review)
    assert sup.accept_strategy(review, latest_observation(clock))["status"] == "rejected"


def test_cancel_and_duplicate_review_are_rejected(environment):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    review = validated_envelope(ready, obs)
    sup.cancel(ready["binding"]["gate_id"], "synthetic user cancellation")
    assert sup.accept_strategy(review, latest_observation(clock))["status"] == "rejected"


@pytest.mark.parametrize("mutation", ["invalidate", "enqueue"])
def test_external_native_queue_changes_reject_strategy(environment, mutation):
    queue, clock = environment
    sup, obs, ready = ready_gate(environment)
    if mutation == "invalidate":
        queue.invalidate()
    else:
        assert queue.merge(actions(1.), actions(1.), 0, snapshot=queue.snapshot())
    assert sup.accept_strategy(validated_envelope(ready, obs), latest_observation(clock))["status"] == "rejected"


def test_superseded_gate_rejects_previous_strategy(environment):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    old_review = validated_envelope(ready, obs)
    newer = latest_observation(clock)
    sup.request_gate(sup.inspect_proposal(actions(0.), newer, 1.), newer, synthetic_hold_callback(clock))
    assert sup.accept_strategy(old_review, latest_observation(clock, observation_id="newest"))["status"] == "rejected"


def test_post_grasp_requires_measured_closure_attestation_before_lift(environment):
    _, clock = environment
    sup, obs, ready = ready_gate(environment)
    latest = latest_observation(clock)
    follower = sup.accept_strategy(validated_envelope(ready, obs), latest)
    clock.now += .1
    closed = observation(clock, "closure-observation")
    receipt = {"source": "measured_controller_feedback", "confirmed": True,
               "closure_observed": True, "gripper_command": 0., "checked_at_s": clock(),
               "observation_id": closed["observation_id"], "task_id": closed["task_id"],
               "attempt_id": closed["attempt_id"], "episode_id": closed["episode_id"],
               "follower_request_id": follower["follower_request_id"], "fixture_type": "synthetic_test_only"}
    rejected = sup.mark_post_grasp(closed, {**receipt, "source": "command_submitted"})
    assert rejected["status"] == "rejected"
    event = sup.mark_post_grasp(closed, receipt)
    assert event["kind"] == "post_grasp_verify"
    assert event["lift_requires_review"] is True
    assert event["grasp_success"] == "unknown"
    assert event["physical_feedback_verified_by_wrapper"] is False


@pytest.mark.parametrize("config", [{"gripper_open": 0., "gripper_closed": 0.},
                                   {"gripper_open": -7., "gripper_closed": 99.},
                                   {"gripper_open": .2, "gripper_closed": .8},
                                   {"gripper_open": float("nan")}, {"max_observation_age_s": 0.},
                                   {"max_hold_receipt_age_s": -1.}])
def test_configuration_requires_distinct_explicit_encoding_and_freshness(environment, config):
    with pytest.raises(ValueError):
        supervisor(environment, **config)


@pytest.mark.parametrize("external_preemption", [False, True])
def test_post_grasp_allows_native_follower_consumption_but_rejects_new_generation(environment, external_preemption):
    queue, clock = environment
    sup, obs, ready = ready_gate(environment)
    follower = sup.accept_strategy(validated_envelope(ready, obs), latest_observation(clock))
    assert follower["status"] == "follower_required"
    assert queue.merge(actions(1., 0.), actions(1., 0.), 0, snapshot=queue.snapshot())
    assert queue.get() is not None
    assert queue.get() is not None
    assert queue.empty()
    assert queue.snapshot().cursor != ready["binding"]["queue_cursor"]
    if external_preemption:
        queue.invalidate()
    clock.now += .1
    closed = observation(clock, "closure-observation")
    receipt = {"source": "measured_controller_feedback", "confirmed": True,
               "closure_observed": True, "gripper_command": 0., "checked_at_s": clock(),
               "observation_id": closed["observation_id"], "task_id": closed["task_id"],
               "attempt_id": closed["attempt_id"], "episode_id": closed["episode_id"],
               "follower_request_id": follower["follower_request_id"], "fixture_type": "synthetic_test_only"}
    result = sup.mark_post_grasp(closed, receipt)
    if external_preemption:
        assert result["status"] == "rejected"
        assert "generation" in result["reason"]
    else:
        assert result["kind"] == "post_grasp_verify"
