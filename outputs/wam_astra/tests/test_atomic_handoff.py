"""Structured demonstration comparisons; fixtures are not physical evidence."""
from copy import deepcopy

import jsonschema
import numpy as np
import pytest

from wam_astra.atomic_motion import HybridPolicy


def supported_evidence():
    return {
        "target_visible": {"status": "supported", "evidence": "Current wrist view shows the cube inside the box.",
                           "reference_ids": []},
        "base_placement_supported": {"status": "supported", "evidence": "Current robot/table arrangement resembles the supplied pre-grasp demonstration.",
                                     "reference_ids": ["synthetic-demo-pregrasp"]},
        "arm_start_supported": {"status": "supported", "evidence": "Measured arm start and wrist view are consistent with the demonstration start.",
                                "reference_ids": ["synthetic-demo-pregrasp"]},
    }


@pytest.mark.parametrize("evidence", [None, {}, {"target_visible": {"status": "supported"}}])
def test_missing_handoff_evidence_observes_without_changing_phase(evidence):
    action = {"decision": "handoff_pi", "handoff_evidence": evidence}
    policy = HybridPolicy(lambda _: action, None)
    assert policy.infer_prepared(policy.prepare_input({"state": np.zeros(14)})) is None
    assert policy.phase == "positioning"
    assert policy.last_handoff_assessment["status"] == "observe"
    assert not policy.last_handoff_assessment["ready"]


@pytest.mark.parametrize("field", ["target_visible", "base_placement_supported", "arm_start_supported"])
@pytest.mark.parametrize("status", ["unknown", "unsupported"])
def test_incomplete_comparison_observes_without_pi_calls(field, status):
    evidence = supported_evidence()
    evidence[field]["status"] = status
    policy = HybridPolicy(lambda _: {"decision": "handoff_pi", "handoff_evidence": evidence}, None)
    assert policy.infer_prepared(policy.prepare_input({"state": np.zeros(14)})) is None
    assert policy.phase == "positioning"
    assert field in " ".join(policy.last_handoff_assessment["reasons"])


def test_supported_comparisons_require_fresh_pi_observation_and_preserve_h45():
    class Pi:
        def __init__(self):
            self.observations = []
            self.chunk = np.arange(45 * 14, dtype=float).reshape(45, 14)

        def prepare_input(self, observation, rtc=None):
            self.observations.append(observation)
            return observation

        def infer_prepared(self, prepared, **kwargs):
            return self.chunk

    pi = Pi()
    policy = HybridPolicy(lambda _: {"decision": "handoff_pi", "handoff_evidence": supported_evidence()}, pi)
    before = {"state": np.zeros(14)}
    assert policy.infer_prepared(policy.prepare_input(before)) is None
    assert policy.phase == "grasp"
    assert pi.observations == []
    after = {"state": np.ones(14)}
    chunk = policy.infer_prepared(policy.prepare_input(after))
    assert pi.observations[0]['state'] is after['state']
    np.testing.assert_array_equal(chunk, pi.chunk)
    assert chunk is not pi.chunk
    assert policy.last_action_source == "pi_original_h45"


def test_handoff_schema_requires_every_check_and_accepts_unknown():
    from wam_astra.atomic_handoff import handoff_evidence_schema
    evidence = supported_evidence()
    evidence["arm_start_supported"] = {"status": "unknown", "evidence": "No demonstrated arm start is supplied.",
                                       "reference_ids": []}
    jsonschema.validate(evidence, handoff_evidence_schema())
    del evidence["base_placement_supported"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(evidence, handoff_evidence_schema())


@pytest.mark.parametrize("field", ["target_visible", "base_placement_supported", "arm_start_supported"])
def test_supported_checks_need_explanation(field):
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    evidence = supported_evidence()
    evidence[field]["evidence"] = "  "
    result = evaluate_handoff_evidence(evidence)
    assert not result["ready"]
    assert field in " ".join(result["reasons"])


@pytest.mark.parametrize("field", ["base_placement_supported", "arm_start_supported"])
@pytest.mark.parametrize("references", [[], [" "], "invented-reference", [1]])
def test_demonstration_comparisons_need_reference_identifiers(field, references):
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    evidence = supported_evidence()
    evidence[field]["reference_ids"] = references
    assert not evaluate_handoff_evidence(evidence)["ready"]


@pytest.mark.parametrize("malformed", [True, [], "supported", {"unexpected": 1}])
def test_malformed_evidence_abstains_instead_of_raising(malformed):
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    result = evaluate_handoff_evidence(malformed)
    assert result["status"] == "observe"
    assert not result["ready"]


def test_supported_result_labels_scope_and_does_not_modify_input():
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    evidence = supported_evidence()
    before = deepcopy(evidence)
    result = evaluate_handoff_evidence(evidence)
    assert result["ready"]
    assert result["status"] == "handoff_pi"
    assert result["scope"] == "structured_demonstration_support_only"
    assert result["physical_reachability_verified"] is False
    assert evidence == before


@pytest.mark.parametrize("field", ["base_placement_supported", "arm_start_supported"])
def test_fabricated_demonstration_id_abstains(field):
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    evidence = supported_evidence()
    evidence[field]["reference_ids"] = ["fabricated-demo"]
    result = evaluate_handoff_evidence(evidence, allowed_reference_ids={"synthetic-demo-pregrasp"})
    assert not result["ready"]
    assert "fabricated-demo" in " ".join(result["reasons"])


def test_empty_demonstration_registry_cannot_support_handoff():
    from wam_astra.atomic_handoff import evaluate_handoff_evidence
    assert not evaluate_handoff_evidence(supported_evidence(), allowed_reference_ids=set())["ready"]


def test_policy_checks_reference_ids_against_planner_registry():
    class Planner:
        reference_ids = {"different-supplied-demo"}

        def decide(self, observation, check_wait=None):
            return {"decision": "handoff_pi", "handoff_evidence": supported_evidence()}

    policy = HybridPolicy(Planner(), None)
    assert policy.infer_prepared(policy.prepare_input({"state": np.zeros(14)})) is None
    assert policy.phase == "positioning"
    assert not policy.last_handoff_assessment["ready"]


def test_policy_returns_denial_reason_to_planner_before_next_turn():
    class Planner:
        def decide(self, observation, check_wait=None):
            return {"decision": "handoff_pi"}

        def record_handoff_assessment(self, assessment):
            self.assessment = assessment

    planner = Planner()
    policy = HybridPolicy(planner, None)
    assert policy.infer_prepared(policy.prepare_input({"state": np.zeros(14)})) is None
    assert planner.assessment is policy.last_handoff_assessment
    assert planner.assessment["status"] == "observe"
    assert planner.assessment["reasons"]


def test_policy_forwards_execution_receipts_to_planner():
    class Planner:
        last_execution_feedback = {"status": "test_feedback"}

        def record_execution(self, **receipt):
            self.receipt = receipt

    planner = Planner()
    policy = HybridPolicy(planner, None)
    policy.record_execution(packet={"observation_id": "measured-before"}, committed=True)
    assert planner.receipt == {"packet": {"observation_id": "measured-before"}, "committed": True}
    assert policy.last_execution_feedback is planner.last_execution_feedback


def test_policy_accepts_receipts_when_simple_planner_has_no_feedback_hook():
    policy = HybridPolicy(lambda _: {"decision": "observe"}, None)
    assert policy.record_execution(committed=False) is None
    assert policy.last_execution_feedback is None
