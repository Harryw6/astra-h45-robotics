"""Demonstration-supported Pi handoff contract; no robot I/O or IK solver.

This checks the planner's evidence structure and supplied reference identifiers.
It cannot establish that a model's visual comparison is true or prove physical
reachability, collision clearance, or grasp success.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator


HANDOFF_CHECKS = ("target_visible", "base_placement_supported", "arm_start_supported")

HANDOFF_EVIDENCE_PROMPT = """Every action includes handoff_evidence with target_visible, base_placement_supported and arm_start_supported checks. Each check has status supported/unsupported/unknown, concise evidence and reference_ids.
Seeing the target alone does not support handoff_pi. Compare the CURRENT robot/table arrangement and measured arm starting pose plus wrist view with supplied demonstrated pre-grasp states. Explain each comparison; cite actual supplied demonstration IDs for base_placement_supported and arm_start_supported. target_visible may use an empty reference_ids list because it is a current-image observation.
Use only supplied_reference_ids for demonstration citations. Missing demonstrated base/table relation or arm-start evidence means unknown, not supported. For actions that do not propose handoff, use unknown for unassessed checks. Never invent a distance threshold, joint tolerance, camera calibration, transform, IK result or reachability proof. These are demonstration-support judgments, not physical reachability verification. If any check is unsupported or unknown, continue positioning or observe; do not hand off. Pi retains its original H45 behavior after a supported handoff and receives a new measured observation."""


def handoff_evidence_schema():
    """Strict JSON shape used by the planner, including explicit abstention."""
    check = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string", "enum": ["supported", "unsupported", "unknown"]},
            "evidence": {"type": "string"},
            "reference_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "required": ["status", "evidence", "reference_ids"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: deepcopy(check) for name in HANDOFF_CHECKS},
        "required": list(HANDOFF_CHECKS),
    }


def evaluate_handoff_evidence(evidence, *, allowed_reference_ids=None):
    """Return ordinary observe status unless every comparison is supported.

If supplied, allowed_reference_ids is the caller's registry of demonstrations
actually presented to the planner. An empty registry cannot support handoff.
Omitting the registry checks citation structure only (for standalone callers),
and is explicitly reported in the result. Neither mode validates prose truth.
"""
    result = {
        "ready": False,
        "status": "observe",
        "checks": {},
        "reasons": [],
        "scope": "structured_demonstration_support_only",
        "reference_registry_checked": allowed_reference_ids is not None,
        "physical_reachability_verified": False,
    }
    errors = list(Draft202012Validator(handoff_evidence_schema()).iter_errors(evidence))
    if errors:
        for error in errors:
            path = ".".join(["handoff_evidence", *(str(part) for part in error.absolute_path)])
            result["reasons"].append(f"{path}: invalid or missing evidence structure")
        return result

    result["checks"] = deepcopy(evidence)
    references = set(allowed_reference_ids) if allowed_reference_ids is not None else None
    for name in HANDOFF_CHECKS:
        check = evidence[name]
        if check["status"] != "supported":
            result["reasons"].append(f"{name}: {check['status']}")
        elif not check["evidence"].strip():
            result["reasons"].append(f"{name}: missing supporting observation or comparison")
        if any(not reference.strip() for reference in check["reference_ids"]):
            result["reasons"].append(f"{name}: blank reference identifier")
        if name != "target_visible":
            if not check["reference_ids"]:
                result["reasons"].append(f"{name}: missing demonstration reference_ids")
            elif references is not None:
                unknown = sorted(set(check["reference_ids"]) - references)
                if unknown:
                    result["reasons"].append(f"{name}: references not supplied: {', '.join(unknown)}")
    result["ready"] = not result["reasons"]
    if result["ready"]:
        result["status"] = "handoff_pi"
    return result
