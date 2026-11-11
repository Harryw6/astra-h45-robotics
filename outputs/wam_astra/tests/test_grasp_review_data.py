"""Causal data preparation must use verified source time, never video origin guesses."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    path = Path(__file__).resolve().parents[1] / "examples/prepare_grasp_review.py"
    assert path.is_file(), "grasp review data preparation is missing"
    spec = importlib.util.spec_from_file_location("grasp_review_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partition_uses_actual_capture_ns_and_never_requested_offset():
    module = api()
    assert module.temporal_role(99, 100, alignment_verified=True) == "causal_before_decision"
    assert module.temporal_role(100, 100, alignment_verified=True) == "causal_before_decision"
    assert module.temporal_role(101, 100, alignment_verified=True) == "retrospective_after"


def test_three_view_partition_requires_every_component_before_decision_with_clock_bound():
    module = api()
    assert module.composite_temporal_role([9.97, 9.98, 9.99], 10., .004) == "causal_before_decision"
    assert module.composite_temporal_role([10.1, 10.2, 10.3], 10., .004) == "retrospective_after"
    assert module.composite_temporal_role([9.9, 10.001, 9.9], 10., .004) == "alignment_unknown"
    assert module.composite_temporal_role([9.999, 9.9, 9.9], 10., .004) == "alignment_unknown"


def test_unknown_alignment_never_fabricates_host_time_or_causal_claim():
    result = api().verify_event_anchor(None, None, 244.2)
    assert result["status"] == "unavailable"
    assert result["decision_monotonic_ns"] is None
    assert api().temporal_role(None, None, alignment_verified=False) == "alignment_unknown"


@pytest.fixture
def aligned(tmp_path):
    columns = ["source_row", "relative_time_s", "monotonic_ns", *[f"joint_{i}.pos" for i in range(1, 7)], "gripper.pos"]
    row = [7, 244.25, 9282000000000, 0, .1, .2, .3, .4, .5, .02485]
    csv = tmp_path / "state.csv"
    csv.write_text(",".join(columns) + "\n" + ",".join(map(str, row)) + "\n")
    npz = tmp_path / "signals.npz"
    np.savez(npz, state_source_row=[7], state_t_ns=[9282000000000],
             state_q=[[0, .1, .2, .3, .4, .5]], state_g=[.02485])
    return csv, npz


def test_event_anchor_requires_matching_aligned_source_row_and_npz(aligned):
    result = api().verify_event_anchor(*aligned, 244.25)
    assert result["status"] == "verified"
    assert result["source_row"] == 7
    assert result["decision_monotonic_ns"] == 9282000000000
    assert result["gripper_measured_m"] == .02485
    assert result["joint_measured_rad"] == [0, .1, .2, .3, .4, .5]


def test_distant_video_time_does_not_get_a_guessed_state_anchor(aligned):
    result = api().verify_event_anchor(*aligned, 244.3)
    assert result["status"] == "unavailable"
    assert result["decision_monotonic_ns"] is None


def test_disagreeing_npz_and_alignment_is_unavailable(aligned):
    csv, npz = aligned
    np.savez(npz, state_source_row=[7], state_t_ns=[1], state_q=[[0]*6], state_g=[.02485])
    result = api().verify_event_anchor(csv, npz, 244.25)
    assert result["status"] == "unavailable"
    assert result["reason"] == "aligned_state_and_npz_disagree"


def test_manifest_causal_subset_cannot_include_retrospective_frames(tmp_path):
    frames = [{"frame_id": "before", "temporal_role": "causal_before_decision", "camera": "front",
               "video_relative_time_s": 1., "path": str(tmp_path / "before.png")},
              {"frame_id": "after", "temporal_role": "retrospective_after", "camera": "wrist",
               "video_relative_time_s": 2., "path": str(tmp_path / "after.png")}]
    result = api().make_input_manifest(frames, "causal_before_decision", "抓取盒内方块")
    images = [item for item in result["content"] if isinstance(item, dict)]
    assert [item["label"].split()[0] for item in images] == ["before"]
    assert "after.png" not in json.dumps(result)


def test_manifest_rejects_over_24_images():
    frames = [{"frame_id": str(i), "temporal_role": "causal_before_decision", "camera": "front",
               "video_relative_time_s": float(i), "path": f"{i}.png"} for i in range(25)]
    with pytest.raises(ValueError, match="24"):
        api().make_input_manifest(frames, "causal_before_decision", "Inspect")


def test_next_second_archive_preserves_only_recorded_sent_targets_with_unknown_units(tmp_path):
    path = tmp_path / "signals.npz"
    np.savez(path, command_t_ns=[99, 100, 500000100, 1000000100],
             command_q=[[1]*6, [2]*6, [3]*6, [4]*6], command_g=[.1, .2, .3, .4],
             command_source_row=[7, 8, 9, 10], command_command_sent=[1, 1, 1, 1],
             state_q=[[99]*6])
    result = api().recorded_command_archive(path, 100)
    assert result["status"] == "available"
    assert [row["source_row"] for row in result["recorded_teleoperation"]["samples"]] == [8, 9]
    assert result["recorded_teleoperation"]["samples"][0]["piper"]["recorded_joint_target"] == [2]*6
    assert result["vla_generated"] is False
    assert result["target_units_verified"] is False
    assert "state_q" not in json.dumps(result)


def test_missing_command_archive_is_explicitly_unavailable(tmp_path):
    path = tmp_path / "signals.npz"
    np.savez(path, state_q=[[0]*6])
    assert api().recorded_command_archive(path, 100)["status"] == "unavailable"
