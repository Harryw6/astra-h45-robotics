"""Real SQLite and scikit-learn retrieval over a small provenance fixture."""
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("wam_astra.retrieval") is not None, "retrieval adapter is missing"
    return importlib.import_module("wam_astra.retrieval")


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    episodes = [
        {"episode_id": "episode-a", "date": "2026-01-01", "quality": "poor"},
        {"episode_id": "episode-b", "date": "2026-01-02"},
        {"episode_id": "auxiliary-only", "date": "2026-01-03"},
    ]
    blocks = []
    specs = [("tie-b", "episode-a", 0., [1., 0., 0., 0., 0., 0.], ["gripper_transition"]),
             ("tie-a", "episode-a", 2., [-1., 0., 0., 0., 0., 0.], []),
             ("exact", "episode-b", 1., [0.] * 6, ["gripper_transition"]),
             ("missing", "episode-b", 4., None, ["unknown_state"]),
             ("invalid", "episode-b", 6., [0.] * 5, [])]
    for block_id, episode_id, start, q, events in specs:
        date = next(e["date"] for e in episodes if e["episode_id"] == episode_id)
        record_path = "records/" + block_id + ".json"
        path = root / record_path
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"O0": {"q": q}, "A": [], "receipt": {"status": "unknown"}, "O1": {}}))
        blocks.append({"block_id": block_id, "episode_id": episode_id, "date": date,
                       "start_ns": int(start * 1.e9), "end_ns": int((start + 1.) * 1.e9),
                       "start_s": start, "end_s": start + 1., "kind": "fixed",
                       "q_start": q, "gripper_start_m": None, "events": events,
                       "record_path": record_path, "quality_issues": ["not_a_filter"]})
    (root / "manifest.json").write_text(json.dumps({"episodes": episodes, "block_count": len(blocks)}))
    (root / "blocks.jsonl").write_text("\n".join(json.dumps(b) for b in blocks) + "\n")
    return root


def test_index_retains_every_episode_and_block_without_quality_gate(library):
    result = api().build_index(library)
    assert result["episode_count"] == 3
    assert result["block_count"] == 5
    assert result["similarity_eligible_count"] == 3
    assert result["similarity_excluded_count"] == 2
    with sqlite3.connect(result["index_path"]) as db:
        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 5
        assert db.execute("SELECT COUNT(*) FROM episodes WHERE episode_id='auxiliary-only'").fetchone()[0] == 1


def test_joint_similarity_uses_all_eligible_blocks_with_deterministic_ties(library):
    api().build_index(library)
    found = api().query(library, joint_positions=[0.] * 6, k=10)
    assert [r["block_id"] for r in found] == ["exact", "tie-a", "tie-b"]
    assert [r["distance"] for r in found] == [0., 1., 1.]
    assert all(r["similarity_eligible"] for r in found)


def test_metadata_queries_include_missing_and_invalid_states_with_reasons(library):
    api().build_index(library)
    found = {r["block_id"]: r for r in api().query(library, k=10)}
    assert set(found) == {"exact", "tie-a", "tie-b", "missing", "invalid"}
    assert found["missing"]["similarity_exclusion_reason"] == "missing_joint_positions"
    assert found["invalid"]["similarity_exclusion_reason"] == "joint_positions_must_be_finite_q6"
    assert all(row["distance"] is None for row in found.values())


def test_nonfinite_source_q_is_retained_without_nonstandard_json_output(library):
    path = library / "blocks.jsonl"
    blocks = [json.loads(line) for line in path.read_text().splitlines()]
    blocks[0]["q_start"][0] = float("nan")
    path.write_text("\n".join(json.dumps(block) for block in blocks))
    result = api().build_index(library)
    assert result["block_count"] == 5
    assert result["similarity_eligible_count"] == 2
    metadata = api().query(library, episode_id="episode-a", event="gripper_transition")[0]
    assert metadata["q_start"] is None
    assert "NaN" in metadata["q_start_source_json"]
    json.dumps(metadata, allow_nan=False)


def test_sqlite_filters_compose_before_similarity(library):
    api().build_index(library)
    found = api().query(library, joint_positions=np.zeros(6), episode_id="episode-a",
                        date="2026-01-01", event="gripper_transition", at_s=.5, k=5)
    assert [r["block_id"] for r in found] == ["tie-b"]
    assert api().query(library, episode_id="episode-a", date="2026-01-02") == []


def test_time_filter_matches_closed_intervals(library):
    api().build_index(library)
    assert [r["block_id"] for r in api().query(library, episode_id="episode-a", at_s=3.)] == ["tie-a"]
    assert api().query(library, episode_id="episode-a", at_s=3.01) == []


def test_event_filter_is_exact_and_sql_parameterized(library):
    api().build_index(library)
    assert len(api().query(library, event="gripper_transition")) == 2
    assert api().query(library, event="gripper") == []
    assert api().query(library, episode_id="' OR 1=1 --") == []


def test_causal_cutoff_filters_full_windows_before_ranking(library):
    api().build_index(library)
    # The later tied-neighbor would otherwise win the stable block-ID tie.
    rows=api().query(library,episode_id='episode-a',joint_positions=[0.]*6,
                     before_monotonic_ns=1_000_000_000,k=1)
    assert [r['block_id'] for r in rows]==['tie-b']
    assert api().query(library,episode_id='episode-a',before_monotonic_ns=999_999_999)==[]


@pytest.mark.parametrize('bad',[True,-1,1.5,'100'])
def test_causal_cutoff_requires_exact_nonnegative_integer_and_episode(library,bad):
    api().build_index(library)
    with pytest.raises(ValueError,match='monotonic|episode'):
        api().query(library,episode_id='episode-a',before_monotonic_ns=bad)
    with pytest.raises(ValueError,match='episode'):
        api().query(library,before_monotonic_ns=1_000_000_000)


def test_record_path_points_to_original_full_observation_action_record(library):
    api().build_index(library)
    result = api().query(library, event="unknown_state")[0]
    path = Path(result["record_absolute_path"])
    assert path.is_relative_to(library)
    assert path == library / result["record_path"]
    assert set(json.loads(path.read_text())) == {"O0", "A", "receipt", "O1"}


@pytest.mark.parametrize("q", [[], [0.] * 5, [0.] * 7, [[0.] * 6],
                              [np.nan] * 6, [np.inf] * 6, ["bad"] * 6])
def test_query_rejects_nonfinite_or_non_q6_joint_positions(library, q):
    api().build_index(library)
    with pytest.raises(ValueError, match="finite q6"):
        api().query(library, joint_positions=q)


@pytest.mark.parametrize("k", [0, -1, 1.5, True, 1001])
def test_query_k_constraints(library, k):
    api().build_index(library)
    with pytest.raises(ValueError, match="k"):
        api().query(library, k=k)


@pytest.mark.parametrize("at_s", [float("nan"), float("inf"), -1., "bad"])
def test_invalid_time_filter_rejected(library, at_s):
    api().build_index(library)
    with pytest.raises(ValueError, match="at_s"):
        api().query(library, at_s=at_s)


def test_joint_query_with_only_missing_states_returns_empty_without_fitting(library):
    api().build_index(library)
    assert api().query(library, joint_positions=[0.] * 6, event="unknown_state") == []


def test_rebuild_is_idempotent_and_does_not_change_library_inputs(library):
    before = {p.relative_to(library).as_posix(): p.read_bytes()
              for p in library.rglob("*") if p.is_file()}
    api().build_index(library)
    first = api().query(library, joint_positions=[0.] * 6, k=10)
    api().build_index(library)
    assert api().query(library, joint_positions=[0.] * 6, k=10) == first
    assert all((library / relative).read_bytes() == content for relative, content in before.items())
    assert not list(library.rglob("*.pkl"))


def test_duplicate_block_id_is_rejected_without_destroying_previous_index(library):
    api().build_index(library)
    path = library / "blocks.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate block_id"):
        api().build_index(library)
    assert len(api().query(library, k=10)) == 5


def test_record_path_escape_is_rejected(library):
    path = library / "blocks.jsonl"
    blocks = [json.loads(line) for line in path.read_text().splitlines()]
    blocks[0]["record_path"] = "../outside.json"
    path.write_text("\n".join(json.dumps(b) for b in blocks))
    with pytest.raises(ValueError, match="record_path"):
        api().build_index(library)


def test_unbuilt_index_is_read_only_failure(library):
    with pytest.raises(FileNotFoundError, match="build_index"):
        api().query(library)
    assert not (library / "retrieval.sqlite3").exists()
