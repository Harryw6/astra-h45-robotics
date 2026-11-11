"""On-demand selections use actual MCAP RGB, measured NPZ, and native compile."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "work/wam_demo_library/vendor"))


def api():
    assert importlib.util.find_spec("wam_astra.selection") is not None, "selection adapter is missing"
    return importlib.import_module("wam_astra.selection")


def hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def selected_library(tmp_path):
    from mcap.writer import Writer

    raw = tmp_path / "raw"
    raw.mkdir()
    mcap = raw / "camera.mcap"
    with mcap.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema("heterovla.raw_image.v1", "heterovla-layout-v1", b"{}")
        channel = writer.register_channel("/camera/front", "heterovla.raw_image.v1", schema)
        for i, t in enumerate([2_000_000_000, 2_100_000_000]):
            meta = {"camera": "front", "monotonic_ns": t, "wall_time_ns": t + 10_000_000_000,
                    "width": 2, "height": 1, "stride": 6, "pixel_format": "bgr8", "dtype": "uint8", "seq": i}
            encoded = json.dumps(meta).encode()
            pixels = bytes([0, 0, 255] * 2) if i == 0 else bytes([255, 0, 0] * 2)
            writer.add_message(channel, t + 10_000_000_000,
                               struct.pack("<I", len(encoded)) + encoded + pixels, t + 10_000_000_000, i)
        writer.finish()
    library = tmp_path / "library"
    episode = library / "episodes/episode-a"
    episode.mkdir(parents=True)
    times = np.arange(12, dtype=np.int64) * 100_000_000 + 1_000_000_000
    np.savez_compressed(episode / "signals.npz", state_t_ns=times,
                        state_q=np.repeat(np.arange(12)[:, None] * .01, 6, axis=1),
                        state_g=np.full(12, .02), state_source_row=np.arange(12),
                        command_q=np.full((12, 6), 99.), command_t_ns=times)
    record = {"schema_version": 1, "episode_id": "episode-a", "block_id": "late-dense", "kind": "dense",
              "start_ns": 2_000_000_000, "end_ns": 2_100_000_000, "start_s": 1., "end_s": 1.1,
              "O0": {"piper": {"joint_measured_rad": [99.] * 6}},
              "A": {"samples": [{"piper": {"recorded_joint_target": [99.] * 6}}]},
              "receipt": {"hardware_acknowledged": None}, "O1": {"piper": None}}
    path = episode / "block.json"
    path.write_text(json.dumps(record))
    row = {"episode_id": "episode-a", "block_id": "late-dense", "record_path": "episodes/episode-a/block.json"}
    catalog = {"episode_id": "episode-a", "origin_monotonic_ns": 1_000_000_000,
               "signals_path": "episodes/episode-a/signals.npz",
               "camera_shards": {"front": [{"path": str(mcap), "first_monotonic_ns": 2_000_000_000,
                                               "last_monotonic_ns": 2_100_000_000,
                                               "first_wall_time_ns": 12_000_000_000}]}}
    (library / "manifest.json").write_text(json.dumps({"source_root": str(raw), "episodes": [catalog]}))
    return library, raw, row


def test_real_rgb_selection_preserves_times_measured_states_and_sources(selected_library, tmp_path):
    library, raw, row = selected_library
    library_before, raw_before = hashes(library), hashes(raw)
    result = api().export_selection(library, [row], tmp_path / "selection", "Inspect current action", compile_contexts=False)
    assert result["image_count"] == 2
    assert result["block_count"] == 1
    assert result["compiled_context_count"] == 0
    episode = result["episodes"][0]
    demo_path = Path(episode["source_demo"])
    demo = json.loads(demo_path.read_text())
    assert demo_path == tmp_path / "selection/sources/episode-a/demo.json"
    assert [f["t_s"] for f in demo["keyframes"]] == [1., 1.1]
    assert demo["keyframes"][0]["state"]["piper"]["joint_measured_rad"] == [.1] * 6
    samples = demo["keyframes"][0]["action"]["samples"]
    assert samples[0]["t_s"] == 1.
    assert samples[0]["piper"]["joint_measured_rad"] == [.1] * 6
    assert "recorded_joint_target" not in json.dumps(demo)
    assert Image.open(demo_path.parent / demo["keyframes"][0]["images"]["front"]).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(demo_path.parent / demo["keyframes"][1]["images"]["front"]).getpixel((0, 0)) == (0, 0, 255)
    receipts = demo["keyframes"][0]["alignment"]["camera_receipts"]
    assert receipts["front"]["metadata"]["monotonic_ns"] == 2_000_000_000
    assert receipts["front"]["raw_message_sha256"]
    assert hashes(library) == library_before
    assert hashes(raw) == raw_before
    assert Path(result["selection_receipt"]).is_file()
    assert result["file_count"] == len([p for p in (tmp_path / "selection").rglob("*") if p.is_file()])


def test_missing_camera_creates_no_fake_images_or_compiled_bundle(selected_library, tmp_path):
    library, _, row = selected_library
    manifest_path = library / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"][0]["camera_shards"] = {"front": []}
    manifest_path.write_text(json.dumps(manifest))
    result = api().export_selection(library, [row], tmp_path / "missing", "Inspect")
    assert result["image_count"] == 0
    assert result["compiled_context_count"] == 0
    assert result["episodes"][0]["context_status"] == "unavailable"
    assert "no_actual_reference_images" in result["episodes"][0]["issues"]
    assert not list((tmp_path / "missing").rglob("*.png"))
    assert not (tmp_path / "missing/contexts/episode-a").exists()


def test_missing_signals_never_promotes_recorded_commands_or_forged_endpoints(selected_library, tmp_path):
    library, _, row = selected_library
    (library / "episodes/episode-a/signals.npz").unlink()
    result = api().export_selection(library, [row], tmp_path / "missing-state", "Inspect")
    episode = result["episodes"][0]
    assert episode["context_status"] == "unavailable"
    assert "measured_signals_unavailable" in episode["issues"]
    demo = json.loads(Path(episode["source_demo"]).read_text())
    assert all(not frame.get("state") and not frame.get("action") for frame in demo["keyframes"])
    assert "recorded_joint_target" not in json.dumps(demo)


def test_missing_endpoint_rgb_keeps_measured_segment_in_exported_record(selected_library, tmp_path):
    library, _, row = selected_library
    path = library / row["record_path"]
    record = json.loads(path.read_text())
    record["start_ns"] = 1_000_000_000
    path.write_text(json.dumps(record))
    result = api().export_selection(library, [row], tmp_path / "partial-rgb", "Inspect", compile_contexts=False)
    episode = result["episodes"][0]
    assert "endpoint_omitted_without_images:1000000000" in episode["issues"]
    exported = json.loads(Path(episode["blocks"][0]["record_path"]).read_text())
    samples = exported["selection_measured_reference"]["samples"]
    assert len(samples) == 11
    assert samples[0]["t_s"] == 0.
    assert samples[-1]["t_s"] == 1.
    assert len(json.loads(Path(episode["source_demo"]).read_text())["keyframes"]) == 1


def test_actual_rgb_times_remain_distinct_from_requested_endpoint_times(selected_library, tmp_path):
    library, _, row = selected_library
    path = library / row["record_path"]
    record = json.loads(path.read_text())
    record["start_ns"] += 20_000_000
    record["end_ns"] += 20_000_000
    path.write_text(json.dumps(record))
    result = api().export_selection(library, [row], tmp_path / "skew", "Inspect", compile_contexts=False)
    demo = json.loads(Path(result["episodes"][0]["source_demo"]).read_text())
    assert [f["t_s"] for f in demo["keyframes"]] == [1.02, 1.12]
    assert [f["image_times"]["front"] for f in demo["keyframes"]] == [1., 1.1]
    assert all(f["alignment"]["camera_receipts"]["front"]["skew_ns"] == -20_000_000 for f in demo["keyframes"])


def test_actual_native_compiled_selection_is_usable_as_astra_input(selected_library, tmp_path):
    library, _, row = selected_library
    times = 2_000_000_000 + np.arange(5, dtype=np.int64) * 20_000_000
    np.savez_compressed(library / "episodes/episode-a/signals.npz", state_t_ns=times,
                        state_q=np.full((5, 6), .1), state_g=[.02, .02, 0., .02, .02],
                        state_source_row=np.arange(10, 15))
    result = api().export_selection(library, [row], tmp_path / "compiled", "Review current Piper observations")
    episode = result["episodes"][0]
    assert result["compiled_context_count"] == 1
    assert episode["context_status"] == "compiled"
    context = episode["context"]
    assert Path(context["destination"]) == tmp_path / "compiled/contexts/episode-a"
    native_input = json.loads(Path(context["input_json"]).read_text())
    assert native_input["model"] == "gpt-6-astra"
    assert native_input["instruction"] == "Review current Piper observations"
    items = json.loads(Path(context["app_server_items_json"]).read_text())
    assert len([item for item in items if item["type"] == "image"]) == 2
    provenance = json.loads(Path(context["provenance_json"]).read_text())
    assert provenance["adapter"]["upstream_commit"] == "2ca201638e1139fe35973c8a2d209a894b389db1"
    assert provenance["adapter"]["model_calls"] == 0
    original = json.loads(Path(episode["source_demo"]).read_text())
    assert original["keyframes"][0]["action"]["preserve_sample_indices"] == [1, 2, 3]
    native = json.loads(Path(context["demo_json"]).read_text())
    samples = native["keyframes"][0]["action"]["samples"]
    assert [s["left"]["gripper_measured_m"] for s in samples] == [.02, .02, 0., .02, .02]
    assert provenance["output_action_samples"] == 5


@pytest.mark.parametrize("kind", ["library", "raw", "existing", "wam"])
def test_unsafe_or_existing_destination_is_rejected_without_writes(selected_library, tmp_path, kind):
    library, raw, row = selected_library
    destinations = {"library": library / "nested", "raw": raw / "nested",
                    "existing": tmp_path / "existing", "wam": Path("E:/WAM/selection-must-not-exist")}
    if kind == "existing":
        destinations[kind].mkdir()
    before = hashes(library)
    with pytest.raises((ValueError, FileExistsError)):
        api().export_selection(library, [row], destinations[kind], "Inspect", compile_contexts=False)
    assert hashes(library) == before
    if kind != "existing":
        assert not destinations[kind].exists()


def test_record_path_escape_rejected_before_output_creation(selected_library, tmp_path):
    library, _, row = selected_library
    row["record_path"] = "../outside.json"
    with pytest.raises(ValueError, match="record_path"):
        api().export_selection(library, [row], tmp_path / "escape", "Inspect", compile_contexts=False)
    assert not (tmp_path / "escape").exists()


def test_more_than_24_original_endpoints_rejected_without_silent_sampling(selected_library, tmp_path):
    library, _, row = selected_library
    original = json.loads((library / row["record_path"]).read_text())
    records = []
    for i in range(13):
        block = {**original, "block_id": f"block-{i}", "start_ns": 3_000_000_000 + i * 200_000_000,
                 "end_ns": 3_100_000_000 + i * 200_000_000}
        relative = f"episodes/episode-a/block-{i}.json"
        (library / relative).write_text(json.dumps(block))
        records.append({**row, "block_id": block["block_id"], "record_path": relative})
    with pytest.raises(ValueError, match="24"):
        api().export_selection(library, records, tmp_path / "over-budget", "Inspect", compile_contexts=False)
    assert not (tmp_path / "over-budget").exists()


def test_endpoint_order_and_duplicate_blocks_are_preserved_without_duplicate_frames(selected_library, tmp_path):
    library, _, row = selected_library
    result = api().export_selection(library, [row, row], tmp_path / "dedup", "Inspect", compile_contexts=False)
    assert result["block_count"] == 1
    assert result["image_count"] == 2
    demo = json.loads(Path(result["episodes"][0]["source_demo"]).read_text())
    assert len(demo["keyframes"]) == 2


@pytest.mark.parametrize("instruction", ["", "   ", None])
def test_instruction_required_before_writes(selected_library, tmp_path, instruction):
    library, _, row = selected_library
    with pytest.raises(ValueError, match="instruction"):
        api().export_selection(library, [row], tmp_path / "no-instruction", instruction, compile_contexts=False)
    assert not (tmp_path / "no-instruction").exists()
