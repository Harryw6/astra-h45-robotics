"""Causal selection uses recorded captures, never composite presentation time."""
import csv
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def row(frame, front, wrist, drone, error="1"):
    result = {"output_frame": str(frame), "output_time_s": str(frame / 30),
              "inside_manifest_requested_interval": "1"}
    for camera, timestamp in zip(("front", "wrist", "drone"), (front, wrist, drone)):
        result[camera + "_mapped_relative_time_s"] = str(timestamp)
        result[camera + "_clock_error_ms"] = error
        result[camera + "_source_frame"] = str(frame)
    return result


def test_latest_composite_rejects_future_component_despite_past_output_pts():
    from wam_astra.multiview_reference import select_causal_frame
    rows = [row(30, .99, .99, 1.01), row(29, .95, .96, .97)]
    found = select_causal_frame(rows, common_start_ns=1_000_000_000,
                                cutoff_desktop_ns=2_000_000_000, cutoff_error_ns=1_000_000)
    assert found["output_frame"] == 29
    assert found["evidence_upper_desktop_ns"] == 1_971_000_000
    assert found["cutoff_lower_desktop_ns"] == 1_999_000_000


def test_sampling_skew_is_already_in_component_capture_not_added_twice():
    from wam_astra.multiview_reference import select_causal_frame
    found = select_causal_frame([row(1, .91, .98, .95)], common_start_ns=0,
                                cutoff_desktop_ns=982_000_000, cutoff_error_ns=1_000_000)
    assert found["output_frame"] == 1  # Equality is causal; span is not another clock bound.


@pytest.mark.parametrize("bad", ["", "nan", "-1", "-0.0000000001", None])
def test_missing_invalid_clock_error_never_defaults_to_zero(bad):
    from wam_astra.multiview_reference import select_causal_frame
    candidate = row(1, .1, .1, .1)
    candidate["drone_clock_error_ms"] = bad
    assert select_causal_frame([candidate], common_start_ns=0,
                              cutoff_desktop_ns=1_000_000_000, cutoff_error_ns=0) is None


def test_monotonic_cutoff_uses_prior_recorded_clock_anchor():
    from wam_astra.multiview_reference import clock_anchor
    rows = [{"source_time_ns": "100", "desktop_time_ns": "1000000", "clock_estimated_error_ns": "3"},
            {"source_time_ns": "200", "desktop_time_ns": "1000100", "clock_estimated_error_ns": "4"}]
    found = clock_anchor(rows, 199)
    assert found["source_time_ns"] == 100
    assert found["desktop_time_ns"] == 1000000
    assert found["clock_estimated_error_ns"] == 3
    assert clock_anchor(rows, 99) is None


def test_unavailable_source_keeps_original_demo_and_compilation_explicit(tmp_path):
    from wam_astra.multiview_reference import enrich_reference
    source_dir = tmp_path / "selection" / "sources" / "episode"
    source_dir.mkdir(parents=True)
    (source_dir / "front.png").write_bytes(b"original-image")
    demo = {"schema_version": 1, "title": "episode", "keyframes": [
        {"t_s": 1., "images": {"front": "front.png"}, "state": {"piper": {}},
         "alignment": {"origin_monotonic_ns": 100, "requested_monotonic_ns": 1_000_000_100}}]}
    source = source_dir / "demo.json"
    source.write_text(json.dumps(demo))
    before = source.read_bytes()
    result = enrich_reference(source, None, tmp_path / "derived", "Review", compile_contexts=False)
    assert source.read_bytes() == before
    assert result["keyframes"][0]["available"] is False
    assert "unavailable" in result["keyframes"][0]["issue"]
    exported = json.loads(Path(result["source_demo_json"]).read_text())
    assert exported["keyframes"][0]["state"] == demo["keyframes"][0]["state"]
    assert set(exported["keyframes"][0]["images"]) == {"front"}
    assert result["context_status"] == "not_requested"


def test_destination_inside_original_selection_is_rejected(tmp_path):
    from wam_astra.multiview_reference import enrich_reference
    source = tmp_path / "source.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="source"):
        enrich_reference(source, None, tmp_path / "new", "Review", compile_contexts=False)


def test_two_original_cameras_with_same_filename_keep_distinct_bytes(tmp_path):
    from wam_astra.multiview_reference import enrich_reference
    source_dir = tmp_path / "selection"
    for camera in ("front", "wrist"):
        (source_dir / camera).mkdir(parents=True)
        (source_dir / camera / "rgb.png").write_bytes(camera.encode())
    demo = {"schema_version": 1, "keyframes": [{"t_s": 0., "images": {
        "front": "front/rgb.png", "wrist": "wrist/rgb.png"}}]}
    source = source_dir / "demo.json"
    source.write_text(json.dumps(demo))
    result = enrich_reference(source, None, tmp_path / "derived", "Review", compile_contexts=False)
    output = Path(result["source_demo_json"])
    frame = json.loads(output.read_text())["keyframes"][0]
    for camera in ("front", "wrist"):
        assert (output.parent / frame["images"][camera]).read_bytes() == camera.encode()


def test_real_ffmpeg_and_native_compiler_keep_pixels_motion_and_causal_receipts(tmp_path):
    """Small generated fixture only; real example uses unmodified E:/WAM RGB."""
    from PIL import Image
    from wam_astra.data import file_sha256
    from wam_astra.multiview_reference import enrich_reference
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for actual-video integration")
    source_dir = tmp_path / "selection"
    source_dir.mkdir()
    Image.new("RGB", (16, 16), "blue").save(source_dir / "front.png")
    original_pixel_hash = file_sha256(source_dir / "front.png")
    demo = {"schema_version": 1, "title": "episode", "keyframes": [
        {"t_s": 0., "images": {"front": "front.png"},
         "state": {"piper": {"joint_measured_rad": [0.] * 6, "gripper_measured_m": .02}},
         "alignment": {"origin_monotonic_ns": 100, "requested_monotonic_ns": 100}}]}
    source = source_dir / "demo.json"
    source.write_text(json.dumps(demo))
    original_demo_hash = file_sha256(source)
    video = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=red:s=16x16:r=30:d=0.1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
                   check=True, capture_output=True, timeout=30)
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"episode_id": "episode", "clock_model_validated": True,
                                    "common_interval": {"start_desktop_ns": 1_000_000_000}}))
    cameras = tmp_path / "相机时间戳"
    cameras.mkdir()
    (cameras / "front.csv").write_text("source_time_ns,desktop_time_ns,clock_estimated_error_ns\n100,2000000000,1000000\n")
    rows = [row(0, .1, .1, .1), row(1, .2, .2, 1.2)]
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inspection = tmp_path / "inspection.json"
    inspection.write_text(json.dumps({"episode_id": "episode", "source_video": str(video),
        "alignment_path": str(alignment), "alignment_sha256": file_sha256(alignment),
        "index_path": str(index), "index_sha256": file_sha256(index)}))
    result = enrich_reference(source, inspection, tmp_path / "enriched", "Historical reference only")
    assert result["context_status"] == "compiled"
    assert result["sources_unchanged"] is True
    assert result["keyframes"][0]["output_frame"] == 0
    enriched = json.loads(Path(result["source_demo_json"]).read_text())
    frame = enriched["keyframes"][0]
    assert frame["state"] == demo["keyframes"][0]["state"]
    assert frame["alignment"]["camera_receipts"]["three_view"]["output_frame"] == 0
    assert file_sha256(source) == original_demo_hash
    assert file_sha256(source_dir / "front.png") == original_pixel_hash
    assert file_sha256(Path(result["source_demo_json"]).parent / frame["images"]["front"]) == original_pixel_hash
    native = json.loads(Path(result["context"]["demo_json"]).read_text())
    assert set(native["keyframes"][0]["images"]) == {"front", "three_view"}
    assert native["keyframes"][0]["state"]["left"] == frame["state"]["piper"]
    items = json.loads(Path(result["context"]["app_server_items_json"]).read_text())
    assert len([item for item in items if item["type"] == "image"]) == 2
