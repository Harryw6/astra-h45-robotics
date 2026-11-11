"""Add causal, recorded three-view RGB to an existing selected demonstration.

No models, codecs, clocks, images, motion or task labels are invented here.
FFmpeg decodes exact indexed frames; the existing native compiler handles input.
"""
from __future__ import annotations

import copy
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
from pathlib import Path
import shutil
import subprocess

from .contexts import _under_wam, compile_demo
from .data import file_sha256, write_json

CAMERAS = ("front", "wrist", "drone")


def _integer(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    text = str(value)
    result = int(text)
    if result < 0 or str(result) != text:
        raise ValueError("requires a nonnegative integer")
    return result


def _scaled_ns(value, scale):
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    number = Decimal(str(value)) * scale
    if not number.is_finite():
        raise ValueError("requires a finite timestamp")
    return int(number.to_integral_value(rounding=ROUND_CEILING))


def clock_anchor(rows, requested_monotonic_ns):
    """Latest actual local capture at/before cutoff; never infer clock offsets.

    Its recorded desktop clock minus its uncertainty is a conservative cutoff.
    Using the earlier capture avoids interpolating a new wall/monotonic mapping.
    """
    requested = _integer(requested_monotonic_ns)
    best = None
    for row in rows:
        try:
            candidate = {key: _integer(row[key]) for key in
                         ("source_time_ns", "desktop_time_ns", "clock_estimated_error_ns")}
        except (ValueError, KeyError, TypeError):
            continue
        if candidate["source_time_ns"] <= requested and (best is None or
                candidate["source_time_ns"] > best["source_time_ns"]):
            best = {**candidate, "recorded_row": dict(row)}
    return best


def select_causal_frame(rows, *, common_start_ns, cutoff_desktop_ns, cutoff_error_ns):
    """Latest output frame whose every component's clock upper bound is past.

    Relative component timestamps already include sampling mismatch. Each
    component clock error and cutoff anchor error are counted exactly once.
    Output PTS is retained as provenance only and never decides causality.
    """
    start, cutoff, error = map(_integer, (common_start_ns, cutoff_desktop_ns, cutoff_error_ns))
    lower = cutoff - error
    best = None
    for row in rows:
        try:
            frame = _integer(row["output_frame"])
            components = {}
            for camera in CAMERAS:
                _integer(row[camera + "_source_frame"])
                relative = row.get(camera + "_mapped_relative_time_s", row.get(camera + "_relative_time_s"))
                capture = start + _scaled_ns(relative, 1_000_000_000)
                if Decimal(str(row[camera + "_clock_error_ms"])) < 0:
                    raise ValueError("negative clock error")
                bound = _scaled_ns(row[camera + "_clock_error_ms"], 1_000_000)
                components[camera] = {"capture_desktop_ns": capture, "clock_error_ns": bound,
                                      "upper_desktop_ns": capture + bound}
            upper = max(component["upper_desktop_ns"] for component in components.values())
        except (ValueError, KeyError, TypeError, InvalidOperation):
            continue
        if upper <= lower and (best is None or frame > best["output_frame"]):
            best = {"output_frame": frame, "components": components,
                    "evidence_upper_desktop_ns": upper, "cutoff_lower_desktop_ns": lower,
                    "alignment_index_row": dict(row)}
    return best


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def enrich_reference(source_demo_json, inspection_provenance_json, destination, instruction,
                     *, compile_contexts=True, compiler_options=None, ffmpeg="ffmpeg"):
    """Return an isolated enhanced source plus original native compilation paths.

    Accept a selection's canonical demo.json or its archived source_demo.json.
    The inspection provenance supplies explicit video/index/alignment paths.
    Missing three-view evidence remains unavailable per keyframe. Existing
    destination, E:/WAM, library, source directories are never overwritten.
    """
    source = Path(source_demo_json).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().resolve()
    if _under_wam(target) or target == source.parent or source.parent in target.parents:
        raise ValueError("destination must be outside raw data and source directory")
    if target.exists():
        raise FileExistsError("destination already exists")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    demo = _read(source)
    library = Path(demo.get("source", source.parent)).resolve()
    if target == library or library in target.parents:
        raise ValueError("destination must be outside source library")
    if type(demo.get("schema_version")) is not int or demo["schema_version"] != 1:
        raise ValueError("source requires schema_version 1")
    if not isinstance(demo.get("keyframes"), list) or not demo["keyframes"]:
        raise ValueError("source requires keyframes")
    if any("three_view" in f.get("images", {}) for f in demo["keyframes"]):
        raise ValueError("three_view camera already present")
    tracked = {str(source): file_sha256(source)}
    receipt = {"schema_version": 1, "source_demo": str(source), "source_sha256": tracked[str(source)],
               "destination": str(target), "model_calls": 0, "hardware_calls": 0,
               "context_status": "not_requested", "context": None, "keyframes": [],
               "timing_semantics": "Each component capture upper bound <= prior local clock anchor lower bound <= keyframe cutoff; output PTS is not capture time. Native image label is endpoint time; component captures are in this receipt.",
               "uncertainty_limit": "Recorded clock-model bounds exclude camera exposure/driver latency; no physical-exposure bound is claimed."}
    def track(path):
        path = Path(path).resolve(strict=True)
        tracked[str(path)] = file_sha256(path)
        return path
    inputs, failure = None, "multiview_source_unavailable"
    if inspection_provenance_json is not None:
        try:
            provenance_path = track(inspection_provenance_json)
            provenance = _read(provenance_path)
            if provenance.get("episode_id") != demo.get("title"):
                raise ValueError("episode identity mismatch")
            alignment_path, index = track(provenance["alignment_path"]), track(provenance["index_path"])
            for key, path in (("alignment_sha256", alignment_path), ("index_sha256", index)):
                if provenance.get(key) != tracked[str(path)]:
                    raise ValueError(key + " mismatch")
            alignment = _read(alignment_path)
            if alignment.get("episode_id") != demo.get("title") or alignment.get("clock_model_validated") is not True:
                raise ValueError("alignment identity or clock validation unavailable")
            start = _integer(alignment["common_interval"]["start_desktop_ns"])
            video = track(provenance["source_video"])
            anchor_csv = track(alignment_path.parent / "相机时间戳" / "front.csv")
            inputs = (video, _csv(index), _csv(anchor_csv), start)
            receipt["inspection_provenance"] = str(provenance_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failure += ":" + type(exc).__name__ + ":" + str(exc)
    # Resolve and verify original RGB before any output is created. Archived
    # source_demo.json paths refer to its selection source, not native keyframes.
    originals = []
    for number, frame in enumerate(demo["keyframes"]):
        images = {}
        for camera, name in frame.get("images", {}).items():
            path = Path(name)
            path = path if path.is_absolute() else source.parent / path
            if not path.is_file():
                old = frame.get("alignment", {}).get("camera_receipts", {}).get(camera, {})
                path = Path(old.get("path", ""))
                if not path.is_file() or old.get("image_sha256") != file_sha256(path):
                    raise ValueError("original camera image unavailable or unverified: " + camera)
            images[camera] = track(path)
        originals.append(images)
        result = {"keyframe": number, "available": False, "issue": failure}
        if inputs:
            video, rows, anchors, start = inputs
            try:
                requested = _integer(frame["alignment"]["requested_monotonic_ns"])
                origin = _integer(frame["alignment"]["origin_monotonic_ns"])
                if abs(_scaled_ns(frame["t_s"], 1_000_000_000) - (requested - origin)) > 1:
                    raise ValueError("keyframe t_s/monotonic clock mismatch")
                anchor = clock_anchor(anchors, requested)
                if anchor is None:
                    raise ValueError("prior recorded clock anchor unavailable")
                found = select_causal_frame(rows, common_start_ns=start,
                    cutoff_desktop_ns=anchor["desktop_time_ns"], cutoff_error_ns=anchor["clock_estimated_error_ns"])
                result.update({"clock_anchor": anchor, "requested_monotonic_ns": requested,
                               "anchor_age_ns": requested - anchor["source_time_ns"]})
                if found:
                    result.update(found)
                    result.update(available=True, issue=None)
                else:
                    result["issue"] = "causal_three_view_frame_unavailable"
            except (ValueError, KeyError, TypeError) as exc:
                result["issue"] = "clock_mapping_unavailable:" + str(exc)
        receipt["keyframes"].append(result)
    target.mkdir(parents=True)
    copied = copy.deepcopy(demo)
    derived = target / "source"
    derived.mkdir()
    (target / "original_source_demo.json").write_bytes(source.read_bytes())
    for number, images in enumerate(originals):
        for camera_index, (camera, path) in enumerate(images.items()):
            out = derived / "images" / str(number) / str(camera_index) / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            copied["keyframes"][number]["images"][camera] = out.relative_to(derived).as_posix()
    selected = sorted({r["output_frame"] for r in receipt["keyframes"] if r["available"]})
    if selected:
        frames_dir = derived / "three_view"
        frames_dir.mkdir()
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(inputs[0]),
                   "-vf", "select=" + "+".join(f"eq(n\\,{i})" for i in selected),
                   "-fps_mode", "passthrough", "-start_number", "0", str(frames_dir / "%04d.png")]
        subprocess.run(command, check=True, capture_output=True, timeout=240)
        receipt["ffmpeg_command"] = command
        for result in receipt["keyframes"]:
            if result["available"]:
                image = frames_dir / f"{selected.index(result['output_frame']):04d}.png"
                if not image.is_file():
                    raise RuntimeError("FFmpeg did not decode exact indexed frame")
                result["path"], result["image_sha256"] = str(image), file_sha256(image)
                frame = copied["keyframes"][result["keyframe"]]
                frame["images"]["three_view"] = image.relative_to(derived).as_posix()
                frame.setdefault("alignment", {}).setdefault("camera_receipts", {})["three_view"] = copy.deepcopy(result)
    source_out = derived / "demo.json"
    write_json(source_out, copied)
    receipt["source_demo_json"] = str(source_out)
    receipt["enriched_source_sha256"] = file_sha256(source_out)
    if compile_contexts:
        receipt["context"] = compile_demo(source_out, target / "native_context", instruction,
                                           **(compiler_options or {}))
        receipt["context_status"] = "compiled"
    for path, digest in tracked.items():
        if file_sha256(path) != digest:
            raise RuntimeError("source changed during enrichment: " + path)
    receipt["sources_unchanged"] = True
    receipt["source_file_sha256"] = tracked
    receipt["provenance_json"] = str(target / "multiview_provenance.json")
    write_json(receipt["provenance_json"], receipt)
    return receipt
