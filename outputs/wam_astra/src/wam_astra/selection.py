"""Materialize retrieved historical blocks with real RGB and native contexts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np

from .contexts import _under_wam, compile_demo
from .data import file_sha256, measured_action, observation_at, write_json
from .media import extract_frame


def _within(root, relative, label):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a relative library path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} must remain within library")
    return path


def _identifier(value, label):
    if not isinstance(value, str) or value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} must be a safe, nonempty file identifier")
    return value


def _ns(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be integer monotonic nanoseconds")
    return value


def _measured_state(root, episode):
    """Read state arrays only; command targets cannot become demonstrations."""
    issues = []
    try:
        path = _within(root, episode["signals_path"], "signals_path")
        with np.load(path, allow_pickle=False) as signals:
            state = {key: signals["state_" + key].copy() for key in ("t_ns", "q", "source_row")}
            state["g"] = (signals["state_g"].copy() if "state_g" in signals
                          else np.full(len(state["t_ns"]), np.nan))
        n = len(state["t_ns"])
        if (state["t_ns"].shape != (n,) or state["t_ns"].dtype.kind not in "iu"
                or state["q"].shape != (n, 6) or state["g"].shape != (n,)
                or state["source_row"].shape != (n,)
                or state["source_row"].dtype.kind not in "iu"
                or (n > 1 and np.any(np.diff(state["t_ns"]) <= 0))):
            raise ValueError("invalid measured-state shape or nonmonotonic sample clock")
        state["q"] = np.asarray(state["q"], dtype=float)
        state["g"] = np.asarray(state["g"], dtype=float)
        if not np.isfinite(state["q"]).all():
            raise ValueError("measured q6 must be finite")
        return state, issues, {"path": str(path), "sha256": file_sha256(path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, ["measured_signals_unavailable", type(exc).__name__ + ":" + str(exc)], None


def export_selection(library_root, records, destination, instruction, *, compile_contexts=True) -> dict:
    """Export selected blocks without modifying library or raw source.

    One source demo/context is produced per episode. Bounds are 24 distinct
    selected endpoint timestamps and 48 actual images per episode; no automatic
    sampling hides an over-budget selection. Existing destinations are refused.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty text")
    if _under_wam(destination):
        raise ValueError("destination must be outside E:/WAM")
    root = Path(library_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if _under_wam(target) or target.is_relative_to(root):
        raise ValueError("destination must be outside raw data and library")
    if target.exists():
        raise FileExistsError(f"destination already exists: {target}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("source_root") and target.is_relative_to(Path(manifest["source_root"]).resolve()):
        raise ValueError("destination must be outside raw source_root")
    catalogs = {episode["episode_id"]: episode for episode in manifest["episodes"]}
    selected, seen = {}, set()
    for row in records:
        source = _within(root, row["record_path"], "record_path")
        block = json.loads(source.read_text(encoding="utf-8-sig"))
        eid = _identifier(block["episode_id"], "episode_id")
        bid = _identifier(block["block_id"], "block_id")
        if eid not in catalogs or row.get("episode_id", eid) != eid or row.get("block_id", bid) != bid:
            raise ValueError("selected record identity must match library episode and block")
        if (eid, bid) in seen:
            continue
        seen.add((eid, bid))
        start, end = _ns(block["start_ns"], "start_ns"), _ns(block["end_ns"], "end_ns")
        if end < start:
            raise ValueError("selected record end_ns precedes start_ns")
        if not all(key in block for key in ("O0", "A", "receipt", "O1")):
            raise ValueError("selected record requires O0,A,receipt,O1")
        selected.setdefault(eid, []).append((start, end, block, source))
    if not selected:
        raise ValueError("records must contain at least one selected block")
    for eid, blocks in selected.items():
        timestamps = {t for start, end, _, _ in blocks for t in (start, end)}
        if len(timestamps) > 24:
            raise ValueError(f"episode {eid} exceeds 24 keyframes; select fewer blocks explicitly")
        origin = _ns(catalogs[eid].get("origin_monotonic_ns"), "origin_monotonic_ns")
        if min(timestamps) < origin:
            raise ValueError("selected endpoint precedes source episode origin")
        if catalogs[eid].get("signals_path"):
            _within(root, catalogs[eid]["signals_path"], "signals_path")

    target.mkdir(parents=True)
    receipt = {"schema_version": 1, "library_root": str(root), "destination": str(target),
               "instruction": instruction.strip(), "manifest_source_sha256": file_sha256(manifest_path),
               "numeric_reference_semantics": "historical_measured_states_not_issued_controls",
               "model_calls": 0, "block_count": len(seen), "image_count": 0,
               "compiled_context_count": 0, "episodes": []}
    for eid in sorted(selected):
        blocks = sorted(selected[eid], key=lambda b: (b[0], b[1], b[2]["block_id"]))
        catalog, source_dir = catalogs[eid], target / "sources" / eid
        source_dir.mkdir(parents=True)
        origin = catalog["origin_monotonic_ns"]
        state, issues, signal_source = _measured_state(root, catalog)
        entry = {"episode_id": eid, "source_demo": None, "source_signals": signal_source,
                 "context_status": "not_requested", "context": None, "issues": issues,
                 "blocks": [], "image_count": 0, "keyframe_count": 0}
        frames = {}
        for timestamp in sorted({t for a, b, _, _ in blocks for t in (a, b)}):
            images, image_receipts = {}, {}
            for camera, shards in sorted(catalog.get("camera_shards", {}).items()):
                _identifier(camera, "camera")
                image_path = source_dir / "images" / f"{timestamp-origin:015d}_{camera}.png"
                result = extract_frame(shards, timestamp, image_path)
                image_receipts[camera] = result
                if result["available"]:
                    images[camera] = image_path.relative_to(source_dir).as_posix()
                    entry["image_count"] += 1
                else:
                    issues.append(f"missing_actual_frame:{timestamp}:{camera}")
            if not catalog.get("camera_shards"):
                issues.append(f"camera_catalog_unavailable:{timestamp}")
            observation = observation_at(state, timestamp) if state is not None else {"piper": None}
            piper = copy.deepcopy(observation.get("piper"))
            if piper and piper.get("gripper_measured_m") is None:
                piper.pop("gripper_measured_m", None)
            frame = {"t_s": (timestamp - origin) / 1.e9, "images": images,
                     "image_times": {camera: (int(result["metadata"]["monotonic_ns"]) - origin) / 1.e9
                                     for camera, result in image_receipts.items() if result["available"]},
                     "state": {"piper": piper} if piper else {},
                     "alignment": {"requested_monotonic_ns": timestamp,
                                   "origin_monotonic_ns": origin, "camera_receipts": image_receipts,
                                   "state_monotonic_ns": observation.get("state_monotonic_ns"),
                                   "state_source_row": observation.get("source_row"),
                                   "state_age_ns": observation.get("age_ns")}}
            intervals = [(a, b) for a, b, _, _ in blocks if a == timestamp]
            if state is not None and intervals:
                # Same-start windows form one contiguous measured segment.
                # The shared adapter supplies native preserve_sample_indices
                # for every recorded metre-valued aperture change.
                action = measured_action(state, timestamp, max(end for _, end in intervals), origin)
                if action["samples"]:
                    frame["action"] = action
            frames[timestamp] = frame
        for start, end, original, source in blocks:
            exported = copy.deepcopy(original)
            exported["selection_source"] = {"path": str(source), "sha256": file_sha256(source)}
            if state is not None:
                exported["selection_measured_reference"] = measured_action(state, start, end, origin)
            for key, timestamp in (("O0", start), ("O1", end)):
                exported[key]["images"] = frames[timestamp]["alignment"]["camera_receipts"]
            path = target / "records" / eid / (original["block_id"] + ".json")
            write_json(path, exported)
            entry["blocks"].append({"block_id": original["block_id"], "record_path": str(path),
                                    "source_record_path": str(source), "source_sha256": file_sha256(source)})
        valid_frames = [frame for _, frame in sorted(frames.items()) if frame["images"]]
        issues.extend(f"endpoint_omitted_without_images:{timestamp}" for timestamp, frame in frames.items()
                      if not frame["images"])
        entry["keyframe_count"] = len(valid_frames)
        receipt["image_count"] += entry["image_count"]
        if entry["image_count"] > 48:
            issues.append("image_budget_exceeded:48")
            entry["context_status"] = "unavailable"
        elif not valid_frames:
            issues.append("no_actual_reference_images")
            entry["context_status"] = "unavailable"
        else:
            demo = {"schema_version": 1, "title": eid,
                    "source": str(root), "demonstrator": "recorded_human_teleoperation",
                    "summary": "Selected historical measured states and actual RGB. Task outcome unverified.",
                    "outcome": "unknown", "source_robot": {"model": "AgileX Piper", "bimanual": False},
                    "coordinate_frame": "joint_angles_rad; gripper_measured_m; shared_host_monotonic_time",
                    "numeric_reference_semantics": "measured_states", "keyframes": valid_frames}
            source_demo = source_dir / "demo.json"
            write_json(source_demo, demo)
            entry["source_demo"] = str(source_demo)
            if not any(frame.get("state") or frame.get("action") for frame in valid_frames):
                issues.append("no_usable_measured_state_for_native_context")
                entry["context_status"] = "unavailable"
            elif compile_contexts:
                try:
                    entry["context"] = compile_demo(source_demo, target / "contexts" / eid, instruction.strip())
                    entry["context_status"] = "compiled"
                    receipt["compiled_context_count"] += 1
                except (RuntimeError, OSError, ValueError) as exc:
                    entry["context_status"] = "unavailable"
                    issues.append("native_context_compilation_failed:" + str(exc))
        entry["issues"] = list(dict.fromkeys(issues))
        receipt["episodes"].append(entry)
    receipt_path = target / "selection_receipt.json"
    receipt["selection_receipt"] = str(receipt_path)
    receipt["manifest"] = str(receipt_path)
    receipt["files"] = sorted([str(path) for path in target.rglob("*") if path.is_file()] + [str(receipt_path)])
    receipt["file_count"] = len(receipt["files"])
    write_json(receipt_path, receipt)
    return receipt
