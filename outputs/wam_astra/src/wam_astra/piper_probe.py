"""Receive-only Piper feedback using the unchanged official v0.6.1 decoder.

No SDK interface, port connection, query, command encoding or sender exists in
this adapter. Parsed frames are evidence, not physical-state verification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from threading import RLock
import time
from types import ModuleType, SimpleNamespace


UPSTREAM_REPOSITORY = "https://github.com/agilexrobotics/piper_sdk"
UPSTREAM_COMMIT = "081e7c588e5b79eeaefa67a0469bcc701c81014f"
UPSTREAM_TAG = "0.6.1"
SOURCE_CANONICAL_SHA256 = "920774da43c0a5ed614f57173bf9e8b918b86c5b798e01665946827ed6e59471"
DEFAULT_UPSTREAM_ROOT = Path(__file__).resolve().parents[4] / "work/upstream/piper_sdk"
_IMPORT_LOCK = RLock()
_CACHE = {}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite seconds")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite seconds")
    return float(value)


def _json(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("verification_context must be finite JSON") from exc


def load_native_feedback(upstream_root=DEFAULT_UPSTREAM_ROOT):
    """Hash canonical-LF native corpus before importing private decoder modules.

    Raw byte hashes are retained too. Only newline conversion is tolerated;
    root package initialization, interface and hardware-port modules are skipped.
    The interface source is hashed solely as the authoritative unit reference.
    """
    root = Path(upstream_root).resolve()
    fixed = ["piper_sdk/protocol/piper_protocol_base.py",
             "piper_sdk/protocol/protocol_v2/piper_protocol_v2.py",
             "piper_sdk/interface/piper_interface_v2.py", "piper_sdk/version.py"]
    files = {root / path for path in fixed}
    for directory in ("piper_sdk/utils", "piper_sdk/piper_msgs/msg_v2"):
        files.update((root / directory).rglob("*.py"))
    canonical, byte_hashes = [], {}
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(f"Pinned Piper SDK source missing: {path}")
        relative, raw = path.relative_to(root).as_posix(), path.read_bytes()
        byte_hashes[relative] = hashlib.sha256(raw).hexdigest()
        canonical.append(relative + ":" + hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
    digest = hashlib.sha256("\n".join(canonical).encode()).hexdigest()
    if digest != SOURCE_CANONICAL_SHA256:
        raise ValueError("Piper SDK canonical source hash mismatch; require unchanged official tag 0.6.1")
    with _IMPORT_LOCK:
        if str(root) in _CACHE:
            return _CACHE[str(root)]
        namespace = "_wam_piper_feedback_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        for suffix, relative in (("", "piper_sdk"), (".protocol", "piper_sdk/protocol"),
                                 (".protocol.protocol_v2", "piper_sdk/protocol/protocol_v2"),
                                 (".piper_msgs", "piper_sdk/piper_msgs")):
            module = ModuleType(namespace + suffix)
            module.__path__ = [str(root / relative)]
            sys.modules[module.__name__] = module

        def execute(suffix, relative, *, package=False):
            path = root / relative
            spec = importlib.util.spec_from_file_location(namespace + suffix, path,
                    submodule_search_locations=[str(path.parent)] if package else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module.__name__] = module
            spec.loader.exec_module(module)
            return module

        execute(".utils", "piper_sdk/utils/__init__.py", package=True)
        messages = execute(".piper_msgs.msg_v2", "piper_sdk/piper_msgs/msg_v2/__init__.py", package=True)
        execute(".protocol.piper_protocol_base", "piper_sdk/protocol/piper_protocol_base.py")
        decoder = execute(".protocol.protocol_v2.piper_protocol_v2", "piper_sdk/protocol/protocol_v2/piper_protocol_v2.py")
        result = SimpleNamespace(Parser=decoder.C_PiperParserV2, PiperMessage=messages.PiperMessage,
                                 CanIDPiper=messages.CanIDPiper, ArmMsgType=messages.ArmMsgType,
                                 ArmStatus=messages.ArmMsgFeedbackStatusEnum.ArmStatus,
                                 byte_hashes=byte_hashes, source_root=root)
        _CACHE[str(root)] = result
        return result


class PassivePiperFeedback:
    def __init__(self, upstream_root=None, *, max_feedback_age_s, clock=time.monotonic,
                 verification_context=None):
        self.max_feedback_age_s = _number(max_feedback_age_s, "max_feedback_age_s")
        if self.max_feedback_age_s <= 0 or not callable(clock):
            raise ValueError("max_feedback_age_s must be positive and clock callable")
        self.clock = clock
        self.verification_context = _json({} if verification_context is None else verification_context)
        self.native = load_native_feedback(DEFAULT_UPSTREAM_ROOT if upstream_root is None else upstream_root)
        self.parser = self.native.Parser()
        ids = self.native.CanIDPiper
        self._pair_fields = {
            ids.ARM_JOINT_FEEDBACK_12.value: ("joint_1", "joint_2"),
            ids.ARM_JOINT_FEEDBACK_34.value: ("joint_3", "joint_4"),
            ids.ARM_JOINT_FEEDBACK_56.value: ("joint_5", "joint_6"),
        }
        self._gripper_id, self._status_id = ids.ARM_GRIPPER_FEEDBACK.value, ids.ARM_STATUS_FEEDBACK.value
        self._frames = {}
        self.received_frames = self.ignored_frames = 0
        self._lock = RLock()

    def _now(self, now=None):
        value = _number(self.clock() if now is None else now, "monotonic timestamp")
        if value < 0:
            raise ValueError("monotonic timestamp must be nonnegative")
        return value

    def update(self, frame, *, received_at_s=None):
        """Decode one actual eight-byte RX feedback using native DecodeMessage.

        received_at_s is host monotonic arrival time. The CAN driver timestamp
        is retained as metadata and never used as the freshness clock.
        """
        import can
        with self._lock:
            now = self._now()
            received = now if received_at_s is None else self._now(received_at_s)
            if received > now:
                raise ValueError("receive timestamp is in the future")
            if not isinstance(frame, can.Message):
                raise ValueError("frame must be an actual python-can.Message")
            if frame.is_extended_id or frame.is_error_frame or frame.is_remote_frame or frame.is_fd or not frame.is_rx:
                raise ValueError("require a standard classical CAN received data frame")
            _number(frame.timestamp, "CAN timestamp metadata")
            identifier = int(frame.arbitration_id)
            recognized = identifier in self._pair_fields or identifier in (self._gripper_id, self._status_id)
            if not recognized:
                self.received_frames += 1
                self.ignored_frames += 1
                return {"status": "ignored_non_feedback", "arbitration_id": identifier}
            if len(frame.data) != 8 or frame.dlc != 8:
                raise ValueError("official feedback requires exactly 8 data bytes and DLC 8")
            if identifier in self._frames and received < self._frames[identifier]["received_at_s"]:
                raise ValueError("out-of-order receive timestamp cannot overwrite fresher feedback")
            message = self.native.PiperMessage()
            if not self.parser.DecodeMessage(frame, message):
                raise ValueError("native decoder did not recognize feedback frame")
            receipt = {"arbitration_id": identifier, "can_id": hex(identifier),
                       "native_message_type": message.type_.name, "received_at_s": received,
                       "can_timestamp": float(frame.timestamp), "data_hex": bytes(frame.data).hex()}
            if identifier in self._pair_fields:
                receipt["raw_millidegrees"] = [int(getattr(message.arm_joint_feedback, name))
                                               for name in self._pair_fields[identifier]]
            elif identifier == self._gripper_id:
                native = message.gripper_feedback
                receipt.update(raw_micrometres=int(native.grippers_angle),
                               raw_effort_milli_nm=int(native.grippers_effort), status_code=int(native.status_code),
                               foc_status={key: bool(value) for key, value in vars(native.foc_status).items()})
            else:
                native = message.arm_status_msgs
                for field in ("ctrl_mode", "arm_status", "mode_feed", "teach_status", "motion_status"):
                    enum = getattr(native, field)
                    receipt[field + "_raw"], receipt[field + "_name"] = int(enum), enum.name
                receipt.update(trajectory_num=int(native.trajectory_num), err_code=int(native.err_code),
                               err_status={key: bool(value) for key, value in vars(native.err_status).items()})
            self._frames[identifier] = receipt
            self.received_frames += 1
            return {"status": "decoded", **deepcopy(receipt)}

    def collect(self, now_s=None):
        """Snapshot only fresh received evidence; never expose SDK defaults."""
        with self._lock:
            now = self._now(now_s)
            frames = deepcopy(self._frames)
            for frame in frames.values():
                frame["age_s"] = now - frame["received_at_s"]
                frame["fresh"] = 0 <= frame["age_s"] <= self.max_feedback_age_s
            pairs = {hex(identifier): frames[identifier] for identifier in self._pair_fields if identifier in frames}
            missing = [hex(identifier) for identifier in self._pair_fields if identifier not in frames]
            stale = [hex(identifier) for identifier in self._pair_fields if identifier in frames and not frames[identifier]["fresh"]]
            joint_complete = not missing and not stale
            q = None
            if joint_complete:
                q = [value * math.pi / 180000. for identifier in self._pair_fields
                     for value in frames[identifier]["raw_millidegrees"]]
            gripper = frames.get(self._gripper_id)
            status = frames.get(self._status_id)
            fresh_gripper = gripper if gripper and gripper["fresh"] else None
            fresh_status = status if status and status["fresh"] else None
            aperture = fresh_gripper["raw_micrometres"] * 1.e-6 if fresh_gripper else None
            fault = bool(fresh_status and (fresh_status["err_code"] or fresh_status["arm_status_name"] not in {
                "NORMAL", "TEACHING_RECORD", "TEACHING_EXECUTION", "TEACHING_PAUSE", "UNKNOWN"}))
            if fresh_gripper:
                fault = fault or any(value for key, value in fresh_gripper["foc_status"].items()
                                     if key not in {"driver_enable_status", "homing_status"})
            complete = joint_complete and fresh_gripper is not None and fresh_status is not None
            availability = "fault_reported" if fault else ("fresh_complete" if complete else (
                "no_feedback" if not frames else "partial_or_stale"))
            times = [frames[identifier]["received_at_s"] for identifier in self._pair_fields if identifier in frames]
            def verification_label(key):
                return ("caller_attestation_provided" if isinstance(self.verification_context, dict)
                        and key in self.verification_context else "unknown")
            return {
                "status": availability, "collected_at_s": now, "feedback_complete": complete,
                "joint_feedback_complete": joint_complete, "joint_positions_rad": q,
                "joint_observed_at_s": min(times) if joint_complete else None,
                "joint_pair_time_span_s": max(times) - min(times) if len(times) == 3 else None,
                "joint_pairs": pairs, "missing_joint_pair_ids": missing, "stale_joint_pair_ids": stale,
                "gripper_measured_m": aperture, "gripper": fresh_gripper, "arm_status": fresh_status,
                "last_gripper_frame": gripper, "last_status_frame": status,
                "state": q + [aperture] if joint_complete and aperture is not None else None,
                "state_joint_units": "rad", "state_gripper_units": "m", "state_verified": False,
                "protocol_hardware_match": verification_label("protocol_hardware_match"),
                "firmware_verification": verification_label("firmware_version"),
                "physical_state_verification": verification_label("physical_state_verification"),
                "verification_context": deepcopy(self.verification_context),
                "verification_context_is_caller_attestation": True, "ready_for_actuation": False,
                "fault_indication_present": fault, "received_frames": self.received_frames,
                "ignored_frames": self.ignored_frames, "api_transmit_calls": 0,
                "upstream": {"repository": UPSTREAM_REPOSITORY, "tag": UPSTREAM_TAG,
                    "commit": UPSTREAM_COMMIT, "decoder_protocol": self.parser.GetParserProtocolVersion().name,
                    "source_root": str(self.native.source_root), "source_canonical_sha256": SOURCE_CANONICAL_SHA256,
                    "source_hash_policy": "sha256_per_file_then_aggregate_canonical_lf",
                    "source_byte_hashes": deepcopy(self.native.byte_hashes)},
            }


def transport_failure_snapshot(probe, error, *, phase):
    """Report an actual python-can transport failure, retaining prior evidence.

    Historical frames retain their receipt times and ages, but a failed
    transport cannot expose an active complete state or readiness flag.
    """
    import can
    if not isinstance(error, can.CanError):
        raise TypeError("transport failure must be an actual python-can.CanError")
    result = probe.collect()
    evidence_fields = ("status", "collected_at_s", "feedback_complete", "joint_feedback_complete",
                       "joint_positions_rad", "joint_observed_at_s", "gripper_measured_m",
                       "gripper", "arm_status", "state")
    result["last_received_evidence"] = {key: deepcopy(result[key]) for key in evidence_fields}
    result.update(status="transport_error", feedback_complete=False, joint_feedback_complete=False,
                  joint_positions_rad=None, joint_observed_at_s=None, gripper_measured_m=None,
                  gripper=None, arm_status=None, state=None, state_verified=False, ready_for_actuation=False,
                  transport_api_mode="receive_only", transport_ok=False,
                  transport_error={"type": type(error).__name__, "message": str(error),
                                   "phase": phase, "observed_at_s": result["collected_at_s"]})
    return result


def collect_feedback(receive, *, probe, duration_s):
    """Bounded RX-only callback loop; no bus/sender/control API is accepted."""
    duration = _number(duration_s, "duration_s")
    if duration <= 0 or not callable(receive):
        raise ValueError("duration_s must be positive and receive callable")
    import can
    start, errors, error_count, transport_error = probe._now(), [], 0, None
    deadline = start + duration
    while probe._now() < deadline:
        try:
            frame = receive(timeout=min(.1, max(0., deadline - probe._now())))
        except can.CanError as exc:
            transport_error = exc
            break
        if frame is not None:
            try:
                probe.update(frame, received_at_s=probe._now())
            except ValueError as exc:
                error_count += 1
                if len(errors) < 16:
                    errors.append(str(exc))
    result = (transport_failure_snapshot(probe, transport_error, phase="receive")
              if transport_error is not None else probe.collect())
    if transport_error is None:
        result["transport_ok"] = True
    result.update(transport_api_mode="receive_only", collection_started_at_s=start,
                  collection_duration_s=probe._now() - start, invalid_input_count=error_count,
                  invalid_input_examples=errors)
    return result
