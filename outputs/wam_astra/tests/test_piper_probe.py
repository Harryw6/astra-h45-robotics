"""Actual official parser and python-can frames; no bus or hardware connection."""
import ast
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil
import struct
import sys

import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
SDK = WORKSPACE / "work/upstream/piper_sdk"
sys.path.insert(0, str(WORKSPACE / "work/wam_demo_library/vendor"))
import can


def api():
    assert importlib.util.find_spec("wam_astra.piper_probe") is not None, "passive native feedback probe is missing"
    return importlib.import_module("wam_astra.piper_probe")


class Clock:
    def __init__(self):
        self.now = 100.
    def __call__(self):
        return self.now


def probe(clock=None, **changes):
    config = {"max_feedback_age_s": .25, "clock": clock or Clock()}
    config.update(changes)
    return api().PassivePiperFeedback(SDK, **config)


def message(identifier, data, **changes):
    options = {"arbitration_id": identifier, "data": data, "is_extended_id": False,
               "timestamp": 1791793200.0, "is_rx": True}
    options.update(changes)
    return can.Message(**options)


def joint_message(identifier, first=0, second=0):
    # Official protocol: two signed big-endian int32 millidegree fields.
    return message(identifier, struct.pack(">ii", first, second))


def all_feedback(p, clock, *, zero=False):
    values = [(0, 0)] * 3 if zero else [(90000, -90000), (1000, -1000), (1, -1)]
    for index, pair in enumerate(values):
        p.update(joint_message(0x2A5 + index, *pair), received_at_s=clock())
    p.update(message(0x2A8, struct.pack(">ihBB", 24500, -123, 0x40, 0)), received_at_s=clock())
    p.update(message(0x2A1, bytes([1, 0, 1, 0, 0, 0, 0, 0])), received_at_s=clock())


def test_no_packets_never_expose_native_constructor_zeros_as_measurements():
    result = probe().collect()
    assert result["status"] == "no_feedback"
    assert result["joint_positions_rad"] is None
    assert result["gripper_measured_m"] is None
    assert result["feedback_complete"] is False
    assert result["state_verified"] is False


def test_provided_verification_is_labeled_caller_context_not_decoder_proof():
    p = probe(verification_context={"firmware_version": "operator-recorded-version",
                                    "protocol_hardware_match": False,
                                    "physical_state_verification": "not yet performed"})
    result = p.collect()
    assert result["firmware_verification"] == "caller_attestation_provided"
    assert result["protocol_hardware_match"] == "caller_attestation_provided"
    assert result["physical_state_verification"] == "caller_attestation_provided"
    assert result["verification_context"]["protocol_hardware_match"] is False
    assert result["state_verified"] is False
    assert result["ready_for_actuation"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1., True, 10**1000])
def test_invalid_freshness_configuration_rejected(value):
    with pytest.raises(ValueError):
        probe(max_feedback_age_s=value)


def test_real_native_parser_is_loaded_without_control_interface_or_hardware_package():
    p = probe()
    module = importlib.import_module(type(p.parser).__module__)
    assert Path(module.__file__).resolve() == SDK / "piper_sdk/protocol/protocol_v2/piper_protocol_v2.py"
    assert type(p.parser).__name__ == "C_PiperParserV2"
    namespace = type(p.parser).__module__.split(".")[0]
    assert not any(name.startswith(namespace + ".interface") or name.startswith(namespace + ".hardware_port") for name in sys.modules)
    assert p.collect()["upstream"]["commit"] == "081e7c588e5b79eeaefa67a0469bcc701c81014f"


def test_three_fresh_pairs_use_official_signed_precision_and_units():
    clock = Clock()
    p = probe(clock)
    all_feedback(p, clock)
    result = p.collect()
    assert result["feedback_complete"]
    assert result["joint_feedback_complete"]
    assert result["joint_positions_rad"] == pytest.approx([math.pi/2, -math.pi/2, math.pi/180, -math.pi/180,
                                                           math.pi/180000, -math.pi/180000])
    assert result["gripper_measured_m"] == pytest.approx(.0245)
    assert result["gripper"]["raw_micrometres"] == 24500
    assert result["gripper"]["raw_effort_milli_nm"] == -123
    assert result["state_joint_units"] == "rad"
    assert result["state_gripper_units"] == "m"
    assert result["state_verified"] is False
    assert result["ready_for_actuation"] is False
    assert result["firmware_verification"] == "unknown"
    assert result["protocol_hardware_match"] == "unknown"
    json.dumps(result, allow_nan=False)


def test_fresh_zero_joint_feedback_is_legitimate_data_not_a_fault():
    clock = Clock()
    p = probe(clock)
    all_feedback(p, clock, zero=True)
    result = p.collect()
    assert result["joint_positions_rad"] == [0.] * 6
    assert result["feedback_complete"]
    assert result["fault_indication_present"] is False
    assert result["state_verified"] is False


def test_partial_pair_does_not_fill_other_joints_from_native_defaults():
    p = probe()
    p.update(joint_message(0x2A5, 1000, 2000))
    result = p.collect()
    assert result["joint_positions_rad"] is None
    assert result["joint_feedback_complete"] is False
    assert result["missing_joint_pair_ids"] == ["0x2a6", "0x2a7"]
    pair = result["joint_pairs"]["0x2a5"]
    assert pair["raw_millidegrees"] == [1000, 2000]
    assert pair["received_at_s"] == 100.
    assert pair["data_hex"] == "000003e8000007d0"


def test_each_pair_has_own_receive_monotonic_time_and_stale_pair_blocks_q6():
    clock = Clock()
    p = probe(clock)
    p.update(joint_message(0x2A5), received_at_s=100.)
    clock.now = 100.2
    p.update(joint_message(0x2A6), received_at_s=100.2)
    clock.now = 100.3
    p.update(joint_message(0x2A7), received_at_s=100.3)
    result = p.collect()
    assert result["joint_positions_rad"] is None
    assert result["stale_joint_pair_ids"] == ["0x2a5"]
    assert result["joint_pairs"]["0x2a6"]["received_at_s"] == 100.2
    p.update(joint_message(0x2A5), received_at_s=100.3)
    assert p.collect()["joint_feedback_complete"]


def test_old_can_wall_timestamp_cannot_make_stale_receive_evidence_fresh():
    clock = Clock()
    p = probe(clock)
    all_feedback(p, clock)
    clock.now += .3
    result = p.collect()
    assert result["joint_positions_rad"] is None
    assert result["gripper_measured_m"] is None
    assert result["feedback_complete"] is False
    assert result["arm_status"] is None


def test_native_status_and_native_fault_bit_properties_are_preserved():
    p = probe()
    p.update(message(0x2A1, bytes([1, 1, 1, 0, 1, 7, 1, 2])))
    result = p.collect()
    status = result["arm_status"]
    assert status["arm_status_name"] == "EMERGENCY_STOP"
    assert status["err_code"] == 0x0102
    assert status["err_status"]["joint_1_angle_limit"] is True
    assert status["err_status"]["communication_status_joint_2"] is True
    assert result["fault_indication_present"]
    assert result["status"] == "fault_reported"


def test_native_gripper_fault_flags_are_not_reimplemented():
    p = probe()
    p.update(message(0x2A8, struct.pack(">ihBB", 0, 0, 0x41, 0)))
    result = p.collect()
    assert result["gripper"]["foc_status"]["voltage_too_low"] is True
    assert result["gripper"]["foc_status"]["driver_enable_status"] is True
    assert result["fault_indication_present"]


def test_commands_are_ignored_even_if_native_parser_can_decode_them():
    p = probe()
    result = p.update(message(0x155, bytes(8)))
    assert result["status"] == "ignored_non_feedback"
    assert p.collect()["joint_positions_rad"] is None


@pytest.mark.parametrize("options", [{"is_extended_id": True}, {"is_error_frame": True},
                                     {"is_remote_frame": True}, {"is_rx": False}, {"is_fd": True},
                                     {"timestamp": float("nan")}])
def test_invalid_frame_kind_or_timestamp_cannot_refresh_feedback(options):
    p = probe()
    with pytest.raises(ValueError):
        p.update(message(0x2A5, bytes(8), **options))
    assert not p.collect()["joint_feedback_complete"]


@pytest.mark.parametrize("payload", [b"", bytes(7), bytes(9)])
def test_non_eight_byte_feedback_rejected_before_native_decode(payload):
    with pytest.raises(ValueError, match="8"):
        probe().update(message(0x2A5, payload))


@pytest.mark.parametrize("received_at", [float("nan"), float("inf"), -1., 101.])
def test_invalid_receive_monotonic_time_rejected(received_at):
    with pytest.raises(ValueError):
        probe().update(joint_message(0x2A5), received_at_s=received_at)


def test_out_of_order_pair_cannot_overwrite_fresher_measurement():
    p = probe()
    p.update(joint_message(0x2A5, 1000, 2000), received_at_s=100.)
    with pytest.raises(ValueError, match="order"):
        p.update(joint_message(0x2A5, 3000, 4000), received_at_s=99.9)
    assert p.collect()["joint_pairs"]["0x2a5"]["raw_millidegrees"] == [1000, 2000]


def test_source_lf_canonical_hash_allows_checkout_newline_conversion(tmp_path):
    root = tmp_path / "sdk"
    for path in (SDK / "piper_sdk").rglob("*.py"):
        target = root / path.relative_to(SDK)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    result = api().PassivePiperFeedback(root, max_feedback_age_s=.25, clock=Clock()).collect()
    assert result["upstream"]["source_hash_policy"] == "sha256_per_file_then_aggregate_canonical_lf"


def test_substantive_native_source_edit_rejected_before_import(tmp_path):
    root = tmp_path / "sdk"
    shutil.copytree(SDK / "piper_sdk", root / "piper_sdk")
    parser = root / "piper_sdk/protocol/protocol_v2/piper_protocol_v2.py"
    parser.write_bytes(parser.read_bytes() + b"\nraise RuntimeError('must never execute')\n")
    with pytest.raises(ValueError, match="hash"):
        api().PassivePiperFeedback(root, max_feedback_age_s=.25, clock=Clock())


def test_receive_only_collector_uses_real_frames_without_any_send_callback():
    clock = Clock()
    p = probe(clock)
    incoming = [joint_message(0x2A5), joint_message(0x2A6), joint_message(0x2A7)]
    def synthetic_receive_only(timeout):
        clock.now += .01 if incoming else timeout
        return incoming.pop(0) if incoming else None
    result = api().collect_feedback(synthetic_receive_only, probe=p, duration_s=.1)
    assert result["joint_feedback_complete"]
    assert result["transport_api_mode"] == "receive_only"
    assert result["api_transmit_calls"] == 0
    assert "send" not in inspect.signature(api().collect_feedback).parameters


def test_actual_can_receive_error_returns_failure_with_last_received_evidence():
    clock = Clock()
    p = probe(clock)
    all_feedback(p, clock)
    def synthetic_failed_receive(timeout):
        clock.now += .01
        raise can.CanOperationError("synthetic RX device disconnected")
    result = api().collect_feedback(synthetic_failed_receive, probe=p, duration_s=.1)
    assert result["status"] == "transport_error"
    assert result["feedback_complete"] is False
    assert result["joint_feedback_complete"] is False
    assert result["ready_for_actuation"] is False
    assert result["state"] is None
    assert result["joint_positions_rad"] is None
    assert result["transport_error"]["type"] == "CanOperationError"
    assert result["transport_error"]["phase"] == "receive"
    assert "disconnected" in result["transport_error"]["message"]
    assert result["last_received_evidence"]["feedback_complete"] is True
    assert result["last_received_evidence"]["joint_positions_rad"][0] == pytest.approx(math.pi/2)
    assert result["joint_pairs"]["0x2a5"]["received_at_s"] == 100.
    json.dumps(result, allow_nan=False)


def test_programming_error_from_receive_is_not_swallowed():
    def synthetic_bug(timeout):
        raise RuntimeError("caller bug")
    with pytest.raises(RuntimeError, match="caller bug"):
        api().collect_feedback(synthetic_bug, probe=probe(), duration_s=.1)


def test_cli_actual_can_initialization_error_writes_explicit_failure_without_bus(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples/probe_piper.py"
    spec = importlib.util.spec_from_file_location("synthetic_probe_cli_test", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "failure.json"
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", [str(path), "--channel", "synthetic0", "--duration-s", ".1",
                                         "--max-feedback-age-s", ".25", "--sdk-source", str(SDK),
                                         "--output", str(output)])
    def synthetic_failed_open(**kwargs):
        raise can.CanInitializationError("synthetic CAN initialization failure")
    monkeypatch.setattr(can, "Bus", synthetic_failed_open)
    cli.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "transport_error"
    assert result["feedback_complete"] is False
    assert result["ready_for_actuation"] is False
    assert result["transport_error"]["type"] == "CanInitializationError"
    assert result["api_transmit_calls"] == 0


def test_probe_source_and_cli_have_no_transmit_or_sdk_interface_calls():
    module = api()
    paths = [Path(module.__file__), Path(__file__).resolve().parents[1] / "examples/probe_piper.py"]
    for path in paths:
        assert path.is_file()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        called = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                  for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, (ast.Attribute, ast.Name))}
        assert not called.intersection({"send", "send_periodic", "C_PiperInterface", "C_PiperInterface_V2",
                                        "ConnectPort", "EnablePiper", "MotionCtrl_1", "MotionCtrl_2",
                                        "JointCtrl", "GripperCtrl", "EncodeMessage"})
