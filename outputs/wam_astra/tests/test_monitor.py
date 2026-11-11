"""Adapter tests execute pinned upstream code; no algorithm doubles or weights."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
ROBOICL = WORKSPACE / 'work/upstream/RoboICL'
VLA = WORKSPACE / 'work/upstream/vla-corrector'
VENDOR = WORKSPACE / 'work/wam_demo_library/vendor'
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))


def api():
    assert importlib.util.find_spec('wam_astra.monitor') is not None, 'native monitor adapter is missing'
    return importlib.import_module('wam_astra.monitor')


def tracking(**kwargs):
    return api().NativeTrackingMonitor(ROBOICL, **kwargs)


def joints(value):
    return {'left_arm_joint_state': [value] * 6}


def test_original_guard_triggers_after_two_large_joint_residuals():
    m = tracking()
    first = m.update(joints(1.), joints(0.), observation_id='first')
    second = m.update(joints(1.), joints(0.), observation_id='second')
    assert not first['triggered']
    assert second['triggered']
    assert second['status'] == 'triggered'
    assert second['receipt']['left']['max_joint_error_rad'] == 1.
    assert second['count'] == second['required'] == 2


def test_native_class_file_and_behavior_are_used_without_reimplementation():
    m = tracking()
    assert Path(sys.modules[type(m.guard).__module__].__file__).resolve() == (ROBOICL / 'roboicl/policy/action_feedback.py').resolve()
    direct = type(m.guard)()
    for target, observed in [(joints(1.), joints(0.)), (joints(0.), joints(0.)),
                             (joints(2.), joints(0.)), (joints(2.), joints(0.)),
                             (joints(2.), joints(0.))]:
        result = m.update(target, observed)
        assert result['triggered'] == direct.observe(result['receipt'])
        assert result['count'] == direct.count


def test_event_monitor_alias_is_native_tracking_monitor():
    assert api().EventMonitor is api().NativeTrackingMonitor


def test_tracking_evidence_identifies_source_and_capability_limit():
    result = tracking().update(joints(0.), joints(0.))
    evidence = result['evidence']
    assert evidence['model_identity'] == 'RoboICL.TrackingGuard'
    assert evidence['commit'] == '629290545af0d139a955a6555dedd651a60f3db9'
    assert evidence['source_sha256'] == '0dead10a31340b114a8ad449157934777482d5efcfd86ce5398efb8c41dae474'
    assert evidence['capability'] == 'proprioceptive_tracking_only'
    assert evidence['source_url'].startswith('https://github.com/Mosi-AI/RoboICL/')
    assert result['status'] == 'within_tracking_limits'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_native_receipt_rejects_nonfinite_joints_and_interrupts_count(value):
    m = tracking()
    m.update(joints(1.), joints(0.))
    invalid = m.update(joints(value), joints(0.))
    assert invalid['status'] == 'unavailable'
    assert invalid['reason'] == 'Invalid joints in tracking receipt'
    assert not invalid['triggered']
    assert not m.update(joints(1.), joints(0.))['triggered']


def test_unmeasured_gripper_command_is_not_tracking_evidence():
    result = tracking().update({'left_ee_joint_state': [1.]}, {'left_ee_joint_state': [0.]})
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'no_comparable_arm_measurements'


def test_missing_measurements_do_not_count_as_healthy_or_consecutive():
    m = tracking()
    m.update(joints(1.), joints(0.))
    assert m.update(joints(1.), {})['status'] == 'unavailable'
    assert not m.update(joints(1.), joints(0.))['triggered']


def test_reset_uses_original_guard_and_clears_consecutive_count():
    m = tracking()
    m.update(joints(1.), joints(0.))
    m.reset()
    assert not m.update(joints(1.), joints(0.))['triggered']


def test_native_pose_receipt_preserves_position_and_orientation_units():
    target = {'right_ee_pose': [0., 0., 0., 1., 0., 0., 0.]}
    actual = {'right_ee_pose': [0.1, 0., 0., 1., 0., 0., 0.]}
    result = tracking().update(target, actual)
    assert result['receipt']['right']['position_error_m'] == pytest.approx(.1)
    assert result['receipt']['right']['orientation_error_rad'] == 0.


def test_native_zero_quaternion_rejection_is_preserved():
    pose = {'left_ee_pose': [0.] * 7}
    result = tracking().update(pose, pose)
    assert result['reason'] == 'Zero quaternion in tracking receipt'


def test_native_threshold_parameters_are_forwarded():
    module = api()
    m = module.NativeTrackingMonitor(ROBOICL, config=module.MonitorConfig(joint_rad=2., consecutive=1))
    assert not m.update(joints(1.), joints(0.))['triggered']
    assert m.update(joints(3.), joints(0.))['triggered']


@pytest.mark.parametrize('kwargs', [
    {'position_m': float('nan')}, {'orientation_rad': float('inf')},
    {'joint_rad': 0.}, {'joint_rad': -1.}, {'consecutive': 0},
    {'consecutive': 1.5}, {'consecutive': True},
])
def test_invalid_tracking_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        api().MonitorConfig(**kwargs)


def test_missing_dependency_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='RoboICL'):
        api().NativeTrackingMonitor(tmp_path)


def test_tampered_upstream_source_is_rejected_before_import(tmp_path):
    path = tmp_path / 'roboicl/policy/action_feedback.py'
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must not execute')", encoding='utf-8')
    with pytest.raises(ValueError, match='source hash'):
        api().NativeTrackingMonitor(tmp_path)


def vision(**kwargs):
    return api().NativeVisionMonitor(VLA, model_identity='test/synthetic-feature-pairs', **kwargs)


def test_vision_requires_explicit_prediction_model_identity():
    with pytest.raises(ValueError, match='model_identity'):
        api().NativeVisionMonitor(VLA, model_identity='')


@pytest.mark.parametrize('kwargs', [
    {'k_mad': float('nan')}, {'k_mad': float('inf')}, {'k_mad': 0.},
    {'k_mad': -1.}, {'k_mad': True}, {'k_mad': 'bad'},
    {'window_size': 9}, {'window_size': 0}, {'window_size': 10.5}, {'window_size': True},
    {'cooldown_steps': -1}, {'cooldown_steps': 1.5}, {'cooldown_steps': True},
])
def test_vision_rejects_configuration_that_disables_native_detection(kwargs):
    with pytest.raises(ValueError):
        vision(**kwargs)


def test_vision_accepts_minimum_warmup_window_and_zero_cooldown():
    m = vision(window_size=10, cooldown_steps=0)
    assert m.breaker.window_size == 10
    assert m.breaker.cooldown_steps == 0


@pytest.mark.parametrize('predicted,observed', [(None, [1., 0.]), ([1., 0.], None)])
def test_vision_rejects_missing_actual_feature_inputs(predicted, observed):
    with pytest.raises(ValueError, match='features'):
        vision().update(predicted, observed)


def test_original_circuit_breaker_warmup_then_deliberate_feature_mismatch():
    m = vision()
    for i in range(10):
        result = m.update([1., 0.], [1., 0.], observation_id=str(i))
        assert not result['triggered']
    assert result['threshold'] == 0.
    anomaly = m.update([1., 0.], [-1., 0.])
    assert anomaly['triggered']
    assert anomaly['score'] == pytest.approx(2.)
    assert anomaly['evidence']['prediction_model_identity'] == 'test/synthetic-feature-pairs'
    assert anomaly['evidence']['model_identity'] == 'VLA-Corrector.CircuitBreaker'


def test_vision_delegates_native_state_and_preserves_zero_vector_behavior():
    m = vision()
    assert Path(sys.modules[type(m.breaker).__module__].__file__).resolve() == (VLA / 'src/siglip_dynamics/inference/circuit_breaker.py').resolve()
    result = m.update([0., 0.], [0., 0.])
    assert result['score'] == 1.
    assert result['threshold'] is None
    assert result['reason'] == 'native_threshold_warmup'
    json.dumps(result, allow_nan=False)
    m.reset()
    assert not m.breaker.error_history


def test_vision_nonfinite_features_rejected_before_native_history_mutation():
    m = vision()
    with pytest.raises(ValueError, match='finite'):
        m.update([np.nan], [0.])
    assert not m.breaker.error_history


def test_vision_shape_mismatch_rejected():
    with pytest.raises(ValueError, match='shape'):
        vision().update([1.], [1., 0.])


def test_native_breaker_triggered_state_is_preserved_without_cooldown_patch():
    m = vision()
    for _ in range(10):
        m.update([1., 0.], [1., 0.])
    assert m.update([1., 0.], [-1., 0.])['triggered']
    assert m.breaker.state.value == 'triggered'
    assert m.update([1., 0.], [-1., 0.])['triggered']
