"""Strict offline reviewer-envelope validation; no model or transport calls."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'work/wam_demo_library/vendor'))


def api():
    assert importlib.util.find_spec('wam_astra.review_contract') is not None, 'Reviewer contract must exist'
    from wam_astra import review_contract
    return review_contract


FRAMES = {'front-1': {'camera': 'front', 'time_s': 1.},
          'wrist-2': {'camera': 'wrist', 'time_s': 2.},
          'future-front': {'camera': 'front', 'time_s': 3.}}


def reply():
    return {'name': 'review_action_stream', 'arguments': {
        'schema_version': 1, 'case_id': 'case-a', 'decision_time_s': 2.,
        'assessment': {'status': 'empty_grasp', 'confidence': .8,
            'evidence': [{'frame_id': 'front-1', 'camera': 'front', 'time_s': 1.,
                          'observation': '夹爪附近未见物体随动；接触状态仍不确定。'}],
            'limitations': ['画面不能确认接触力或硬件执行回执。']},
        'correction': {'strategy': '先保持，补充腕部观测，再核对抓取对齐。',
            'steps': [{'kind': 'observe', 'description': '补充腕部视角确认目标和夹爪关系。',
                       'evidence_frame_ids': ['front-1', 'wrist-2']}],
            'joint_proposal': None,
            'not_executable_reason': '缺少已验证的物理限位、速度约束和先前命令，不能下发控制。'}}}


def validate(value=None, **changes):
    args = {'case_id': 'case-a', 'decision_time_s': 2., 'available_frames': FRAMES}
    args.update(changes)
    return api().validate_review_reply(reply() if value is None else value, **args)


def test_strict_native_envelope_schema_and_guard_instructions():
    schema = api().model_schema()
    def inspect(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node['additionalProperties'] is False
                assert set(node['required']) == set(node['properties'])
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    inspect(schema)
    assert schema['properties']['name']['enum'] == ['review_action_stream']
    instructions = api().review_instructions()
    assert 'Piper' in instructions
    assert '单臂' in instructions
    assert 'ground truth' in instructions
    assert '未来' in instructions
    assert 'normalized' in instructions


def test_symbolic_correction_without_calibration_is_preserved_and_held():
    original = reply()
    result = validate(original)
    assert result['arguments'] == original['arguments']
    assert result['executable'] is False
    assert result['effective_decision'] == 'hold'
    assert result['task_outcome'] == 'unknown'
    assert result['assessment_is_ground_truth'] is False
    assert result['confidence_is_calibrated'] is False
    assert result['offline_only'] is True
    assert original == reply()


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(name='move_robot'),
    lambda x: x['arguments'].update(schema_version=True),
    lambda x: x['arguments'].update(case_id='foreign'),
    lambda x: x['arguments'].update(decision_time_s=1.99),
    lambda x: x['arguments']['assessment'].update(status='success'),
    lambda x: x['arguments']['assessment'].update(confidence=1.1),
    lambda x: x['arguments']['assessment'].update(confidence=float('nan')),
    lambda x: x['arguments']['correction'].update(extra='unsupported'),
    lambda x: x['arguments']['correction'].update(not_executable_reason=''),
])
def test_unsupported_contract_and_nonfinite_inputs_rejected(mutation):
    value = reply()
    mutation(value)
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize('fields', [
    {'frame_id': 'invented'}, {'camera': 'overhead'}, {'time_s': 1.01},
    {'frame_id': 'future-front', 'camera': 'front', 'time_s': 3.},
])
def test_visual_evidence_ids_camera_time_and_causality_are_bound(fields):
    value = reply()
    value['arguments']['assessment']['evidence'][0].update(fields)
    with pytest.raises(ValueError):
        validate(value)


def test_future_frame_cannot_be_sneaked_into_symbolic_steps():
    value = reply()
    value['arguments']['correction']['steps'][0]['evidence_frame_ids'] = ['future-front']
    with pytest.raises(ValueError, match='future|Future|causal'):
        validate(value)


def test_retrospective_future_evidence_is_labeled_and_never_executable():
    value = reply()
    value['arguments']['assessment']['evidence'][0].update(frame_id='future-front', time_s=3.)
    result = validate(value, allow_future_evidence=True)
    assert result['evidence_mode'] == 'retrospective'
    assert result['used_future_evidence'] is True
    assert result['executable'] is False


def numeric_reply():
    value = reply()
    value['arguments']['correction']['joint_proposal'] = {
        'joint_units': 'rad', 'gripper_units': 'normalized', 'step_s': .1,
        'actions': [[0.] * 7, [.01] * 6 + [0.], [.02] * 6 + [0.]]}
    return value


def limits():
    return {'verified': True, 'joint_units': 'rad', 'gripper_units': 'normalized',
            'joint_lower': [-1.] * 6, 'joint_upper': [1.] * 6,
            'max_velocity': [1.] * 6, 'max_acceleration': [4.] * 6,
            'previous_action': [0.] * 7, 'previous_velocity': [0.] * 6}


def test_uncalibrated_numeric_proposal_is_archived_but_held():
    value = numeric_reply()
    result = validate(value)
    assert result['arguments'] == value['arguments']
    assert result['executable'] is False
    assert result['effective_decision'] == 'hold'


def test_complete_verified_limits_allow_only_offline_proposal_validation():
    value = numeric_reply()
    value['arguments']['correction']['not_executable_reason'] = ''
    result = validate(value, physical_limits=limits())
    assert result['executable'] is True
    assert result['offline_only'] is True
    assert result['actuator_calls'] == 0
    assert result['task_outcome'] == 'unknown'


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(joint_units='m'),
    lambda x: x.update(gripper_units='m'),
    lambda x: x.update(step_s=0.),
    lambda x: x.update(actions=[[0.] * 6]),
    lambda x: x.update(actions=[[0.] * 6 + [.035]]),
    lambda x: x.update(actions=[[False] * 7]),
    lambda x: x.update(actions=[['0'] * 7]),
])
def test_metric_and_malformed_numeric_commands_are_rejected(mutation):
    value = numeric_reply()
    mutation(value['arguments']['correction']['joint_proposal'])
    with pytest.raises(ValueError):
        validate(value)


def test_joint_limit_violation_cannot_be_marked_executable():
    value = numeric_reply()
    value['arguments']['correction']['joint_proposal']['actions'][0][0] = 2.
    with pytest.raises(ValueError, match='bounds'):
        validate(value, physical_limits=limits())


def test_missing_or_unverified_physical_metadata_keeps_hold():
    for metadata in ({}, {'verified': False}, {**limits(), 'previous_velocity': None}):
        result = validate(numeric_reply(), physical_limits=metadata)
        assert result['executable'] is False
        assert result['effective_decision'] == 'hold'


def test_available_frame_metadata_itself_must_be_finite_and_unambiguous():
    with pytest.raises(ValueError):
        validate(available_frames={'front-1': {'camera': 'front', 'time_s': float('nan')}})


def test_model_non_executable_reason_is_not_overridden_by_physical_metadata():
    result = validate(numeric_reply(), physical_limits=limits())
    assert result['executable'] is False
    assert result['effective_decision'] == 'hold'


def test_retrospective_mode_is_never_executable_even_without_future_citations():
    result = validate(numeric_reply(), physical_limits=limits(), allow_future_evidence=True)
    assert result['used_future_evidence'] is False
    assert result['executable'] is False


def test_corrective_strategy_must_include_chinese_text():
    value = reply()
    value['arguments']['correction']['strategy'] = 'Observe then align'
    with pytest.raises(ValueError):
        validate(value)


@pytest.mark.parametrize('status', ['aligned', 'misaligned', 'empty_grasp', 'grasped'])
def test_categorical_visual_assessments_require_bound_image_evidence(status):
    value = reply()
    value['arguments']['assessment'].update(status=status, evidence=[])
    with pytest.raises(ValueError, match='evidence'):
        validate(value)


def test_uncertain_assessment_can_report_no_adequate_visual_evidence():
    value = reply()
    value['arguments']['assessment'].update(status='uncertain', evidence=[])
    assert validate(value)['arguments']['assessment']['status'] == 'uncertain'


def test_ground_start_scope_is_preserved_as_unverified_user_wording():
    instructions = api().review_instructions()
    assert '具体指机器人还是机械臂未独立确认' in instructions
    assert '这里地面起点指机械臂/机器人起点' not in instructions


def mixed_clock_case():
    value = reply()
    value['arguments']['decision_time_s'] = 5.
    value['arguments']['assessment']['evidence'] = [
        {'frame_id': 'live-front', 'camera': 'front', 'time_s': 4.8, 'observation': '当前夹爪和盒子可见。'},
        {'frame_id': 'ref-front', 'camera': 'front', 'time_s': 147., 'observation': '旧示范显示接近方向，不能证明当前抓取。'},
    ]
    value['arguments']['correction']['steps'][0]['evidence_frame_ids'] = ['ref-front']
    frames = {
        'live-front': {'camera': 'front', 'time_s': 4.8, 'evidence_role': 'current_observation', 'source_episode_id': 'live'},
        'ref-front': {'camera': 'front', 'time_s': 147., 'evidence_role': 'historical_reference',
                      'source_episode_id': 'old-demo', 'availability_attested': True, 'reference_manifest_sha256': 'a' * 64},
    }
    return value, frames


def validate_mixed(value, frames, **options):
    return validate(value, decision_time_s=5., available_frames=frames, current_episode_id='live', **options)


def test_attested_other_episode_reference_keeps_its_clock_without_becoming_future_evidence():
    value, frames = mixed_clock_case()
    result = validate_mixed(value, frames)
    assert result['historical_reference_ids'] == ['ref-front']
    assert result['used_future_evidence'] is False
    assert result['arguments'] == value['arguments']
    assert result['arguments']['assessment']['evidence'][1]['time_s'] == 147.


def test_same_episode_reference_cannot_bypass_current_causal_cutoff():
    value, frames = mixed_clock_case()
    frames['ref-front']['source_episode_id'] = 'live'
    with pytest.raises(ValueError, match='Future|future|causal'):
        validate_mixed(value, frames)


@pytest.mark.parametrize('field,value', [
    ('evidence_role', None), ('evidence_role', 'unrecognized-role'),
    ('source_episode_id', None), ('source_episode_id', ''),
    ('availability_attested', None), ('availability_attested', False), ('availability_attested', 1),
    ('reference_manifest_sha256', None), ('reference_manifest_sha256', 'not-a-hash'),
])
def test_partial_or_invalid_historical_attestation_is_rejected_before_bypass(field, value):
    reply_value, frames = mixed_clock_case()
    if value is None:
        frames['ref-front'].pop(field)
    else:
        frames['ref-front'][field] = value
    with pytest.raises(ValueError):
        validate_mixed(reply_value, frames)


def test_historical_reference_requires_explicit_current_episode_identity():
    value, frames = mixed_clock_case()
    with pytest.raises(ValueError, match='episode'):
        validate(value, decision_time_s=5., available_frames=frames)


@pytest.mark.parametrize('status', ['aligned', 'misaligned', 'empty_grasp', 'grasped'])
def test_historical_reference_alone_cannot_support_current_categorical_assessment(status):
    value, frames = mixed_clock_case()
    value['arguments']['assessment']['status'] = status
    value['arguments']['assessment']['evidence'] = value['arguments']['assessment']['evidence'][1:]
    with pytest.raises(ValueError, match='current|Current'):
        validate_mixed(value, frames)


def test_uncertain_reference_only_assessment_remains_explicitly_unknown():
    value, frames = mixed_clock_case()
    value['arguments']['assessment']['status'] = 'uncertain'
    value['arguments']['assessment']['evidence'] = value['arguments']['assessment']['evidence'][1:]
    assert validate_mixed(value, frames)['task_outcome'] == 'unknown'


@pytest.mark.parametrize('update', [{'time_s': 146.}, {'camera': 'wrist'}])
def test_reference_evidence_still_requires_exact_registered_camera_and_timestamp(update):
    value, frames = mixed_clock_case()
    value['arguments']['assessment']['evidence'][1].update(update)
    with pytest.raises(ValueError, match='camera|time'):
        validate_mixed(value, frames)


def test_future_current_image_still_rejected_among_valid_old_references():
    value, frames = mixed_clock_case()
    frames['live-front']['time_s'] = 6.
    value['arguments']['assessment']['evidence'][0]['time_s'] = 6.
    with pytest.raises(ValueError, match='Future|future|causal'):
        validate_mixed(value, frames)
    retrospective = validate_mixed(value, frames, allow_future_evidence=True)
    assert retrospective['used_future_evidence'] is True
    assert retrospective['historical_reference_ids'] == ['ref-front']


def test_legacy_summary_keeps_exact_shape_without_new_reference_metadata():
    result = validate()
    assert 'historical_reference_ids' not in result
    assert validate(current_episode_id='live') == result


def test_step_only_historical_reference_is_registered_without_future_flag():
    value, frames = mixed_clock_case()
    value['arguments']['assessment']['evidence'] = value['arguments']['assessment']['evidence'][:1]
    result = validate_mixed(value, frames)
    assert result['historical_reference_ids'] == ['ref-front']
    assert result['used_future_evidence'] is False
    frames['ref-front']['source_episode_id'] = 'live'
    with pytest.raises(ValueError, match='Future|future|causal'):
        validate_mixed(value, frames)


def test_unused_historical_metadata_does_not_change_summary_shape():
    value, frames = mixed_clock_case()
    value['arguments']['assessment']['evidence'] = value['arguments']['assessment']['evidence'][:1]
    value['arguments']['correction']['steps'][0]['evidence_frame_ids'] = ['live-front']
    assert 'historical_reference_ids' not in validate_mixed(value, frames)


@pytest.mark.parametrize('run_name', ['candidate_2_causal_retry1', 'candidate_2_retrospective_small'])
def test_saved_real_legacy_validation_is_byte_value_compatible_without_model_calls(run_name):
    run = Path(__file__).resolve().parents[1] / 'astra_runs' / run_name
    if not (run / 'validation.json').is_file():
        pytest.skip('Saved real legacy archive absent; no inference is fabricated')
    import json
    request = json.loads((run / 'request.json').read_text(encoding='utf-8'))
    original = json.loads((run / 'validation.json').read_text(encoding='utf-8'))
    wire = json.loads((run / 'reply.json').read_text(encoding='utf-8'))
    current = api().validate_review_reply(wire, case_id=request['case_id'],
        decision_time_s=request['decision_time_s'], available_frames=request['available_frames'],
        allow_future_evidence=request['phase'] == 'retrospective_audit', physical_limits=None)
    assert current == original
