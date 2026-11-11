"""H45 checkpoint events and bound numeric suggestions; no robot commands."""
from copy import deepcopy
import importlib
import math

import numpy as np
import pytest


def api():
    return importlib.import_module('wam_astra.action_checkpoints')


def config():
    # Synthetic test thresholds, never a deployable calibration.
    return dict(rotation_start_rad=.1,rotation_stop_rate_rad_s=.02,
                rotation_target_tolerance_rad=.03,min_rotation_progress_rad=.08,
                rotation_settle_s=.2,min_settle_samples=3,max_feedback_age_s=.15,
                gripper_close_delta_m=.005)


def candidate():
    actions=np.zeros((45,14));actions[:,9]=.04
    return dict(observation_id='obs',generation=2,origin_at=10.,model_actions=actions.tolist())


def sample(stamp,yaw,rate,source=None):
    return dict(stamp=stamp,source_stamp=stamp if source is None else source,
                pose=[0.,0.,1.,yaw],velocity=[0.,0.,0.,rate])


def test_scans_entire_chunk_before_first_motion_and_bundles_simultaneous_nodes():
    gate=api().CheckpointMonitor(**config());c=candidate()
    c['model_actions'][44][13]=.2;c['model_actions'][32][9]=.02
    event=gate.prediction(c,gripper_m=.04)
    assert event['checkpoint_kinds']==['uav_rotation_start','gripper_pre_close']
    assert event['first_rows']==dict(uav_rotation_start=44,gripper_pre_close=32)
    assert event['candidate_sha256']==api().candidate_digest(c)
    assert gate.prediction(c,gripper_m=.04)['event_id']==event['event_id']


def test_idle_and_gripper_opening_do_not_call_astra():
    gate=api().CheckpointMonitor(**config())
    assert gate.prediction(candidate(),gripper_m=.02) is None


def test_rotation_end_needs_actual_motion_target_arrival_and_distinct_stable_feedback():
    gate=api().CheckpointMonitor(**config())
    gate.arm_rotation(sample(10.,3.1,0.),target_yaw=-3.0,rotation_id='turn')
    assert gate.feedback(sample(10.1,-3.0,.2),now_s=10.1) is None
    assert gate.feedback(sample(10.2,-3.0,0.),now_s=10.2) is None
    for _ in range(5):assert gate.feedback(sample(10.2,-3.0,0.),now_s=10.21) is None
    assert gate.feedback(sample(10.3,-3.0,0.),now_s=10.3) is None
    event=gate.feedback(sample(10.41,-3.0,0.),now_s=10.41)
    assert event['checkpoint_kinds']==['uav_rotation_end']
    assert gate.feedback(sample(10.5,-3.0,0.),now_s=10.5) is None


def test_pause_before_rotation_or_stall_does_not_count_as_rotation_finished():
    gate=api().CheckpointMonitor(**config())
    gate.arm_rotation(sample(10.,0.,0.),target_yaw=.5,rotation_id='turn')
    for t in (10.1,10.2,10.4):assert gate.feedback(sample(t,0.,0.),now_s=t) is None


@pytest.mark.parametrize('bad',[sample(9.,.5,0.),sample(10.1,.5,0.,source=0.),sample(11.,.5,0.)])
def test_stale_future_or_unproven_feedback_fails_closed(bad):
    gate=api().CheckpointMonitor(**config())
    gate.arm_rotation(sample(10.,0.,0.),target_yaw=.5,rotation_id='turn')
    with pytest.raises(ValueError):gate.feedback(bad,now_s=10.2)


def request():
    return api().make_request(dict(event_id='checkpoint',checkpoint_kinds=['gripper_pre_close']),candidate())


def answer():
    c=candidate();row=c['model_actions'][0][:];row[9]=.05
    return dict(checkpoint_id='checkpoint',candidate_sha256=api().candidate_digest(c),decision='invalid',
                reason='保持夹爪张开，先核对对准关系。',evidence_frame_ids=['now'],
                replacements=[dict(row=0,vector=row)],cannot_correct_reason='')


def test_invalid_action_returns_exact_14d_replacements_and_full_chunk_without_execution_authority():
    result=api().validate_answer(answer(),request())
    assert len(result['corrected_actions'])==45 and result['corrected_actions'][0][9]==.05
    assert result['corrected_actions'][1:]==candidate()['model_actions'][1:]
    assert result['executable'] is False


@pytest.mark.parametrize('mutate',[
    lambda a:a.update(candidate_sha256='0'*64),
    lambda a:a['replacements'].append(deepcopy(a['replacements'][0])),
    lambda a:a['replacements'][0].update(row=45),
    lambda a:a['replacements'][0].update(vector=[0.]*7),
    lambda a:a['replacements'][0]['vector'].__setitem__(0,float('nan')),
    lambda a:a.update(decision='valid'),
])
def test_wrong_identity_duplicate_rows_wrong_units_or_nonfinite_vectors_rejected(mutate):
    a=answer();mutate(a)
    with pytest.raises(ValueError):api().validate_answer(a,request())


def test_model_can_reject_unsupported_numeric_guess_instead_of_inventing_joint_targets():
    a=answer();a.update(replacements=[],cannot_correct_reason='缺少外参，无法求出可靠的关节修正。')
    r=api().validate_answer(a,request())
    assert r['decision']=='invalid' and r['corrected_actions'] is None


def test_checkpoint_contract_runs_through_actual_native_prompt_and_result_validator():
    from test_runtime_contract import reply,CONTROL,FRAMES
    from wam_astra.native_review_worker import review_prompt_contract
    from wam_astra.review_contract import validate_review_reply
    text,schema=review_prompt_contract(dict(runtime_control=CONTROL,action_checkpoint=request()))
    assert '14' in text and 'action_checkpoint' in schema['properties']['arguments']['required']
    value=reply();value['arguments']['action_checkpoint']=answer()
    out=validate_review_reply(value,case_id='case-a',decision_time_s=2.,available_frames=FRAMES,
        runtime_control=CONTROL,action_checkpoint=request())
    assert out['checkpoint_action_review']['corrected_actions'][0][9]==.05
    assert out['executable'] is False


def test_checkpoint_cannot_use_only_old_demonstrations_to_authorize_validity():
    from test_runtime_contract import reply
    from wam_astra.review_contract import validate_review_reply
    frames=dict(old=dict(camera='uav',time_s=1.,evidence_role='historical_reference',source_episode_id='old',
        availability_attested=True,reference_manifest_sha256='a'*64))
    value=reply();value['arguments'].pop('runtime_control');value['arguments']['assessment'].update(evidence=[])
    value['arguments']['correction']['steps'][0]['evidence_frame_ids']=[]
    a=answer();a['evidence_frame_ids']=['old'];value['arguments']['action_checkpoint']=a
    with pytest.raises(ValueError,match='current'):
        validate_review_reply(value,case_id='case-a',decision_time_s=2.,available_frames=frames,
            action_checkpoint=request(),current_episode_id='live')


def test_rotation_can_return_to_start_heading_and_still_finish_after_real_motion():
    gate=api().CheckpointMonitor(**config())
    actions=candidate()['model_actions'];actions[12][13]=.3
    gate.track_commit(sample(10.,0.,0.),actions,plan_id='roundtrip')
    assert gate.feedback(sample(10.1,.2,.5),now_s=10.1) is None
    for t in (10.2,10.3):assert gate.feedback(sample(t,0.,0.),now_s=t) is None
    assert gate.feedback(sample(10.41,0.,0.),now_s=10.41)['checkpoint_kinds']==['uav_rotation_end']


def test_second_rotation_is_rearmed_after_finished_feedback():
    gate=api().CheckpointMonitor(**config());gate.arm_rotation(sample(10.,0.,0.),target_yaw=.2,rotation_id='r')
    for t in (10.1,10.2,10.31):gate.feedback(sample(t,.2,0.),now_s=t)
    c=candidate();c['model_actions'][0][13]=-.2
    assert gate.prediction(c,gripper_m=.02)['checkpoint_kinds']==['uav_rotation_start']


def test_same_source_republication_cannot_keep_settling_window_alive():
    gate=api().CheckpointMonitor(**config());gate.arm_rotation(sample(10.,0.,0.),target_yaw=.2,rotation_id='r')
    assert gate.feedback(sample(10.1,.2,0.),now_s=10.1) is None
    assert gate.feedback(sample(10.2,.2,0.,source=10.1),now_s=10.2) is None
    with pytest.raises(ValueError,match='Frozen'):
        gate.feedback(sample(10.3,.2,0.,source=10.1),now_s=10.3)


def test_checkpoint_hooks_arm_only_from_paired_commits_and_use_measured_feedback():
    from wam_astra.checkpoint_hooks import CheckpointHooks
    from types import SimpleNamespace
    plane=dict(stamp=10.,state_ros_stamp=10.,state4=[0.]*4,pose=dict(x=0.,y=0.,z=1.,yaw=0.))
    device=SimpleNamespace(monitor_status=lambda:dict(p450=dict(monotonic=10.4,feedback=sample(10.4,.2,0.))))
    hooks=CheckpointHooks(device,dict(calibration_verified=True,**config()))
    state=[0.]*17;state[16]=.04
    packet=dict(observation_id='obs',stamp=10.,state=state,p450_packet=plane)
    c=candidate();c['model_actions'][-1][13]=.2
    event=hooks(packet,np.array(c['model_actions']),2)
    assert event['checkpoint_kinds']==['uav_rotation_start'] and hooks.monitor.rotation is None
    c['checkpoint_anchor']=dict(p450_packet=plane)
    hooks.committed(c)
    assert hooks.monitor.rotation['target']==pytest.approx(.2)
    assert hooks.monitor_source().get('event') is None


def test_checkpoint_hooks_refuse_unverified_thresholds():
    from wam_astra.checkpoint_hooks import CheckpointHooks
    with pytest.raises(ValueError,match='Verified'):CheckpointHooks(None,dict(calibration_verified=False,**config()))


def test_sparse_feedback_cannot_claim_a_continuous_settling_window():
    gate=api().CheckpointMonitor(**config());gate.arm_rotation(sample(10.,0.,0.),target_yaw=.2,rotation_id='r')
    assert gate.feedback(sample(10.1,.2,0.),now_s=10.1) is None
    assert gate.feedback(sample(10.4,.2,0.),now_s=10.4) is None
    assert gate.feedback(sample(10.5,.2,0.),now_s=10.5) is None


def test_boolean_schema_version_is_not_integer_protocol_identity():
    req=request();req['schema_version']=True
    with pytest.raises(ValueError):api().validate_request(req)
