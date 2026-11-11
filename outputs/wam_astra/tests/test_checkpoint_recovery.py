"""Synthetic bounded checkpoint recovery, no motor transport."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from test_action_checkpoints import config,candidate

def setup():
    from wam_astra.checkpoint_hooks import CheckpointHooks
    now=[10.]
    cfg=dict(calibration_verified=True,**config(),recovery=dict(lifetime_s=1.,
        action_tolerance=[.001]*14,state_tolerance=[.001]*17,pose_tolerance=[.001]*4))
    h=CheckpointHooks(SimpleNamespace(),cfg,clock=lambda:now[0])
    p=dict(observation_id='old',stamp=1.,state=[0.]*16+[.04],p450_packet=dict(
        stamp=1.,state_ros_stamp=1.,state4=[0.]*4,pose=dict(x=0.,y=0.,z=1.,yaw=0.)))
    c=candidate();c['model_actions'][-1][13]=.2
    c['checkpoint_anchor']=dict(state=p['state'],p450_packet=p['p450_packet'])
    e=h(p,np.array(c['model_actions']),2)
    from wam_astra.action_checkpoints import candidate_digest
    a=dict(checkpoint_id=e['event_id'],candidate_sha256=candidate_digest(c),decision='valid',
        reason='合成测试',evidence_frame_ids=['now'],replacements=[],cannot_correct_reason='')
    h.reviewed(dict(checkpoint_action_review=a),candidate=c,event=e,generation=3)
    return h,p,c,e,now

def test_review_permit_allows_only_bounded_fresh_generation_and_is_consumed_on_commit():
    h,p,c,e,now=setup();p=deepcopy(p);p.update(observation_id='fresh',stamp=2.)
    assert h(p,np.array(c['model_actions']),3) is None
    committed=dict(c,observation_id='fresh',origin_at=2.,generation=3)
    h.committed(committed)
    assert h.recovery_permit is None

@pytest.mark.parametrize('change',['expired','generation','state','pose','actions'])
def test_expired_foreign_or_changed_candidates_cannot_use_old_review(change):
    h,p,c,e,now=setup();p=deepcopy(p);p.update(observation_id='fresh',stamp=2.)
    a=np.array(c['model_actions']);g=3
    if change=='expired':now[0]=11.
    if change=='generation':g=4
    if change=='state':p['state'][10]+=.1
    if change=='pose':p['p450_packet']['pose']['x']+=.1
    if change=='actions':a[0,3]+=.1
    assert h(p,a,g) is not None

def test_uncertain_review_never_produces_permission():
    h,p,c,e,now=setup();h.recovery_permit=None
    h.reviewed(dict(checkpoint_action_review=dict(decision='uncertain')),candidate=c,event=e,generation=3)
    assert h.recovery_permit is None

def test_permission_expiring_during_prepare_cannot_silently_be_consumed():
    h,p,c,e,now=setup();assert h(p,np.array(c['model_actions']),3) is None
    now[0]=11.
    with pytest.raises(RuntimeError,match='expired'):h.check()

def test_explicit_invalid_numeric_patch_is_the_target_not_the_rejected_actions():
    h,p,c,e,now=setup()
    from wam_astra.action_checkpoints import candidate_digest
    row=c['model_actions'][-1][:];row[13]=.3
    a=dict(checkpoint_id=e['event_id'],candidate_sha256=candidate_digest(c),decision='invalid',
        reason='合成修正',evidence_frame_ids=['now'],replacements=[dict(row=44,vector=row)],cannot_correct_reason='')
    h.reviewed(dict(checkpoint_action_review=a),candidate=c,event=e,generation=3)
    assert h(p,np.array(c['model_actions']),3) is not None
    actions=np.array(c['model_actions']);actions[-1]=row
    assert h(p,actions,3) is None


@pytest.mark.parametrize('numeric_correction',[False,True])
def test_direct_recovery_preserves_world_targets_and_does_not_replay_committed(monkeypatch,numeric_correction):
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'h45_takeover/staged/client'))
    device_type=type('Device',(),{'__module__':'real_robot.review_device'})
    h,p,c,e,now=setup();h.device=device_type()
    if numeric_correction:
        from wam_astra.action_checkpoints import candidate_digest
        replacement=c['model_actions'][0][:];replacement[9]=.06
        h.reviewed(dict(checkpoint_action_review=dict(checkpoint_id=e['event_id'],
            candidate_sha256=candidate_digest(c),decision='invalid',reason='保持张开',
            evidence_frame_ids=['now'],replacements=[dict(row=0,vector=replacement)],
            cannot_correct_reason='')),candidate=c,event=e,generation=3)
    # The default for unspecified/committed provenance must remain no replay.
    assert h.recovery_actions(p,3) is None
    h.recovery_permit['committed']=False
    p=deepcopy(p);p.update(observation_id='fresh',stamp=2.)
    p['state'][0]=.0005;p['state'][10]=.0005
    p['p450_packet']['pose']['x']=.0005
    actions=h.recovery_actions(p,3)
    np.testing.assert_allclose(actions[:,0],np.array(c['model_actions'])[:,0]-.0005)
    np.testing.assert_allclose(actions[:,3],np.array(c['model_actions'])[:,3]-.0005)
    np.testing.assert_allclose(actions[:,10],np.array(c['model_actions'])[:,10]-.0005)
    np.testing.assert_allclose(actions[:,13],np.array(c['model_actions'])[:,13])
    if numeric_correction:assert actions[0,9]==.06
    assert h(p,actions,3) is None
    h.committed(dict(observation_id='fresh',origin_at=2.,generation=3,
        model_actions=actions.tolist(),checkpoint_anchor=dict(state=p['state'],p450_packet=p['p450_packet'])))
    assert h.recovery_permit is None
