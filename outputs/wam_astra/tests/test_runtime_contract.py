"""Runtime instruction authority is explicit and never a numeric robot action."""
from copy import deepcopy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'work/wam_demo_library/vendor'))
from wam_astra.review_contract import model_schema,validate_review_reply

CONTROL=dict(schema_version=1,request_id='request-a',session_id='session-a',generation=3)
FRAMES={'now':dict(camera='uav',time_s=2.),'later':dict(camera='uav',time_s=3.)}


def reply():
    return dict(name='review_action_stream',arguments=dict(schema_version=1,case_id='case-a',decision_time_s=2.,
        assessment=dict(status='uncertain',confidence=.6,evidence=[dict(frame_id='now',camera='uav',time_s=2.,
                        observation='示例测试图像，不能证明真实抓取。')],limitations=['未进行物理验收。']),
        correction=dict(strategy='先获得无人机观察再重新对齐。',steps=[dict(kind='observe',description='获取高位观察。',
                        evidence_frame_ids=['now'])],joint_proposal=None,not_executable_reason='不提供数值关节控制。'),
        runtime_control=dict(request_id='request-a',session_id='session-a',generation=3,directive='replan_policy',
                        policy_instruction='先改善无人机视野，再核对盒内方块与夹爪的关系。',evidence_frame_ids=['now'])))


def validate(value=None,control=None,**kw):
    return validate_review_reply(reply() if value is None else value,case_id='case-a',decision_time_s=2.,
        available_frames=FRAMES,runtime_control=CONTROL if control is None else control,**kw)


def test_runtime_contract_is_opt_in_and_keeps_legacy_schema_unchanged():
    assert 'runtime_control' not in model_schema()['properties']['arguments']['properties']
    schema=model_schema(runtime_control=True)
    assert 'runtime_control' in schema['properties']['arguments']['required']
    assert 'runtime_control' not in model_schema()['properties']['arguments']['properties']


def test_strict_provider_schema_does_not_use_rejected_unique_items_keyword():
    assert 'uniqueItems' not in json.dumps(model_schema(runtime_control=True))


def test_duplicate_runtime_evidence_is_rejected_locally():
    value=reply();value['arguments']['runtime_control']['evidence_frame_ids']=['now','now']
    with pytest.raises(ValueError):validate(value)


def test_explicit_fresh_instruction_preserved_without_numeric_execution_authority():
    result=validate()
    assert result['runtime_policy_directive']==reply()['arguments']['runtime_control']
    assert result['executable'] is False and result['task_outcome']=='unknown'


@pytest.mark.parametrize('field,value',[('request_id','old'),('session_id','other'),('generation',2),('generation',True)])
def test_runtime_identity_cannot_reauthorize_old_result(field,value):
    value_reply=reply();value_reply['arguments']['runtime_control'][field]=value
    with pytest.raises(ValueError): validate(value_reply)


def test_keep_holding_cannot_smuggle_a_policy_instruction():
    value=reply();value['arguments']['runtime_control']['directive']='keep_holding'
    with pytest.raises(ValueError,match='instruction'):validate(value)
    value['arguments']['runtime_control']['policy_instruction']=''
    assert validate(value)['runtime_policy_directive']['directive']=='keep_holding'


def test_runtime_must_use_causal_current_evidence():
    value=reply();value['arguments']['runtime_control']['evidence_frame_ids']=['later']
    with pytest.raises(ValueError,match='Future|future'):validate(value)
    with pytest.raises(ValueError,match='causal|retrospective'):validate(allow_future_evidence=True)


def test_runtime_never_uses_normalized_joint_proposal_as_h45_actions():
    value=reply();value['arguments']['correction']['joint_proposal']=dict(joint_units='rad',gripper_units='normalized',
        step_s=.1,actions=[[0.,0.,0.,0.,0.,0.,1.]])
    with pytest.raises(ValueError,match='joint|numeric'):validate(value)
