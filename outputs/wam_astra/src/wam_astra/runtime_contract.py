"""Explicit advisory policy instructions, separate from numeric control authority."""
from copy import deepcopy


def output_schema():
    fields=dict(request_id={'type':'string','minLength':1},session_id={'type':'string','minLength':1},
        generation={'type':'integer','minimum':0},directive={'type':'string','enum':['keep_holding','replan_policy']},
        policy_instruction={'type':'string','maxLength':3000},
        evidence_frame_ids={'type':'array','items':{'type':'string','minLength':1},'maxItems':48})
    return dict(type='object',properties=fields,required=list(fields),additionalProperties=False)


def validate_request(value):
    if (not isinstance(value,dict) or set(value)!={'schema_version','request_id','session_id','generation'}
            or type(value['schema_version']) is not int or value['schema_version']!=1
            or type(value['generation']) is not int or value['generation']<0
            or any(not isinstance(value[k],str) or not value[k].strip() for k in ('request_id','session_id'))):
        raise ValueError('Invalid runtime_control request identity')
    return deepcopy(value)


def validate_binding(value,request,arguments,frames,*,allow_future_evidence,current_episode_id):
    expected=validate_request(request)
    if allow_future_evidence:
        raise ValueError('Runtime instruction requires causal, not retrospective, evidence')
    for key in ('request_id','session_id','generation'):
        if (key=='generation' and type(value[key]) is not int) or value[key]!=expected[key]:
            raise ValueError('Runtime '+key+' does not match pending review')
    if arguments['correction']['joint_proposal'] is not None:
        raise ValueError('Runtime policy instruction cannot include a numeric joint proposal')
    instruction=value['policy_instruction'].strip()
    if len(value['evidence_frame_ids'])!=len(set(value['evidence_frame_ids'])):
        raise ValueError('Duplicate runtime evidence frame IDs')
    if value['directive']=='keep_holding' and instruction:
        raise ValueError('keep_holding cannot carry a policy instruction')
    if value['directive']=='replan_policy':
        if not instruction:
            raise ValueError('replan_policy requires an explicit policy instruction')
        ids=value['evidence_frame_ids']
        if not ids or any(identifier not in frames for identifier in ids):
            raise ValueError('Runtime instruction needs bound current image evidence')
        if not any(frames[i].get('evidence_role')!='historical_reference'
                   or frames[i].get('source_episode_id')==current_episode_id for i in ids):
            raise ValueError('Historical references alone cannot authorize a runtime instruction')
    return deepcopy(value)


def instructions():
    return '''
此请求包含 runtime_control，必须明确返回同名结构，原样绑定 request_id、session_id、generation。
directive=keep_holding 表示维持本地保持，此时 policy_instruction 必须为空。
directive=replan_policy 表示让原 VLA 在保持期间基于全新观察重新规划，必须给出明确的 policy_instruction，
并引用当前因果图像。可用于改善无人机观察或纠正抓取，但不能从历史示范直接推断当前目标位置。
若证据不足以提出有意义的下一步，选择 keep_holding。不得输出数值 joint_proposal；它必须为 null。
此决定不是电机指令、物理验收或任务成功证明；设备保持、时限、反馈、限位和恢复检查仍由本地控制器执行。
'''
