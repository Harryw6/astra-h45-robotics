"""H45 checkpoint events and numeric review candidates, never actuator authority.

The native H45 transport/executor still owns coordinates, timing and limits.
These small adapters reuse its raw 45x14 schema, not normalized Piper actions.
"""
from copy import deepcopy
import hashlib
import json
import math
import threading

KINDS=('uav_rotation_start','uav_rotation_end','gripper_pre_close')
SCHEMA='heterovla_v3_goal_distance_h45_masked_v1'
UNITS=['m','m','rad']+['rad']*6+['m','m','m','m','rad']


def number(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Expected finite '+name)
    return float(value)


def matrix(value):
    if not isinstance(value,list) or len(value)!=45:
        raise ValueError('Expected native H45 45x14 candidate')
    for row in value:
        if not isinstance(row,list) or len(row)!=14:raise ValueError('Expected H45 14D vector')
        for v in row:number(v,'H45 vector value')
    return deepcopy(value)


def candidate_digest(candidate):
    if (not isinstance(candidate,dict) or not isinstance(candidate.get('observation_id'),str)
            or not candidate['observation_id'] or type(candidate.get('generation')) is not int
            or candidate['generation']<0):raise ValueError('Missing candidate observation/generation')
    body=dict(observation_id=candidate['observation_id'],generation=candidate['generation'],
        origin_at=number(candidate.get('origin_at'),'candidate origin'),
        model_actions=matrix(candidate.get('model_actions')))
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_request(event,candidate):
    if (not isinstance(event,dict) or not isinstance(event.get('event_id'),str) or not event['event_id']
            or not isinstance(event.get('checkpoint_kinds'),list) or not event['checkpoint_kinds']
            or len(set(event['checkpoint_kinds']))!=len(event['checkpoint_kinds'])
            or any(k not in KINDS for k in event['checkpoint_kinds'])):
        raise ValueError('Invalid checkpoint event')
    return dict(schema_version=1,checkpoint_id=event['event_id'],checkpoint_kinds=deepcopy(event['checkpoint_kinds']),
        action_schema=SCHEMA,action_units=list(UNITS),step_s=1/15,
        candidate_sha256=candidate_digest(candidate),candidate=deepcopy(candidate))


def validate_request(value):
    if not isinstance(value,dict) or type(value.get('schema_version')) is not int:
        raise ValueError('Invalid action checkpoint request')
    rebuilt=make_request(dict(event_id=value.get('checkpoint_id'),checkpoint_kinds=value.get('checkpoint_kinds')),
                         value.get('candidate'))
    if rebuilt!=value:raise ValueError('Checkpoint request hash/schema/units mismatch')
    return rebuilt


def output_schema():
    text=dict(type='string',minLength=1)
    props=dict(checkpoint_id=text,candidate_sha256=dict(type='string',pattern='^[a-f0-9]{64}$'),
        decision=dict(type='string',enum=['valid','invalid','uncertain']),reason=text,
        evidence_frame_ids=dict(type='array',items=text,minItems=1,maxItems=48),
        replacements=dict(type='array',maxItems=45,items=dict(type='object',additionalProperties=False,
            properties=dict(row=dict(type='integer',minimum=0,maximum=44),
                vector=dict(type='array',items=dict(type='number'),minItems=14,maxItems=14)),
            required=['row','vector'])),cannot_correct_reason=dict(type='string'))
    return dict(type='object',properties=props,required=list(props),additionalProperties=False)


def validate_answer(answer,request):
    from jsonschema import Draft202012Validator,ValidationError
    request=validate_request(request)
    try:Draft202012Validator(output_schema()).validate(answer)
    except (ValidationError,TypeError) as exc:raise ValueError('Invalid checkpoint answer') from exc
    if (answer['checkpoint_id']!=request['checkpoint_id'] or answer['candidate_sha256']!=request['candidate_sha256']):
        raise ValueError('Obsolete/foreign checkpoint candidate')
    if len(set(answer['evidence_frame_ids']))!=len(answer['evidence_frame_ids']):raise ValueError('Duplicate evidence')
    seen=set();corrected=matrix(request['candidate']['model_actions'])
    for patch in answer['replacements']:
        row=patch['row']
        if type(row) is not int or row in seen:raise ValueError('Duplicate/noninteger replacement row')
        seen.add(row)
        for v in patch['vector']:number(v,'replacement value')
        if patch['vector']==corrected[row]:raise ValueError('Replacement does not change the candidate')
        corrected[row]=deepcopy(patch['vector'])
    if answer['decision']!='invalid' and seen:raise ValueError('Only invalid actions carry corrections')
    if answer['decision']=='invalid' and not seen and not answer['cannot_correct_reason'].strip():
        raise ValueError('Missing corrected vectors or explicit numerical limitation')
    if seen and answer['cannot_correct_reason'].strip():raise ValueError('Conflicting numerical correction limitation')
    if answer['decision']=='uncertain' and not answer['cannot_correct_reason'].strip():
        raise ValueError('Uncertain review must explain missing evidence')
    return dict(**deepcopy(answer),corrected_actions=corrected if seen else None,
        action_schema=SCHEMA,action_units=list(UNITS),step_s=1/15,executable=False,
        execution_notice='Bound to reviewed observation/anchor only. Fresh anchoring, trajectory/physical constraints and device validation required.')


def instructions():
    return '''
本请求另外包含 action_checkpoint。审查用户指定的节点：uav_rotation_start=无人机旋转开始前；
uav_rotation_end=实际旋转完成后；gripper_pre_close=夹爪闭合前。必须明确 action_checkpoint.decision：
valid（当前动作有证据支持）、invalid（动作无效或存在任务偏差）、uncertain（证据不足）。
原样返回 checkpoint_id 与 candidate_sha256，引用当前因果画面，说明该动作是否有利于看清/对准盒内方块。
无效且具备数值依据时，在 replacements 中按 row=0..44 输出修改后的完整14维向量；未列出的行原样保留。
14维严格为：[狗dx(m),dy(m),dyaw(rad),Piper六关节相对观察锚点增量(rad),夹爪绝对开度(m),
无人机相对观察锚点dx(m),dy(m),dz(m),dyaw(rad)]。不能混用旧7维normalized夹爪格式。
这是绑定原始观察和动作块的数值建议，不允许自行执行，也不能拿几十秒前的锚点直接下发。
只修改有证据支持的维度/行；不可凭图捏造外参、目标3D坐标、关节逆解、速度限制或安全性。
如果明确无效但无法可靠求数值修正，仍标invalid，replacements=[]，cannot_correct_reason解释缺失条件；
uncertain也必须说明原因。valid/uncertain不可夹带替换向量。既有joint_proposal仍为null。
runtime_control若存在，证据不足保持；可明确改善观察/重新规划时给replan_policy指令。
'''


def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))


class CheckpointMonitor:
    def __init__(self,*,rotation_start_rad,rotation_stop_rate_rad_s,rotation_target_tolerance_rad,
                 min_rotation_progress_rad,rotation_settle_s,min_settle_samples,max_feedback_age_s,
                 gripper_close_delta_m):
        for name,value in locals().copy().items():
            if name in ('self','min_settle_samples'):continue
            value=number(value,name)
            if value<=0:raise ValueError('Positive '+name+' required')
            setattr(self,name,value)
        if type(min_settle_samples) is not int or min_settle_samples<2:
            raise ValueError('Distinct settling feedback samples required')
        if max(rotation_start_rad,rotation_target_tolerance_rad,min_rotation_progress_rad)>=math.pi:
            raise ValueError('Rotation thresholds must be below pi')
        self.min_settle_samples=min_settle_samples;self._lock=threading.RLock();self.rotation=None

    def prediction(self,candidate,*,gripper_m):
        digest=candidate_digest(candidate);aperture=number(gripper_m,'measured gripper metres')
        actions=candidate['model_actions'];first={}
        with self._lock:
            if self.rotation is None or self.rotation['finished']:
                for i,row in enumerate(actions):
                    if abs(row[13])>=self.rotation_start_rad:first['uav_rotation_start']=i;break
            for i,row in enumerate(actions):
                if row[9]<=aperture-self.gripper_close_delta_m:first['gripper_pre_close']=i;break
        if not first:return None
        return dict(event_id='checkpoint_'+digest,reason='关键节点动作有效性审查',
            checkpoint_kinds=list(first),first_rows=first,candidate_sha256=digest)

    def _sample(self,sample,now):
        stamp=number(sample.get('stamp'),'feedback stamp');source=number(sample.get('source_stamp'),'source stamp')
        if source<=0 or not 0<=now-stamp<=self.max_feedback_age_s:raise ValueError('Stale/future feedback')
        pose=sample.get('pose');velocity=sample.get('velocity')
        if not isinstance(pose,list) or len(pose)!=4 or not isinstance(velocity,list) or len(velocity)!=4:
            raise ValueError('Measured UAV pose4/velocity4 required')
        return dict(stamp=stamp,source=source,yaw=number(pose[3],'yaw'),rate=number(velocity[3],'yaw rate'))

    def arm_rotation(self,sample,*,target_yaw,rotation_id):
        start=self._sample(sample,number(sample.get('stamp'),'entry time'))
        target=number(target_yaw,'rotation target')
        if not isinstance(rotation_id,str) or not rotation_id:raise ValueError('Rotation identity required')
        with self._lock:
            self.rotation=dict(id=rotation_id,start=start,target=wrap(target),previous=start,advanced=start['stamp'],
                               since=None,count=0,finished=False,moved=False)

    def track_commit(self,sample,actions,*,plan_id):
        """Call only after BOTH native endpoints acknowledged a new action plan."""
        actions=matrix(actions);start=self._sample(sample,number(sample.get('stamp'),'commit observation time'))
        target=wrap(start['yaw']+actions[-1][13])
        with self._lock:
            if self.rotation is None or self.rotation['finished']:
                if not any(abs(row[13])>=self.rotation_start_rad for row in actions):return
                self.arm_rotation(sample,target_yaw=target,rotation_id=plan_id)
            elif abs(wrap(target-self.rotation['target']))>self.rotation_target_tolerance_rad:
                self.rotation['target']=target;self.rotation['since']=None;self.rotation['count']=0

    def feedback(self,sample,*,now_s):
        current=self._sample(sample,number(now_s,'feedback now'))
        with self._lock:
            r=self.rotation
            if r is None or r['finished']:return None
            previous=r['previous']
            if current['stamp']<previous['stamp'] or current['source']<previous['source']:
                raise ValueError('Feedback clock went backwards')
            if current['stamp']==previous['stamp']:
                if current!=previous:raise ValueError('Same timestamp changed feedback')
                return None
            if current['source']==previous['source']:
                r['since']=None;r['count']=0
                if now_s-r['advanced']>self.max_feedback_age_s:raise ValueError('Frozen feedback source')
                return None
            if current['stamp']-previous['stamp']>self.max_feedback_age_s:
                r['since']=None;r['count']=0
            r['advanced']=current['stamp'];r['previous']=current
            reached=abs(wrap(current['yaw']-r['target']))<=self.rotation_target_tolerance_rad
            r['moved']=r['moved'] or abs(wrap(current['yaw']-r['start']['yaw']))>=self.min_rotation_progress_rad
            if not reached or not r['moved'] or abs(current['rate'])>self.rotation_stop_rate_rad_s:
                r['since']=None;r['count']=0;return None
            if r['since'] is None:r['since']=current['stamp']
            r['count']+=1
            if r['count']<self.min_settle_samples or current['stamp']-r['since']<self.rotation_settle_s:return None
            r['finished']=True
            return dict(event_id='rotation_end_'+r['id'],reason='无人机实际旋转结束，核验视野与后续动作',
                checkpoint_kinds=['uav_rotation_end'],rotation_id=r['id'],feedback=deepcopy(sample))
