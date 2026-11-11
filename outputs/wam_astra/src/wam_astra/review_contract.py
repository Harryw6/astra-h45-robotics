"""Strict offline review contract for native GPT-Policy CodexSession output.

No transport, provider or actuator is implemented here. Structured syntax and
model confidence are neither physical verification nor ground truth.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import re


def _object(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def model_schema(*, runtime_control=False, action_checkpoint=False):
    """Return native ``{name, arguments}`` outputSchema; every key is required."""
    text = {'type': 'string', 'minLength': 1}
    evidence = _object({'frame_id': text, 'camera': text, 'time_s': {'type': 'number'},
                        'observation': text})
    step = _object({'kind': {'type': 'string', 'enum': [
        'observe', 'hold', 'retrieve_reference', 'align', 'reopen', 'reapproach', 'close', 'lift_verify']},
        'description': text, 'evidence_frame_ids': {'type': 'array', 'items': text, 'maxItems': 48}})
    proposal = _object({'joint_units': {'type': 'string', 'enum': ['rad']},
        'gripper_units': {'type': 'string', 'enum': ['normalized']},
        'step_s': {'type': 'number', 'exclusiveMinimum': 0},
        'actions': {'type': 'array', 'minItems': 1, 'maxItems': 256,
            'items': {'type': 'array', 'minItems': 7, 'maxItems': 7, 'items': {'type': 'number'}}}})
    arguments = _object({'schema_version': {'type': 'integer', 'enum': [1]},
        'case_id': text, 'decision_time_s': {'type': 'number'},
        'assessment': _object({'status': {'type': 'string', 'enum': [
            'aligned', 'misaligned', 'empty_grasp', 'grasped', 'uncertain']},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'evidence': {'type': 'array', 'items': evidence, 'maxItems': 48},
            'limitations': {'type': 'array', 'items': text, 'minItems': 1, 'maxItems': 24}}),
        'correction': _object({'strategy': {**text, 'pattern': '[\u3400-\u9fff]'},
            'steps': {'type': 'array', 'items': step, 'minItems': 1, 'maxItems': 24},
            'joint_proposal': {'anyOf': [proposal, {'type': 'null'}]},
            'not_executable_reason': {'type': 'string'}})})
    if type(runtime_control) is not bool:
        raise ValueError('runtime_control schema selection must be boolean')
    if runtime_control:
        from .runtime_contract import output_schema
        arguments['properties']['runtime_control']=output_schema()
        arguments['required'].append('runtime_control')
    if type(action_checkpoint) is not bool:raise ValueError('action_checkpoint schema selection must be boolean')
    if action_checkpoint:
        from .action_checkpoints import output_schema
        arguments['properties']['action_checkpoint']=output_schema()
        arguments['required'].append('action_checkpoint')
    return deepcopy(_object({'name': {'type': 'string', 'enum': ['review_action_stream']},
                             'arguments': arguments}))


def review_instructions():
    """Piper/task guidance for a native decision-only session, not YAM tools."""
    return '''你是 Piper 单臂机器人的离线视觉动作流审查员。仅返回严格匹配输出 schema 的 JSON，
外层 name 必须是 review_action_stream。这是审查数据，不是可调用的运动工具；没有执行器、
shell、SSH、网络或机械臂控制权限。不要引入 YAM 双臂、左右双臂控制或未经标定的 TCP/笛卡尔运动假设。
历史上下文中的 left 只是唯一实体 Piper 机械臂的兼容命名，不代表存在另一只机械臂。

优先遵守当前案例中用户给定的任务上下文。当前用户说明为“目标物体在桌上盒子里，是里面的方块，
起点在地面上，桌子有一定高度”。地面起点按用户原文保留，具体指机器人还是机械臂未独立确认；
目标方块位于桌上盒内，不能说物体从地面开始。
没有给出后续放置位置，不要自行补全任务目标。盒子和夹爪可见不代表盒中方块清晰可见。
用户进一步说明：通过无人机飞高观察桌上箱子和方块，狗前视与机械臂腕部的低位视角起初
看不到目标是正常任务条件。先根据因果画面区分无人机获取全局观察、导航接近、抓取接近、
闭合、抓后核验或阶段未知，在 limitations 中明确阶段及意图证据。不能仅因低位视角看不到
方块或夹爪尚在桌面下就判 misaligned/empty_grasp，也不能把无人机暂时失去目标视野当成空抓。
若只在导航且没有抓取意图证据，不要把夹爪尚未对准目标当作抓取失败。若即将闭合或抬升却缺少
新鲜全局目标位置或目标与夹爪相对关系，明确建议暂停该抓取动作并让无人机补充观察，再重新规划。
分清正常的观测获取过程与真正任务偏差；历史无人机视野不能替代当前可见性或当前目标位置。
把历史示范作为动作顺序与可见几何关系的参考，不把历史轨迹、命令请求或旧任务结果当成当前事实。

逐一对照 frame_id、camera 和 time_s 引用真正提供的画面，描述可见证据与遮挡。
默认只能使用 decision_time_s 及之前的因果证据。不得借助未来画面判断当时状态，
也不得把未来 frame_id 隐藏在 correction.steps 的 evidence_frame_ids 中。
若宿主明确注册 historical_reference、不同 source_episode_id、已存在性声明及参考清单哈希，
旧示范的 time_s 保留原始录制时钟，不能和当前开机时钟直接比较或偷偷重置。
旧示范只能辅助解释纠正计划，不能单独证明当前对齐、空抓或抓住；这些判断至少引用一张当前画面。
只有明确标注为 retrospective 的复盘才允许引用其输入中提供的后续画面，并应说明这是事后分析。
若目标不可见、夹爪遮挡、相机视角不足、时间对齐不确定，应降低自报 confidence 或选择 uncertain，
提出补充观察。不能仅凭夹爪开度、闭合请求、方块被遮挡或夹爪位置推断抓住了物体。
判断 grasped/empty_grasp 需要相应可见证据；lift_verify 是后续核验步骤，不是既成成功。
confidence 是模型自报、未经校准的置信度；你的判断不是 ground truth，任务完成状态仍可能 unknown。
严格区分感知判断、已记录反馈、硬件确认和真实任务结果，不杜撰接触力、抓取成功或执行回执。

correction.strategy 和步骤说明用中文，具体说明补充观察、保持、检索对应示范、重新对齐、
必要时张开/再接近/闭合并抬升核验的顺序。每个步骤只引用真实、合时的 evidence_frame_ids。
缺少标定、实际关节限位、速度/加速度约束、前一控制值或状态时，优先给出 observe/hold/align
等符号纠正方案，joint_proposal 设为 null，not_executable_reason 明确说明缺失条件。
不要捏造关节增量、逆运动学结果、桌面高度、目标坐标、机械臂增益或所谓通用安全限位。
若确有充分已验证信息支持数值候选，joint_units 必须 rad、gripper_units 必须 normalized，
actions 每行只能是六个关节目标加一个明确的离散 0/1 夹爪命令；step_s 必须大于零。
测量的 gripper_measured_m 是米，绝不直接当作 normalized 命令，不能从开度猜测先前命令。
没有完整真实物理约束时，任何数值候选也必须明确不可执行，保留 not_executable_reason。
即便语法与离散限位检查通过，也只是一份离线候选，不证明连续时间安全、碰撞可行性或任务成功。'''


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real finite number')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a real finite number')
    return value


def _frames(available, current_episode_id=None):
    if isinstance(available, list):
        result = {}
        for item in available:
            if not isinstance(item, dict) or not isinstance(item.get('frame_id'), str) or item['frame_id'] in result:
                raise ValueError('Frame records require unique frame_id strings')
            result[item['frame_id']] = item
    elif isinstance(available, dict):
        result = deepcopy(available)
    else:
        raise ValueError('available_frames must be an ID mapping or frame-record list')
    for identifier, frame in result.items():
        if not isinstance(identifier, str) or not identifier or not isinstance(frame, dict):
            raise ValueError('Frame IDs and metadata must be unambiguous')
        if not isinstance(frame.get('camera'), str) or not frame['camera']:
            raise ValueError('Every available frame requires a camera')
        if frame.get('frame_id', identifier) != identifier:
            raise ValueError('Frame metadata ID does not match its mapping key')
        _finite_number(frame.get('time_s'), 'Frame time_s')
        role = frame.get('evidence_role')
        if 'evidence_role' in frame and role not in ('current_observation', 'historical_reference'):
            raise ValueError('Unsupported frame evidence_role')
        reference_flags = {'availability_attested', 'reference_manifest_sha256'}
        if reference_flags.intersection(frame) and role != 'historical_reference':
            raise ValueError('Partial historical reference attestation lacks evidence_role')
        if role == 'historical_reference':
            source = frame.get('source_episode_id')
            digest = frame.get('reference_manifest_sha256')
            if not isinstance(source, str) or not source.strip():
                raise ValueError('Historical reference requires a nonempty source_episode_id')
            if frame.get('availability_attested') is not True:
                raise ValueError('Historical reference requires explicit availability_attested=true')
            if not isinstance(digest, str) or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
                raise ValueError('Historical reference requires a valid reference_manifest_sha256')
            if current_episode_id is None:
                raise ValueError('Historical references require explicit current_episode_id')
    return result


def _limits_complete(limits):
    if not isinstance(limits, dict) or limits.get('verified') is not True:
        return False
    if limits.get('joint_units') != 'rad' or limits.get('gripper_units') != 'normalized':
        return False
    for key, length in [('joint_lower', 6), ('joint_upper', 6), ('max_velocity', 6),
                        ('max_acceleration', 6), ('previous_action', 7), ('previous_velocity', 6)]:
        value = limits.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != length:
            return False
        try:
            for item in value:
                _finite_number(item, key)
        except ValueError:
            return False
    if limits['previous_action'][6] not in (0., 1.):
        return False
    return True


def validate_review_reply(reply, *, case_id, decision_time_s, available_frames,
                          allow_future_evidence=False, physical_limits=None, current_episode_id=None,
                          runtime_control=None, action_checkpoint=None):
    """Bind native output to trusted supplied frames and validate offline only.

    ``available_frames`` is ``{id: {camera, time_s, ...}}`` or a unique-ID list.
    Metadata must have been derived from actual supplied images by the caller.
    ``verified`` physical metadata is a caller attestation, not verification by
    this function. The function never sends an action or schedules a queue.
    """
    from jsonschema import Draft202012Validator, ValidationError

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError('case_id must be nonempty text')
    _finite_number(decision_time_s, 'decision_time_s')
    if type(allow_future_evidence) is not bool:
        raise ValueError('allow_future_evidence must be boolean')
    if current_episode_id is not None:
        if not isinstance(current_episode_id, str) or not current_episode_id.strip():
            raise ValueError('current_episode_id must be nonempty text when supplied')
        current_episode_id = current_episode_id.strip()
    try:
        json.dumps(reply, allow_nan=False)
        Draft202012Validator(model_schema(runtime_control=runtime_control is not None,
                                         action_checkpoint=action_checkpoint is not None)).validate(reply)
    except (ValidationError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid review JSON contract: ' + str(exc).splitlines()[0][:300]) from exc
    arguments = deepcopy(reply['arguments'])
    if type(arguments['schema_version']) is not int:
        raise ValueError('schema_version must be integer 1')
    if arguments['case_id'] != case_id:
        raise ValueError('Foreign case_id')
    _finite_number(arguments['decision_time_s'], 'reply decision_time_s')
    if arguments['decision_time_s'] != decision_time_s:
        raise ValueError('Reply decision time does not match case decision time')
    frames = _frames(available_frames, current_episode_id)
    future_used = False
    historical_reference_ids = set()
    current_assessment_ids = set()
    if arguments['assessment']['status'] != 'uncertain' and not arguments['assessment']['evidence']:
        raise ValueError('Categorical visual assessment requires at least one bound image evidence record')

    def bind(identifier):
        nonlocal future_used
        if identifier not in frames:
            raise ValueError(f'Unknown evidence frame_id: {identifier}')
        frame = frames[identifier]
        reference = frame.get('evidence_role') == 'historical_reference'
        if reference:
            historical_reference_ids.add(identifier)
        distinct_prior_episode = reference and frame['source_episode_id'].strip() != current_episode_id
        if not distinct_prior_episode and frame['time_s'] > decision_time_s:
            if not allow_future_evidence:
                raise ValueError(f'Future evidence violates causal decision: {identifier}')
            future_used = True
        return frame

    for evidence in arguments['assessment']['evidence']:
        frame = bind(evidence['frame_id'])
        _finite_number(evidence['time_s'], 'Evidence time_s')
        if evidence['camera'] != frame['camera'] or evidence['time_s'] != frame['time_s']:
            raise ValueError('Evidence camera/time does not exactly match its real supplied frame')
        if frame.get('evidence_role') != 'historical_reference':
            current_assessment_ids.add(evidence['frame_id'])
    if arguments['assessment']['status'] != 'uncertain' and not current_assessment_ids:
        raise ValueError('Categorical visual assessment requires at least one current image evidence record')
    for step in arguments['correction']['steps']:
        for identifier in step['evidence_frame_ids']:
            bind(identifier)
    runtime_directive=None
    if runtime_control is not None:
        from .runtime_contract import validate_binding
        runtime_directive=validate_binding(arguments['runtime_control'],runtime_control,arguments,frames,
            allow_future_evidence=allow_future_evidence,current_episode_id=current_episode_id)
        for identifier in runtime_directive['evidence_frame_ids']:
            bind(identifier)
    proposal = arguments['correction']['joint_proposal']
    checkpoint_result=None
    if action_checkpoint is not None:
        from .action_checkpoints import validate_answer
        if allow_future_evidence:raise ValueError('Checkpoint review requires causal evidence')
        if proposal is not None:raise ValueError('H45 checkpoints cannot use normalized joint_proposal')
        checkpoint_result=validate_answer(arguments['action_checkpoint'],action_checkpoint)
        current=False
        for identifier in checkpoint_result['evidence_frame_ids']:
            frame=bind(identifier)
            if frame.get('evidence_role')!='historical_reference':current=True
        if not current:raise ValueError('Checkpoint review needs current image evidence')
    if proposal is not None:
        _finite_number(proposal['step_s'], 'step_s')
        for row in proposal['actions']:
            for value in row:
                _finite_number(value, 'Joint proposal target')
            if row[6] not in (0., 1.):
                raise ValueError('Unsupported normalized gripper command; only discrete 0/1 is accepted')
    constraints_available = _limits_complete(physical_limits)
    candidate_validatable = proposal is not None and constraints_available
    if candidate_validatable:
        from .rtc import validate_trajectory
        validate_trajectory(proposal['actions'], step_s=proposal['step_s'],
            **{key: physical_limits[key] for key in (
                'joint_lower', 'joint_upper', 'max_velocity', 'max_acceleration',
                'previous_action', 'previous_velocity')})
    reason = arguments['correction']['not_executable_reason']
    executable = candidate_validatable and not allow_future_evidence and not future_used and not reason.strip()
    if not executable and not reason.strip():
        raise ValueError('An explicit not_executable_reason is mandatory without a validated executable proposal')
    summary = {'schema_version': 1, 'case_id': case_id, 'decision_time_s': decision_time_s,
        'arguments': arguments, 'executable': executable,
        'effective_decision': 'offline_proposal_validated' if executable else 'hold',
        'evidence_mode': 'retrospective' if allow_future_evidence else 'causal',
        'used_future_evidence': future_used, 'physical_limits_complete_and_attested': constraints_available,
        'assessment_is_ground_truth': False, 'confidence_is_calibrated': False,
        'task_outcome': 'unknown', 'offline_only': True, 'actuator_calls': 0}
    if historical_reference_ids:
        summary['historical_reference_ids'] = sorted(historical_reference_ids)
    if runtime_directive is not None:
        summary['runtime_policy_directive']=runtime_directive
        summary['runtime_policy_instruction_only']=True
    if checkpoint_result is not None:summary['checkpoint_action_review']=checkpoint_result
    return summary
