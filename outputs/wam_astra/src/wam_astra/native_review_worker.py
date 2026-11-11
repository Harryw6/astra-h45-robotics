"""Linux/WSL entry point for actual, unchanged GPT-Policy CodexSession calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import time

from wam_astra.review_contract import model_schema, review_instructions, validate_review_reply

PIN='2ca201638e1139fe35973c8a2d209a894b389db1'
MODEL='gpt-6-astra'


def review_prompt_contract(request):
    instructions=review_instructions()
    checkpoint='action_checkpoint' in request
    if checkpoint:
        from wam_astra.action_checkpoints import validate_request, instructions as checkpoint_instructions
        validate_request(request['action_checkpoint'])
        instructions+=checkpoint_instructions()
    if 'runtime_control' in request:
        from wam_astra.runtime_contract import validate_request, instructions as runtime_instructions
        validate_request(request['runtime_control'])
        return instructions+runtime_instructions(),model_schema(runtime_control=True,action_checkpoint=checkpoint)
    return instructions,model_schema(action_checkpoint=checkpoint)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def local_path(value,base):
    windows=PureWindowsPath(value)
    if windows.drive:
        if len(windows.drive)!=2 or windows.drive[1]!=':' or not windows.is_absolute():
            raise ValueError('Only absolute local drive paths can cross Windows/WSL')
        return Path('/mnt')/windows.drive[0].lower()/Path(*windows.parts[1:])
    path=Path(value.replace('\\','/'))
    return path if path.is_absolute() else base/path


def event_summary(event):
    method=event.get('method')
    if method not in ('turn/started','turn/completed','item/started','item/completed','error'):
        return None
    params=event.get('params',{});turn=params.get('turn',{});item=params.get('item',{})
    return {'method':method,'thread_id':params.get('threadId'),'turn_id':params.get('turnId',turn.get('id')),
            'status':turn.get('status'),'item_type':item.get('type'),'phase':item.get('phase')}


def configure_native_cwd(session,cwd,*,event_log=None):
    """Keep the same Windows cwd when native recovery recreates its client."""
    original_factory=session._new_client
    def configure(client):
        client.project_root=Path(cwd)
        if event_log is not None:
            receive=client._receive
            def receive_and_journal():
                event=receive();summary=event_summary(event)
                if summary is not None:
                    with Path(event_log).open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'at_s':time.time(),**summary})+'\n')
                return event
            client._receive=receive_and_journal
        return client
    def new_client():
        return configure(original_factory())
    session._new_client=new_client
    configure(session.client)
    # Project-wide Astra default; unrelated Codex model sessions are untouched.
    if getattr(session.client,'model',None)=='gpt-6-astra':
        from wam_astra.astra_runtime import configure_astra_fast
        fast_log=Path(event_log).with_suffix('.fast.jsonl') if event_log is not None else None
        configure_astra_fast(session,event_log=fast_log)


def verify_upstream(repo,*,expected_commit=PIN):
    """Verify tracked canonical content across a CRLF Windows/WSL checkout.

    Options are per-process only. No checkout, source bytes or Git config is
    rewritten. Real content edits, including staged edits, still fail.
    """
    command=['git','-c',f'safe.directory={repo}','-c','core.autocrlf=true',
             '-c','core.filemode=false','-C',str(repo)]
    commit=subprocess.run([*command,'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if commit!=expected_commit:raise ValueError('GPT-Policy must use the pinned upstream commit')
    dirty=subprocess.run([*command,'diff','HEAD','--name-only'],check=True,capture_output=True,text=True).stdout.strip()
    if dirty:raise ValueError('Upstream source is modified; refusing unverified native implementation')
    return commit


def compact_measured_prefix(observation,*,compress_samples=None):
    """Use native historical sampling; keep every aperture/zeroing boundary.

    This changes only the model's context, never the raw measurements or their
    archived request. Zeroing anchors are recording-quality boundaries, not a
    physical reset interpretation or task-success detector.
    """
    if compress_samples is None:
        from gpt_policy.input.action_sampling import compress_samples
    result=deepcopy(observation)
    state=result.get('observation',{})
    rows=state.get('measured_prefix_or_audit',[])
    if not rows:return result
    samples=[];anchors=set()
    for index,row in enumerate(rows):
        left={'joint_measured_rad':row['joint_measured_rad']}
        if row.get('gripper_measured_m') is not None:left['gripper_measured_m']=row['gripper_measured_m']
        samples.append({'t_s':row['video_relative_time_s'],'left':left,'source_index':index})
        if index:
            previous=rows[index-1]
            if (previous.get('gripper_measured_m')!=row.get('gripper_measured_m') or
                all(v==0 for v in previous['joint_measured_rad'])!=all(v==0 for v in row['joint_measured_rad'])):
                anchors.update((index-1,index))
    kept,info=compress_samples(samples,preserve_indices=sorted(anchors))
    state['measured_prefix_or_audit']=[rows[sample['source_index']] for sample in kept]
    state['measurement_context_sampling']={**info,'upstream_commit':PIN,
        'implementation':'gpt_policy.input.action_sampling.compress_samples',
        'anchor_semantics':'all recorded metre-aperture changes and all-zero/nonzero telemetry boundaries; no physical interpretation',
        'full_samples_archived':'request.json'}
    return result


def run(args):
    from gpt_policy.harness.config import AgentConfig
    from gpt_policy.harness.models import AgentContext, AgentTurn
    from gpt_policy.harness.providers.codex import CodexSession
    from gpt_policy.harness.usage import collect_usage
    from gpt_policy.harness.waiting import monitor_health
    from gpt_policy.input.manifest import load_manifest, content_records
    from gpt_policy.harness.input_content import to_app_server_items

    target=Path(args.output);repo=Path(args.upstream_root)
    commit=verify_upstream(repo)
    request=json.loads(Path(args.request).read_text(encoding='utf-8-sig'))
    original_manifest=local_path(request['input_manifest'],Path(args.request).parent).resolve()
    manifest_json=json.loads(original_manifest.read_text(encoding='utf-8-sig'))
    if manifest_json.get('model')!=MODEL:raise ValueError('Requested model is not gpt-6-astra')
    for item in manifest_json['content']:
        if isinstance(item,dict):item['image']=str(local_path(item['image'],original_manifest.parent).resolve(strict=True))
    save(target/'native_input.json',manifest_json)
    manifest=load_manifest(target/'native_input.json')
    save(target/'app_server_items.json',to_app_server_items(manifest.content))
    receipt={'status':'started','model':MODEL,'upstream_commit':commit,
             'native_harness':'gpt_policy.harness.providers.codex.CodexSession',
             'native_sources_unchanged':True,'decision_only':True,'actuator_calls':0,
             'case_id':request['case_id'],'phase':request['phase'],
             'original_manifest_sha256':hashlib.sha256(original_manifest.read_bytes()).hexdigest()}
    save(target/'native_receipt.json',receipt)
    started=time.perf_counter();session=None
    try:
        with collect_usage() as ledger:
            ledger.attach(target)
            with monitor_health(None,deadline=time.monotonic()+args.timeout_s):
                config=AgentConfig(model=MODEL,executable=args.codex_bin,effort=args.effort,decision_only=True)
                session=CodexSession(config,MODEL,False,85)
                configure_native_cwd(session,args.codex_cwd,event_log=target/'native_events.jsonl')
                account=session.client._request('account/read',{'refreshToken':False})
                receipt['account_type']=(account.get('account') or {}).get('type')
                instructions,schema=review_prompt_contract(request)
                (target/'system_instructions.txt').write_text(instructions,encoding='utf-8')
                save(target/'output_schema.json',schema)
                receipt['system_instructions_sha256']=hashlib.sha256((target/'system_instructions.txt').read_bytes()).hexdigest()
                receipt['output_schema_sha256']=hashlib.sha256((target/'output_schema.json').read_bytes()).hexdigest()
                session.start(AgentContext(instructions,[],schema))
                receipt['thread_id']=session.thread_id
                save(target/'native_receipt.json',receipt)
                observation={key:value for key,value in request.items() if key!='input_manifest'}
                observation=compact_measured_prefix(observation)
                observation['instruction']=manifest.instruction
                observation['protocol_notice']=('Return one review_action_stream selection. No robot tool executes in this model session. '
                    'Any runtime policy instruction remains subject to local feedback and recovery checks.' if 'runtime_control' in request else
                    'Return one review_action_stream selection. This is offline review; no robot tool will execute.')
                save(target/'observation.json',observation)
                decision=session.decide(AgentTurn(json.dumps(observation,ensure_ascii=False,allow_nan=False),content=manifest.content))
                save(target/'native_decision.json',decision)
                reply=decision['_wire']
                save(target/'reply.json',reply)
                validation=validate_review_reply(reply,case_id=request['case_id'],decision_time_s=request['decision_time_s'],
                    available_frames=request['available_frames'],allow_future_evidence=request['phase']=='retrospective_audit',physical_limits=None,
                    current_episode_id=request.get('current_episode_id',request.get('observation',{}).get('source_episode')),
                    runtime_control=request.get('runtime_control'),action_checkpoint=request.get('action_checkpoint'))
                save(target/'validation.json',validation)
                completed_calls=[call for call in ledger.calls if call.get('status')=='completed']
                if not completed_calls:raise RuntimeError('Native harness returned no completed usage call')
                receipt.update(status='completed',elapsed_s=time.perf_counter()-started,
                               turn_id=completed_calls[-1].get('turn_id'),model_calls=len(ledger.calls),
                               reply_sha256=hashlib.sha256((target/'reply.json').read_bytes()).hexdigest(),
                               request_sha256=hashlib.sha256(Path(args.request).read_bytes()).hexdigest(),
                               timing=session.last_decision_timing,recovery=session.last_recovery)
    except Exception as exc:
        receipt.update(status='failed',elapsed_s=time.perf_counter()-started,
                       error_type=type(exc).__name__,error=str(exc))
        save(target/'native_receipt.json',receipt)
        raise
    finally:
        if session is not None:session.close()
    save(target/'native_receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'thread_id':receipt['thread_id'],'elapsed_s':receipt['elapsed_s']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','output','upstream-root','codex-bin','codex-cwd'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--effort',default='high')
    parser.add_argument('--timeout-s',type=float,default=600.)
    run(parser.parse_args())


if __name__=='__main__':main()
