"""Explicit real-model launcher using the unchanged GPT-Policy native harness."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

from .contexts import UPSTREAM_COMMIT, _finite_json, _under_wam, _workspace, _wsl_path
from .data import file_sha256, read_json, write_json

MODEL='gpt-6-astra'


def prepare_review(request_path, destination):
    """Validate immutable local inputs before starting any model process."""
    source=Path(request_path).resolve();target=Path(destination).resolve()
    protected=[source.parent,Path(__file__).resolve().parents[2]/'library']
    if _under_wam(destination) or _under_wam(target) or any(target.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Output must be outside the raw source, case source directory and library')
    if target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
    request=read_json(source);_finite_json(request)
    if 'runtime_control' in request:
        from .runtime_contract import validate_request
        validate_request(request['runtime_control'])
    if 'action_checkpoint' in request:
        from .action_checkpoints import validate_request
        validate_request(request['action_checkpoint'])
    if type(request.get('schema_version')) is not int or request['schema_version']!=1:
        raise ValueError('request requires schema_version 1')
    for key in ('case_id','task_instruction','input_manifest'):
        if not isinstance(request.get(key),str) or not request[key].strip():
            raise ValueError(f'{key} must be nonempty text')
    if request.get('phase') not in ('causal_review','retrospective_audit'):
        raise ValueError('phase must be causal_review or retrospective_audit')
    decision=request.get('decision_time_s')
    if isinstance(decision,bool) or not isinstance(decision,(int,float)) or not math.isfinite(decision):
        raise ValueError('decision_time_s must be finite')
    frames=request.get('available_frames')
    if not isinstance(frames,dict) or not frames:raise ValueError('available_frames must be a nonempty ID mapping')
    from .review_contract import _frames
    current_episode=request.get('current_episode_id',request.get('observation',{}).get('source_episode'))
    if current_episode is not None and (not isinstance(current_episode,str) or not current_episode.strip()):
        raise ValueError('current_episode_id must be nonempty text')
    if current_episode is not None and current_episode != current_episode.strip():
        raise ValueError('current_episode_id must use canonical text without surrounding whitespace')
    frames=_frames(frames,current_episode_id=current_episode)
    reference_manifests=[]
    for frame_id,frame in frames.items():
        if not isinstance(frame_id,str) or not frame_id or not isinstance(frame,dict):
            raise ValueError('Each frame requires an ID and metadata')
        if frame.get('evidence_role')=='historical_reference' and frame['source_episode_id']!=frame['source_episode_id'].strip():
            raise ValueError('Historical source_episode_id must use canonical text without surrounding whitespace')
        time_s=frame.get('time_s')
        if not isinstance(frame.get('camera'),str) or not frame['camera']:
            raise ValueError('Each frame requires camera')
        if isinstance(time_s,bool) or not isinstance(time_s,(int,float)) or not math.isfinite(time_s):
            raise ValueError('Frame time must be finite')
        is_historical=frame.get('evidence_role')=='historical_reference' and frame.get('source_episode_id')!=current_episode
        if request['phase']=='causal_review' and time_s>decision and not is_historical:
            raise ValueError('A causal review cannot contain future frames')
        if frame.get('evidence_role')=='historical_reference':
            reference_path=(source.parent/request.get('reference_library_manifest',str(Path(__file__).resolve().parents[2]/'library/manifest.json'))).resolve()
            digest=file_sha256(reference_path)
            if digest!=frame['reference_manifest_sha256']:
                raise ValueError('Historical reference manifest digest does not match actual source')
            reference=read_json(reference_path)
            if frame['source_episode_id'] not in {e['episode_id'] for e in reference.get('episodes',[])}:
                raise ValueError('Historical source episode is absent from the reference manifest')
            entry={'path':str(reference_path),'sha256':digest}
            if entry not in reference_manifests:reference_manifests.append(entry)
    manifest_path=(source.parent/request['input_manifest']).resolve()
    manifest=read_json(manifest_path);_finite_json(manifest)
    if manifest.get('model')!=MODEL:raise ValueError('Only the explicitly requested gpt-6-astra model is supported')
    if not isinstance(manifest.get('content'),list) or not manifest['content']:
        raise ValueError('Native input manifest content is missing')
    images=[]
    supplied_ids=[]
    for part in manifest['content']:
        if isinstance(part,str):continue
        if not isinstance(part,dict) or 'image' not in part or 'video' in part:
            raise ValueError('Prepared native input requires text and actual image parts')
        path=(manifest_path.parent/part['image']).resolve()
        label=part.get('label')
        frame_id=label.split()[0] if isinstance(label,str) and label.strip() else None
        if frame_id not in frames:raise ValueError('Every input image must name a registered frame ID in its label')
        registered_path=frames[frame_id].get('path')
        if registered_path is not None and (source.parent/registered_path).resolve()!=path:
            raise ValueError('Registered frame path does not match supplied input image')
        with Image.open(path) as picture:picture.verify()
        supplied_ids.append(frame_id)
        images.append({'path':str(path),'sha256':file_sha256(path),'label':label,'frame_id':frame_id})
    if not images:raise ValueError('At least one actual image is required')
    if len(supplied_ids)!=len(set(supplied_ids)) or set(supplied_ids)!=set(frames):
        raise ValueError('Registered frame IDs must match exactly the supplied input images')
    return {'model':MODEL,'request_path':str(source),'request_sha256':file_sha256(source),
            'manifest_path':str(manifest_path),'manifest_sha256':file_sha256(manifest_path),
            'destination':str(target),'images':images,'reference_manifests':reference_manifests,'model_calls':0}


def load_review_result(destination):
    root=Path(destination).resolve();receipt=read_json(root/'native_receipt.json')
    if receipt.get('status')!='completed' or receipt.get('model')!=MODEL:
        raise RuntimeError('No completed gpt-6-astra inference turn is recorded')
    if receipt.get('upstream_commit')!=UPSTREAM_COMMIT:
        raise RuntimeError('Unexpected native GPT-Policy revision')
    reply=read_json(root/'reply.json');validation=read_json(root/'validation.json')
    if not reply or not validation:raise RuntimeError('Completed inference artifacts are incomplete')
    request=read_json(root/'request.json')
    if not request:raise RuntimeError('Completed inference request artifact is missing')
    from .review_contract import validate_review_reply
    verified=validate_review_reply(reply,case_id=request['case_id'],decision_time_s=request['decision_time_s'],
        available_frames=request['available_frames'],allow_future_evidence=request['phase']=='retrospective_audit',physical_limits=None,
        current_episode_id=request.get('current_episode_id',request.get('observation',{}).get('source_episode')),
        runtime_control=request.get('runtime_control'),action_checkpoint=request.get('action_checkpoint'))
    if validation!=verified:raise RuntimeError('Archived validation artifact does not match the actual reply/request contract')
    usage=read_json(root/'usage.json')
    if (receipt.get('reply_sha256')!=file_sha256(root/'reply.json') or
        receipt.get('request_sha256')!=file_sha256(root/'request.json')):
        raise RuntimeError('Native review artifact hash mismatch')
    if not receipt.get('thread_id') or not receipt.get('turn_id'):
        raise RuntimeError('Native thread/turn identity is missing')
    if not any(c.get('status')=='completed' and c.get('model')==MODEL
               and c.get('thread_id')==receipt['thread_id'] and c.get('turn_id')==receipt['turn_id']
               for c in [json.loads(line) for line in (root/'usage.jsonl').read_text(encoding='utf-8').splitlines()]):
        raise RuntimeError('No completed native model usage record matching the thread/turn')
    return {'destination':str(root),'receipt':receipt,'validation':validation,
            'reply_path':str(root/'reply.json'),'usage':usage}


def run_review(request_path,destination,*,codex_bin=None,upstream_root=None,vendor_root=None,
               distro='Ubuntu',effort='high',timeout_s=600.):
    """Make an authorized real Astra call. No robot or motor client is created."""
    preflight=prepare_review(request_path,destination)
    if isinstance(timeout_s,bool) or not isinstance(timeout_s,(int,float)) or not math.isfinite(timeout_s) or timeout_s<=0:
        raise ValueError('timeout_s must be finite and positive')
    if effort not in ('low','medium','high','xhigh','max'):raise ValueError('Unsupported Astra effort')
    executable=codex_bin or shutil.which('codex')
    if not executable:raise FileNotFoundError('Codex CLI is required; log in with codex login')
    executable=Path(executable).resolve()
    if not executable.is_file():raise FileNotFoundError(executable)
    repo=Path(upstream_root or _workspace()/'work/upstream/GPT-Policy').resolve()
    vendor=Path(vendor_root or _workspace()/'work/wsl_vendor').resolve()
    worker=Path(__file__).with_name('native_review_worker.py')
    root=Path(preflight['destination']);root.mkdir(parents=True)
    write_json(root/'preflight.json',preflight)
    source=Path(preflight['request_path'])
    request=read_json(source)
    request['input_manifest']=preflight['manifest_path']
    write_json(root/'request.json',request)
    args=[_wsl_path(worker),'--request',_wsl_path(root/'request.json'),'--output',_wsl_path(root),
          '--upstream-root',_wsl_path(repo),'--codex-bin',_wsl_path(executable),
          '--codex-cwd',str(_workspace()/'work') if os.name=='nt' else str(_workspace()/'work'),
          '--effort',effort,'--timeout-s',str(timeout_s)]
    # The native Windows app-server needs a Windows cwd, while the Python
    # harness and its Linux-only camera imports run unmodified inside WSL.
    pythonpath=(':'.join((_wsl_path(repo/'src'),_wsl_path(vendor),_wsl_path(worker.parents[1])))
                if os.name=='nt' else os.pathsep.join((str(repo/'src'),str(vendor),str(worker.parents[1]))))
    if os.name=='nt':
        command=['wsl.exe','-d',distro,'--exec','env','PYTHONPATH='+pythonpath,'python3',*args];env=None
    else:
        command=[sys.executable,*args];env={**os.environ,'PYTHONPATH':pythonpath}
    try:
        result=subprocess.run(command,capture_output=True,env=env,timeout=timeout_s+60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Native Astra review exceeded its deadline; inspect archived receipt before retrying') from exc
    (root/'worker_stdout.log').write_bytes(result.stdout)
    (root/'worker_stderr.log').write_bytes(result.stderr)
    if result.returncode:
        detail=result.stderr.decode('utf-8',errors='replace').replace('\x00','')
        raise RuntimeError('Native GPT-Policy review failed: '+detail[-2500:])
    if (file_sha256(source)!=preflight['request_sha256'] or
        file_sha256(preflight['manifest_path'])!=preflight['manifest_sha256'] or
        any(file_sha256(i['path'])!=i['sha256'] for i in preflight['images']) or
        any(file_sha256(i['path'])!=i['sha256'] for i in preflight['reference_manifests'])):
        raise RuntimeError('Review source changed while inference was running')
    return load_review_result(root)
