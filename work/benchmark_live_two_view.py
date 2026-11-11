"""Measure one persistent Astra session on real, recently captured Go2 frames."""
import json, math, sys, time
from pathlib import Path
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.harness.waiting import monitor_health
from gpt_policy.input.manifest import ImagePart, TextPart, load_manifest
from wam_astra.atomic_planner import INSTRUCTIONS, schema
from wam_astra.astra_runtime import astra_runtime_receipt
from wam_astra.native_review_worker import configure_native_cwd

out, reference, executable, cwd = map(Path, sys.argv[1:5])
frames = json.loads((out/'frames.json').read_text())
def clean(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    return x
session = None
results = dict(mode='recent_live_captures_replayed_in_order', robot_commands_sent=False,
    caveat='Two Go2 cameras only; UAV image and state unavailable; no proposed action was executed.', turns=[])
started = time.perf_counter()
try:
    with monitor_health(None, deadline=time.monotonic()+180):
        session=CodexSession(AgentConfig(model='gpt-6-astra',executable=str(executable),effort='low',decision_only=True),'gpt-6-astra',False,85)
        configure_native_cwd(session,str(cwd))
        session.start(AgentContext(INSTRUCTIONS+'\nThis is a latency test on recent live snapshots. The UAV image and state are unavailable; do not infer them. No proposed action was executed.',[],schema()))
    results['session_setup_s']=time.perf_counter()-started
    results['runtime']=astra_runtime_receipt(session)
    reference_content=load_manifest(reference).content
    for i, frame in enumerate(frames):
        folder=out/('frame_%02d'%i)
        content=(reference_content if i==0 else ())+ (TextPart('CURRENT OBSERVATION: dog front and Piper wrist only; UAV unavailable.'), ImagePart(folder/'front.jpg',label='dog front',detail='high'),ImagePart(folder/'wrist.jpg',label='Piper wrist',detail='high'))
        obs=clean(dict(source='recent_live_snapshots_no_robot_actions',capture_stamp=frame['stamp'],raw_sensors=frame['raw_sensors'],sensor_error=frame['sensor_error'],previous_action_executed=False,missing_fields=['UAV image','UAV state','cross-robot transform']))
        start=time.perf_counter()
        with monitor_health(None,deadline=time.monotonic()+180):
            reply=session.decide(AgentTurn(json.dumps(obs,ensure_ascii=False),content=content))
        row=dict(turn=i+1,elapsed_s=time.perf_counter()-start,thread_id=session.thread_id,decision=reply['arguments'],timing=session.last_decision_timing,runtime=astra_runtime_receipt(session))
        results['turns'].append(row)
        (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'turn':i+1,'elapsed_s':row['elapsed_s'],'decision':row['decision'].get('decision')},ensure_ascii=False),flush=True)
except BaseException as exc:
    results['error']={'type':type(exc).__name__,'detail':str(exc)}
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    raise
finally:
    if session:session.close()
