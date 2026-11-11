"""Offline real-model benchmark; no actuator imports or calls."""
import json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/upstream/GPT-Policy/src'),str(ROOT/'work/wsl_vendor'),str(ROOT/'outputs/wam_astra/src')]
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.input.manifest import ImagePart, load_manifest
from gpt_policy.harness.waiting import monitor_health
from wam_astra.native_review_worker import configure_native_cwd

effort, executable, cwd=sys.argv[1:4]
out=ROOT/'outputs/wam_astra/h45_takeover'/('continuous_atomic_'+effort)
out.mkdir(exist_ok=True)
schema=json.loads((out.parent/'atomic_plan_20261111/schema.json').read_text())
schema={'type':'object','additionalProperties':False,'properties':{'name':{'type':'string','enum':['atomic_plan']},'arguments':schema},'required':['name','arguments']}
frames=out.parent/'first_chunk_live_01/observation'
content=tuple(ImagePart(frames/(name+'.png'),label=label) for name,label in [('base_0_rgb','dog front'),('left_wrist_0_rgb','arm wrist'),('right_wrist_0_rgb','uav')])
demo=ROOT/'outputs/wam_astra/library/contexts/joint_20261012_155725_UTCp0800_faf6/input.json'
content=load_manifest(demo).content+content
instructions=('Plan early robot positioning to grasp a cube inside a box on an elevated table. '
 'UAV provides overhead vision; Go2 carries a six-joint Piper. Pi handles the later grasp. '
 'Return only the structured atomic decision. Forward is body +x; left is body +y; distance in metres. '
 'Select only one robot per action. Do not invent metric distances from uncalibrated images. '
 'If evidence is insufficient return observe or hold with direction none and null distance. '
 'Keep reason under 25 words. These are offline recorded observations; no robot is executing.')
rows=[]; session=None; start=time.perf_counter()
try:
 with monitor_health(None,deadline=time.monotonic()+180):
  session=CodexSession(AgentConfig(model='gpt-6-astra',executable=executable,effort=effort,decision_only=True),'gpt-6-astra',False,85)
  configure_native_cwd(session,cwd)
  session.start(AgentContext(instructions,[],schema))
  setup=time.perf_counter()-start
  for i in range(3):
   t=time.perf_counter()
   reply=session.decide(AgentTurn('Recorded three-view scene. Give the next atomic action.' if i==0 else 'Same recorded scene, unchanged; no action has executed. Reuse preceding images and give the next action.',content=content if i==0 else ()))
   row=dict(turn=i+1,effort=effort,thread_id=session.thread_id,elapsed_s=time.perf_counter()-t,setup_s=setup,reply=reply,timing=session.last_decision_timing)
   rows.append(row)
   (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
   print(json.dumps(row,ensure_ascii=False),flush=True)
except Exception as exc:
 error=dict(error=str(exc),elapsed_s=time.perf_counter()-start,effort=effort)
 (out/'error.json').write_text(json.dumps(error,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(error),flush=True)
finally:
 if session:session.close()
