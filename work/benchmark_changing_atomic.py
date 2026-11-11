"""Six successive real recorded observations, one persistent GPT-Policy session."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'work/upstream/GPT-Policy/src'),str(ROOT/'work/wsl_vendor'),str(ROOT/'outputs/wam_astra/src')]
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.models import AgentContext,AgentTurn
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.harness.waiting import monitor_health
from gpt_policy.input.manifest import load_manifest,ImagePart,TextPart
from wam_astra.native_review_worker import configure_native_cwd,local_path

out=ROOT/'outputs/wam_astra/h45_takeover'/(sys.argv[3] if len(sys.argv)>3 else 'changing_atomic_20261111')
data=json.loads((out/'observations.json').read_text())
demo=Path(sys.argv[4]) if len(sys.argv)>4 else ROOT/'outputs/wam_astra/library/contexts/joint_20261012_155725_UTCp0800_faf6/input.json'
context=load_manifest(demo).content
schema=json.loads((out.parent/'atomic_plan_20261111/schema.json').read_text())
schema['properties']['decision']['enum'].append('handoff_pi')
schema['properties']['direction']['enum']+=['up','down']
schema={'type':'object','additionalProperties':False,'properties':{'name':{'type':'string','enum':['atomic_plan']},'arguments':schema},'required':['name','arguments']}
instructions=('You plan early UAV/Go2 positioning; Pi performs the later Piper cube grasp. '
 'Goal: obtain overhead view of cube inside box on elevated table, position Go2 for grasp, then handoff_pi. '
 'Output one atomic_plan: robot, direction, distance_m, brief reason. Forward=body +x, left=body +y; up=+z for UAV only. '
 'Select move if evidence supports direction and metric distance; an exploratory increment is allowed but explicitly identify it as exploratory, not measured target distance. '
 'Do not invent camera-to-body extrinsics or cross-robot frame transforms. Positions are each robot local odometry, not a shared frame. '
 'If direction or scale cannot be supported choose observe/hold with none direction and null distance. '
 'handoff_pi means the recorded scene is ready for arm manipulation, not successful grasp; use robot none, direction none, null distance. '
 'Treat demo as a different past episode, not current observation. Current frames are an offline sequence driven by recorded actions, NOT your previous commands. '
 'No robots execute during this benchmark. Three image panels left-to-right: dog front, arm wrist, UAV. '
 'Use newest image as current scene. Reason under 30 words. Return only the schema.')
(out/'instructions.txt').write_text(instructions)
rows=[];session=None;start=time.perf_counter()
try:
 with monitor_health(None,deadline=time.monotonic()+500):
  session=CodexSession(AgentConfig(model='gpt-6-astra',executable=sys.argv[1],effort='low',decision_only=True),'gpt-6-astra',False,85)
  configure_native_cwd(session,sys.argv[2])
  session.start(AgentContext(instructions,[],schema))
  setup=time.perf_counter()-start
  limit=int(sys.argv[5]) if len(sys.argv)>5 else len(data['observations'])
  for i,frame in enumerate(data['observations'][:limit]):
   state={}
   fields={'go2':['relative_time_s','position_x','position_y','position_z','yaw','velocity_x','velocity_y','body_height'],
     'uav':['relative_time_s','position.0','position.1','position.2','attitude.2','velocity.0','velocity.1','velocity.2','armed','mode'],
     'piper':['relative_time_s',*[f'joint_{j}.pos' for j in range(1,7)],'gripper.pos']}
   for k,keys in fields.items():state[k]={key:frame['measured'][k].get(key) for key in keys} if frame['measured'][k] else None
   observation=dict(episode=data['episode_id'],time_s=frame['time_s'],measured=state,previous_action_executed=False,camera_extrinsics='not supplied')
   content=(context if i==0 else ())+ (TextPart('CURRENT RECORDED OBSERVATION at %.3fs'%frame['time_s']),ImagePart(local_path(frame['image'],out),label='current three-view frame',detail='high'))
   begin=time.perf_counter()
   reply=session.decide(AgentTurn(json.dumps(observation),content=content))
   row=dict(turn=i+1,time_s=frame['time_s'],image_sha256=frame['image_sha256'],elapsed_s=time.perf_counter()-begin,setup_s=setup,thread_id=session.thread_id,reply=reply['arguments'],timing=session.last_decision_timing)
   rows.append(row)
   (out/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
   print(json.dumps(row,ensure_ascii=False),flush=True)
except Exception as exc:
 (out/'error.json').write_text(json.dumps(dict(error=str(exc),elapsed_s=time.perf_counter()-start)),encoding='utf-8')
 raise
finally:
 if session:session.close()
