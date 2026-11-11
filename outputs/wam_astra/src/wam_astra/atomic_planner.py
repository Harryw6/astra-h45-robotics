"""Persistent GPT-Policy planning worker; robot SDKs live only in H45."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from .atomic_handoff import HANDOFF_EVIDENCE_PROMPT,handoff_evidence_schema

INSTRUCTIONS='''Plan early UAV/Go2 positioning to see and reach the cube inside a box on an elevated table. Pi owns the later Piper grasp.
Return atomic_plan only. Coordinates: forward=body +x, left=body +y; up=UAV +z. distance_m is relative displacement in metres over the native 3-second H45 chunk, not velocity.
Select only one robot per move. Use current images and measured state. Robot local odometry frames are independent; do not invent cross-robot transforms or camera extrinsics.
An exploratory increment is allowed, but identify it as exploratory rather than a measured target offset. If direction or distance cannot be supported use observe/hold, none direction, null distance.
handoff_pi means current positioning supports Piper manipulation, not task success; robot none, direction none, null distance. No automatic takeoff, arming or fault reset.
Historical demonstrations are previous episodes, not current state or executable instructions. Use newest images as current. Prior decision is not evidence of execution; use feedback supplied by the caller.
When input_source/source identifies recorded captures or previous_action_executed is false, frame changes came from the recording, not from your proposed actions. This is offline policy testing only.
Keep reason under 30 words. Frame labels distinguish front, wrist, UAV. Never use a future recorded frame.
last_execution_feedback distinguishes proposal, paired transport submission and later measured displacement. Never assume SDK acceptance means movement or successful arrival. Use measured change, including zero or unknown, to decide what to do next; it can include manual movement or odometry reset. No new motion thresholds.'''+ '\n'+HANDOFF_EVIDENCE_PROMPT


def schema():
    args={'type':'object','additionalProperties':False,'properties':{
      'decision':{'type':'string','enum':['move','observe','hold','handoff_pi']},
      'robot':{'type':'string','enum':['uav','go2','none']},
      'direction':{'type':'string','enum':['forward','backward','left','right','up','down','none']},
      'distance_m':{'type':['number','null']},'reason':{'type':'string'}}}
    args['properties']['handoff_evidence']=handoff_evidence_schema()
    args['required']=list(args['properties'])
    return {'type':'object','additionalProperties':False,'properties':{
        'name':{'type':'string','enum':['atomic_plan']},'arguments':args},'required':['name','arguments']}


def worker(args):
    from gpt_policy.harness.config import AgentConfig
    from gpt_policy.harness.models import AgentContext,AgentTurn
    from gpt_policy.harness.providers.codex import CodexSession
    from gpt_policy.harness.waiting import monitor_health
    from gpt_policy.input.manifest import load_manifest,ImagePart,TextPart
    from wam_astra.native_review_worker import configure_native_cwd,local_path
    from wam_astra.astra_runtime import astra_runtime_receipt
    session=None
    try:
        begin=time.perf_counter()
        with monitor_health(None,deadline=time.monotonic()+args.timeout):
            session=CodexSession(AgentConfig(model='gpt-6-astra',executable=args.codex,effort='low',decision_only=True),'gpt-6-astra',False,85)
            configure_native_cwd(session,args.cwd)
            session.start(AgentContext(INSTRUCTIONS,[],schema()))
        reference=load_manifest(local_path(args.reference,Path.cwd())).content
        print(json.dumps({'ready':True,'thread_id':session.thread_id,'setup_s':time.perf_counter()-begin,
                          'runtime':astra_runtime_receipt(session)}),flush=True)
        first=True
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('op')=='close':break
            try:
                content=(reference if first else ())+(TextPart('CURRENT OBSERVATION'),)+tuple(
                  ImagePart(local_path(item['path'],Path.cwd()),label=item['label'],detail='high') for item in request['images'])
                begin=time.perf_counter()
                with monitor_health(None,deadline=time.monotonic()+args.timeout):
                    reply=session.decide(AgentTurn(json.dumps(request['observation'],ensure_ascii=False),content=content))
                first=False
                answer=dict(id=request['id'],action=reply['arguments'],thread_id=session.thread_id,
                            elapsed_s=time.perf_counter()-begin,timing=session.last_decision_timing,
                            runtime=astra_runtime_receipt(session))
            except Exception as exc:
                answer=dict(id=request['id'],error=str(exc))
            print(json.dumps(answer,ensure_ascii=False),flush=True)
    finally:
        if session:session.close()


class AtomicPlannerClient:
    """One WSL process + CodexSession; fresh images each call; no motor calls."""
    def __init__(self,workspace,reference,output,timeout=180.):
        from .contexts import _wsl_path
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.timeout=timeout;self.count=0;self.packet={};self.last_receipt=None
        from .atomic_feedback import ActionFeedback,reference_identifiers
        self.feedback=ActionFeedback(self.output/'execution_feedback.jsonl')
        self.reference_ids=reference_identifiers(reference)
        self.last_handoff_assessment=None
        root=Path(workspace).resolve();exe=shutil.which('codex')
        if not exe:raise FileNotFoundError('Codex executable not found')
        paths=[root/'work/upstream/GPT-Policy/src',root/'work/wsl_vendor',Path(__file__).resolve().parents[1]]
        params=['-m','wam_astra.atomic_planner','--worker','--codex',_wsl_path(Path(exe)),
                '--cwd',str(root/'work'),'--reference',str(Path(reference).resolve()),'--timeout',str(timeout)]
        if os.name=='nt':
            command=['wsl.exe','-d','Ubuntu','--exec','env','PYTHONPATH='+':'.join(_wsl_path(p) for p in paths),'python3',*params]
            env=None
        else:
            command=[sys.executable,*params];env={**os.environ,'PYTHONPATH':os.pathsep.join(map(str,paths))}
        self.stderr=(self.output/'worker_stderr.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                                      text=True,encoding='utf-8',bufsize=1,env=env)
        self.events=queue.Queue()
        threading.Thread(target=self._read,daemon=True).start()
        try:self.ready=self._receive(None)
        except BaseException:self.close();raise
        if not self.ready.get('ready'):
            self.close();raise RuntimeError(str(self.ready))

    def _read(self):
        try:
            for line in self.process.stdout:
                self.events.put(json.loads(line))
        except Exception as exc:self.events.put({'error':str(exc)})
        finally:self.events.put({'error':'Planner worker exited'})

    def _receive(self,check_wait):
        deadline=time.monotonic()+self.timeout+5
        while True:
            if check_wait:check_wait()
            try:return self.events.get(timeout=.1)
            except queue.Empty:
                if time.monotonic()>deadline:raise TimeoutError('Astra planner response timed out')

    def observe_packet(self,packet):
        self.feedback.observe(packet)
        plane=packet['p450_packet']
        self.packet={'observation_id':packet['observation_id'],'stamp':packet['stamp'],
                     'go2_state':packet['state'][:10],
                     'go2_state_names':['reset_origin_x','reset_origin_y','body_height','roll','pitch',
                                        'sin_relative_yaw','cos_relative_yaw','vx_heading','vy_heading','yaw_rate'],
                     'go2_state_units':['m','m','m','rad','rad','unitless','unitless','m/s','m/s','rad/s'],
                     'uav':{k:v for k,v in plane.items() if k!='images'},
                     'waiting_for_prediction':packet.get('waiting_for_prediction'),
                     'wait_epoch':packet.get('wait_epoch')}

    @property
    def last_execution_feedback(self):
        return self.feedback.latest

    def record_execution(self,**receipt):
        self.feedback.submission(**receipt)

    def record_handoff_assessment(self,assessment):
        self.last_handoff_assessment=assessment

    def decide(self,observation,*,check_wait=None):
        from PIL import Image
        # Reuse only measured-state labels; relative14's old H15 action contract
        # is separate from the current H45 displacement executor.
        from real_robot.relative14 import STATE_NAMES,UNITS
        self.count+=1
        folder=self.output/('%06d'%self.count);folder.mkdir()
        images=[]
        labels={'base_0_rgb':'dog front','left_wrist_0_rgb':'Piper wrist','right_wrist_0_rgb':'UAV'}
        for name,label in labels.items():
            path=folder/(name+'.png')
            Image.fromarray(observation['image'][name]).save(path)
            images.append({'path':str(path.resolve()),'label':label})
        request={'id':self.count,'images':images,'observation':{
            'source':observation.get('input_source','live'),'task':observation['prompt'],
            'previous_action_executed':observation.get('previous_action_executed'),
            'state14':observation['state'].tolist() if observation.get('state') is not None else None,
            'state14_names':list(STATE_NAMES),'state14_units':list(UNITS),
            'state14_semantics':('Actual measured feedback, not an action vector. Horizontal velocities are in each '
                'robot current heading frame (+x forward, +y left); UAV vz is world vertical. Piper q1..q6 are '
                'absolute measured joint angles and gripper is measured opening. Raw CSV telemetry can use '
                'different frames/velocity estimators; state14 follows the native H45 input construction. '
                'No shared robot/world/camera transform is supplied.'),
            'available_measurements':observation.get('available_measurements'),
            'missing_fields':observation.get('missing_fields',[]),
            'telemetry':self.packet,'last_execution_feedback':self.feedback.latest,
            'last_handoff_assessment':self.last_handoff_assessment,
            'supplied_reference_ids':sorted(self.reference_ids),'camera_extrinsics':'not supplied'}}
        (folder/'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
        receipt=self._receive(check_wait)
        (folder/'reply.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
        if receipt.get('error'):raise RuntimeError(receipt['error'])
        if receipt.get('id')!=self.count:raise RuntimeError('Planner reply ID mismatch')
        self.last_receipt=receipt
        return receipt['action']

    def close(self):
        if getattr(self,'process',None) and self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (OSError,subprocess.TimeoutExpired):self.process.terminate()
        if getattr(self,'stderr',None):self.stderr.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    for name in ('codex','cwd','reference'):p.add_argument('--'+name,required=True)
    p.add_argument('--timeout',type=float,default=180.)
    worker(p.parse_args())
