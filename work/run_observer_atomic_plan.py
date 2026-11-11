"""Run one real Astra turn on a saved measured three-view observer capture.

Does not instantiate robot clients, submit chunks or invoke Pi.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'outputs/wam_astra/src'),str(ROOT/'outputs/wam_astra/h45_takeover/staged/client'),str(ROOT/'outputs/H45_live_repro_20261107')]
from wam_astra.atomic_motion import HybridPolicy
from wam_astra.atomic_planner import AtomicPlannerClient
from wam_astra.atomic_handoff import evaluate_handoff_evidence

p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
metadata=json.loads((a.capture/'states.json').read_text(encoding='utf-8'))
partial=metadata['model_state14'] is None
obs=dict(state=None if partial else np.asarray(metadata['model_state14']),prompt=metadata['prompt'],image={},
    available_measurements=metadata.get('available_measurements'),missing_fields=metadata.get('missing_fields',[]),
    input_source='saved_read_only_live_capture_no_actions_executed')
for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'):
    obs['image'][name]=np.asarray(Image.open(a.capture/(name+'.png')).convert('RGB'))
packet=None if partial else dict(state=metadata['go2_state']+metadata['piper_state'],stamp=metadata['unitree_stamp'],
    state_stamps=metadata.get('state_stamps'),observation_id=metadata.get('observation_id'),
    p450_packet=metadata['p450_state'],waiting_for_prediction=False)
started=time.perf_counter();client=AtomicPlannerClient(ROOT,a.reference,a.output/'astra')
try:
    if partial:
        action=client.decide(obs);proposed=None;phase='plan_only_incomplete_native_state'
        assessment=evaluate_handoff_evidence(action.get('handoff_evidence'),allowed_reference_ids=client.reference_ids) if action['decision']=='handoff_pi' else None
    else:
        policy=HybridPolicy(client,None);policy.observe_packet(packet)
        proposed=policy.infer_prepared(policy.prepare_input(obs))
        action=policy.last_decision;assessment=policy.last_handoff_assessment;phase=policy.phase
    result=dict(action=action,handoff_assessment=assessment,
        phase=phase,proposed_chunk_shape=list(proposed.shape) if proposed is not None else None,
        timing=client.last_receipt,wall_elapsed_s=time.perf_counter()-started,
        robot_commands_sent=False,pi_called=False,capture=str(a.capture.resolve()))
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)
finally:client.close()
