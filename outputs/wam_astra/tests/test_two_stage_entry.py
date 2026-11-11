"""Two-stage entry paths must not acquire actuators during offline/plan runs."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hybrid_entry', PROJECT / 'hybrid_live.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_plan_initializes_only_observer_origin():
    device = Mock()
    device.capture.return_value = {'record': 'fresh'}
    with patch('real_robot.controller.Device.call', return_value={'session': 'observe'}) as base_call:
        assert entry.capture_plan_packet(device, {'execution_enabled': False}) == {'record': 'fresh'}
    base_call.assert_called_once_with(device, '/reset', {})
    device.call.assert_not_called()  # paired /reset would start the aircraft bridge
    device.capture.assert_called_once_with(diagnostic=True)


def test_plan_never_resets_an_execution_gateway():
    device = Mock()
    with patch('real_robot.controller.Device.call') as base_call:
        with pytest.raises(RuntimeError, match='observe-only'):
            entry.capture_plan_packet(device, {'execution_enabled': True})
    base_call.assert_not_called()
    device.capture.assert_not_called()


def test_replay_handoff_calls_pi_on_next_capture_only(tmp_path):
    from wam_astra.atomic_motion import HybridPolicy
    captures=[]
    for i in range(3):
        folder=tmp_path / str(i);folder.mkdir();captures.append(folder)
        state=np.zeros(14);state[0]=i
        (folder/'states.json').write_text(json.dumps(dict(model_state14=state.tolist(),prompt='cube',unitree_stamp=10+i,
            available_measurements={'piper':{'joint_1.pos':str(i)}},missing_fields=['camera_extrinsics'])))
        for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'):
            Image.new('RGB',(640,480)).save(folder/(name+'.png'))
    evidence={name:dict(status='supported',evidence='Synthetic protocol evidence',reference_ids=['demo'])
              for name in ('target_visible','base_placement_supported','arm_start_supported')}
    planner=Mock(reference_ids={'demo'});planner.last_receipt={}
    planner.decide.side_effect=[dict(decision='move',robot='go2',direction='forward',distance_m=.1),
                               dict(decision='handoff_pi',handoff_evidence=evidence)]
    pi=Mock();pi.infer_prepared.return_value=np.ones((45,14));pi.last_server_timing={};pi.last_rtc={}
    policy=HybridPolicy(planner,pi)
    report=entry.replay_captures(policy,captures,tmp_path/'run')
    assert report['pi_action_chunks']==1
    assert report['astra_action_chunks']==1
    assert report['handoff_observed'] is True
    assert report['robot_commands_sent'] is False
    assert report['physical_task_success'] is None
    assert pi.prepare_input.call_args.args[0]['state'][0]==2
    assert set(pi.prepare_input.call_args.args[0])=={'state','image','image_mask','prompt'}
    assert pi.prepare_input.call_args.kwargs['rtc']['delay_steps']==0
    assert planner.decide.call_args.args[0]['input_source']=='recorded_capture_model_commands_not_executed'
    assert planner.decide.call_args.args[0]['available_measurements']=={'piper':{'joint_1.pos':'1'}}
    assert planner.decide.call_args.args[0]['missing_fields']==['camera_extrinsics']
    assert np.all(np.load(tmp_path/'run'/'turn_000002'/'actions.npy')==1)


def test_replay_rejects_missing_state(tmp_path):
    (tmp_path/'states.json').write_text(json.dumps(dict(model_state14=None,prompt='cube')))
    with pytest.raises(ValueError, match='state14'):
        entry.load_recorded_capture(tmp_path)
