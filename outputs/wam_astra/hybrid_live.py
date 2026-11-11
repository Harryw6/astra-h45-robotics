"""Astra positioning -> fresh-observation Pi grasp using existing H45 execution."""
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT=Path(__file__).resolve().parent
WORKSPACE=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'h45_takeover/staged/client'),
             str(WORKSPACE/'outputs/H45_live_repro_20261107')]


def capture_plan_packet(device, status):
    """Initialize the observation origin without the paired flight begin call."""
    from real_robot.controller import Device
    if status.get('execution_enabled') is not False:
        raise RuntimeError('Plan mode requires an observe-only gateway')
    Device.call(device, '/reset', {})
    return device.capture(diagnostic=True)


def load_recorded_capture(folder):
    import numpy as np
    from PIL import Image
    folder=Path(folder)
    metadata=json.loads((folder/'states.json').read_text(encoding='utf-8'))
    state=np.asarray(metadata.get('model_state14'),dtype=np.float32)
    if state.shape!=(14,) or not np.isfinite(state).all():
        raise ValueError('Recorded capture requires complete finite state14: '+str(folder))
    images={}
    for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'):
        with Image.open(folder/(name+'.png')) as image:
            images[name]=np.asarray(image.convert('RGB'))
        if images[name].shape!=(480,640,3):
            raise ValueError('Recorded camera must retain 640x480 RGB: '+name)
    observation=dict(state=state,image=images,image_mask={k:np.bool_(True) for k in images},
        prompt=metadata['prompt'],input_source='recorded_capture_model_commands_not_executed',
        previous_action_executed=False,available_measurements=metadata.get('available_measurements'),
        missing_fields=metadata.get('missing_fields',[]))
    return observation,metadata


def replay_captures(policy, captures, output):
    """Run the real policy transition on saved inputs; never acquire a robot client."""
    import numpy as np
    from real_robot.goal_distance_realtime import save
    from real_robot.rtc_h45 import empty_context
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(mode='replay',robot_commands_sent=False,physical_task_success=None,
        handoff_observed=False,astra_action_chunks=0,pi_action_chunks=0,turns=[],
        sequence_notice='Recorded changes are independent of model proposals, not a closed-loop rollout.')
    previous_stamp=None
    for index,capture in enumerate(captures):
        observation,metadata=load_recorded_capture(capture)
        stamp=float(metadata['unitree_stamp'])
        if not np.isfinite(stamp) or previous_stamp is not None and stamp<=previous_stamp:
            raise ValueError('Replay captures must have strictly increasing recorded timestamps')
        previous_stamp=stamp
        folder=output/('turn_%06d'%index);folder.mkdir()
        phase_before=policy.phase;started=time.perf_counter()
        actions=policy.infer_prepared(policy.prepare_input(observation,rtc=empty_context()))
        row=dict(capture=str(Path(capture).resolve()),phase_before=phase_before,phase_after=policy.phase,
            elapsed_s=time.perf_counter()-started,action_source=policy.last_action_source,
            decision=policy.last_decision if phase_before=='positioning' else None,
            handoff_assessment=policy.last_handoff_assessment if phase_before=='positioning' else None,
            robot_commands_sent=False)
        if phase_before!=policy.phase:report['handoff_observed']=True
        if actions is not None:
            np.save(folder/'actions.npy',actions)
            row['action_shape']=list(actions.shape)
            if policy.last_action_source=='astra_atomic':report['astra_action_chunks']+=1
            else:
                report['pi_action_chunks']+=1
                row['server_timing']=policy.pi.last_server_timing
                row['rtc']=policy.pi.last_rtc
        else:row['action_shape']=None
        if phase_before=='positioning':row['astra_receipt']=policy.planner.last_receipt
        save(folder/'result.json',row)
        report['turns'].append(row)
        save(output/'summary.json',report)
        print(json.dumps({k:row[k] for k in ('phase_before','phase_after','elapsed_s','action_source','action_shape')},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['standby','plan','replay','execute'])
    p.add_argument('--config',type=Path,default=PROJECT/'h45_takeover/atomic_deployment_20261111/client.atomic.json')
    p.add_argument('--output',type=Path,default=PROJECT/'h45_takeover/runtime'/('two_stage_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    p.add_argument('--reference',type=Path,default=PROJECT/'two_stage_reference_20261111/input.json')
    p.add_argument('--capture',type=Path,action='append',help='Recorded capture directory; repeat in chronological order for replay')
    a=p.parse_args()
    if a.mode=='replay' and not a.capture:p.error('Replay requires --capture directories')
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    from real_robot import three_robot
    from real_robot.live_h45 import LiveDevice,execute_live,model_observation,save_capture
    from real_robot.goal_distance_realtime import RemotePolicy,save
    from wam_astra.atomic_motion import HybridPolicy
    from wam_astra.atomic_planner import AtomicPlannerClient
    aircraft=planner=pi=device=None
    try:
        # Model setup never takes ownership of motors. Keep it outside the control loop.
        planner=AtomicPlannerClient(WORKSPACE,a.reference,output/'astra')
        if a.mode!='plan':
            pi=RemotePolicy(config['policy_url'],checkpoint_step=config['checkpoint_step'])
        if pi and (pi.metadata.get('action_names',[])[10:]!=three_robot.P450_ACTIONS
                or pi.metadata.get('uav_encoding')!='target_pose_relative_to_measured_chunk_start'
                or pi.metadata.get('uav_xy_frame')!='chunk_start_yaw_forward_left'
                or pi.metadata.get('fps')!=15):
            raise ValueError('Pi metadata does not match existing H45 execution frames')
        policy=HybridPolicy(planner,pi)
        save(output/'models.json',dict(astra=planner.ready,pi_connected=pi is not None,
            checkpoint_step=config['checkpoint_step'],reference=str(a.reference.resolve())))
        if a.mode=='replay':
            replay_captures(policy,a.capture,output/'replay')
            return
        three_robot.BRIDGE_PATH=config['p450_bridge_path']
        aircraft=three_robot.AircraftClient(argparse.Namespace(**config['aircraft']),output/'p450.jsonl')
        device=LiveDevice(config['unitree_url'],Path(config['token_file']).read_text().strip(),aircraft,
                          ground_start=True,no_speed_limit=config['no_speed_limit'])
        status=device.calibrate()
        if status.get('review',{}).get('configured'):
            raise RuntimeError('Use the atomic gateway configuration: legacy review-pause mode replaces native wait/resume')
        prompt=config.get('prompt','Use UAV overhead vision to position Go2 for grasping the cube inside the elevated tabletop box. Then Pi handles the Piper grasp.')
        if a.mode=='standby':
            save(output/'summary.json',dict(mode='standby',astra=planner.ready,pi_connected=True,
                 execution_enabled=status['execution_enabled'],robot_commands_sent=False))
        elif a.mode=='plan':
            packet=capture_plan_packet(device,status)
            obs=model_observation(packet,packet['p450_packet'],prompt)
            save_capture(output/'observation',packet,obs)
            planner.observe_packet(packet)
            decision=planner.decide(obs)
            save(output/'summary.json',dict(mode='plan',decision=decision,receipt=planner.last_receipt,robot_commands_sent=False))
        else:
            save(output/'summary.json',execute_live(device,policy,prompt,0,output))
    except BaseException as exc:
        save(output/'error.json',dict(mode=a.mode,error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        if device and device.plane_started:device.stop_plane()
        if aircraft:aircraft.close()
        if pi:pi.close()
        if planner:planner.close()


if __name__=='__main__':main()
