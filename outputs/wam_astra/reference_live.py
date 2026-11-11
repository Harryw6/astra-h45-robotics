"""Assemble the single-Piper reference memory and the existing H45 live loop.

The SDKs, policy server and robot gateways remain the execution implementations.
This file only supplies their explicit configuration and connects their APIs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parent
WORKSPACE=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'h45_takeover/staged/client'),
              str(WORKSPACE/'outputs/H45_live_repro_20261107')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True,help='Site-supplied JSON configuration')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--standby',action='store_true',help='Initialize clients and configuration, without starting an execution session')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    from real_robot import three_robot
    from real_robot.review_device import ReviewDevice
    from real_robot.live_h45 import execute_live
    from real_robot.goal_distance_realtime import RemotePolicy,save
    from wam_astra.memory import TaskMemory
    from wam_astra.live_review_job import LiveMemoryReviewJob
    from wam_astra.checkpoint_hooks import checkpoint_runtime

    # Use the deployed bridge, keeping the original H45 application intact.
    three_robot.BRIDGE_PATH=config['p450_bridge_path']
    aircraft=three_robot.AircraftClient(argparse.Namespace(**config['aircraft']),output/'p450.jsonl')
    policy=None
    try:
        dog=ReviewDevice(config['unitree_url'],Path(config['token_file']).read_text().strip(),aircraft,
            ground_start=True,no_speed_limit=config['no_speed_limit'],
            review_rpc_timeout_s=config['review_rpc_timeout_s'],
            deadline_clock_max_age_s=config['deadline_clock_max_age_s'],
            deadline_clock_margin_s=config['deadline_clock_margin_s'])
        startup_deadline=time.monotonic()+8.
        while True:
            try:
                dog.calibrate()
                break
            except RuntimeError as exc:
                if ('waiting for state/control/extended/RC telemetry' not in str(exc)
                        or time.monotonic()>=startup_deadline):raise
                time.sleep(.2)
        policy=RemotePolicy(config['policy_url'],checkpoint_step=config['checkpoint_step'])
        memory=TaskMemory(PROJECT/'library',PROJECT/'task_memory')
        job=LiveMemoryReviewJob(memory,config['task_id'],output/'astra_jobs',
            clock_snapshot=dog.review_clock_snapshot,reference_strategy='roboicl_single_arm',
            max_capture_age_s=config['max_capture_age_s'],max_clock_age_s=config['max_clock_age_s'],
            model_timeout_s=config['model_timeout_s'],effort=config['effort'],
            recovery_reserve_s=config['recovery_reserve_s'],
            multiview_inspection_root=config.get('multiview_inspection_root'))
        def audit(event):
            with (output/'review_events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        runtime=checkpoint_runtime(dog,job,config['checkpoint_thresholds'],
            poll_interval_s=config['monitor_period_s'],review_timeout_s=config['review_timeout_s'],audit=audit)
        if args.standby:
            status=dog.call('/status')
            plane=aircraft.call('ground_wait_status',timeout=5)
            result=dict(status='initialized_standby',execution_enabled=status['execution_enabled'],
                review_configured=status['review']['configured'],piper_feedback=status.get('piper_feedback'),
                go2_standing=status.get('go2_standing'),aircraft=plane,model_connected=True,
                reference_strategy=job.reference_strategy,actuator_plans_sent=0,
                physical_takeover_tested=False)
            save(output/'summary.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return
        task=memory.task(config['task_id'])
        result=execute_live(dog,policy,task['instruction'],0,output,review_runtime=runtime)
        save(output/'summary.json',result)
    finally:
        if policy:policy.close()
        aircraft.close()


if __name__=='__main__':main()
