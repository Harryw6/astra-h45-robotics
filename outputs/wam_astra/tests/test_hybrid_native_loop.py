"""Exercise original H45 loop with simulated transports; no hardware connection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'h45_takeover/staged/client'),str(PROJECT/'src')]
from real_robot.live_h45 import execute_live
from real_robot.rtc_h45 import pipeline_prefix_context
from wam_astra.atomic_motion import HybridPolicy
from wam_astra.atomic_feedback import ActionFeedback


class NativeHybridTests(unittest.TestCase):
    def test_astra_wait_does_not_inflate_second_pi_prefix(self):
        evidence={key:dict(status='supported',evidence='Synthetic pre-grasp state.',reference_ids=['demo'])
                  for key in ('target_visible','base_placement_supported','arm_start_supported')}
        planner=Mock(reference_ids={'demo'})
        planner.decide.side_effect=[dict(decision='move',robot='go2',direction='forward',distance_m=.1),
                                   dict(decision='handoff_pi',handoff_evidence=evidence)]
        planner.last_execution_feedback=None
        pi=Mock();pi.infer_prepared.return_value=np.ones((45,14))*.01
        policy=HybridPolicy(planner,pi)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            class Device:
                paired=0;plane_stop=None
                aircraft=Mock()
                def __init__(self):self.capture_times=iter([10.,22.,23.,23.8])
                def wait_for_operator_ready(self):pass
                def abort(self):pass
                def capture(self):
                    stamp=next(self.capture_times);state=np.zeros(17);state[6]=1
                    return dict(stamp=stamp,state=state.tolist(),state_stamps=[stamp]*2,
                        observation_id=str(stamp),waiting_for_prediction=self.paired==1,
                        p450_packet=dict(stamp=stamp,pose=dict(x=0,y=0,z=1,yaw=0)))
                def call(self,endpoint,body=None):
                    if endpoint=='/reset':return {'session':'same-session'}
                    if endpoint=='/status':return dict(capabilities=['h45_live_continuous_v1'],
                        stream={'waiting_for_prediction':self.paired==1})
                    if endpoint=='/stream/prepare':
                        self.body=body
                        return {'accepted':True,'plan_id':str(self.paired+1)}
                    if endpoint=='/stream/commit':
                        self.paired+=1
                        if self.paired==3:(out/'operator_completed.request').write_text('done')
                        delay=8. if self.paired==1 else .7
                        return dict(accepted=True,end_at=None,origin_at=self.body['origin_at'],
                            scheduled_start=self.body['origin_at']+delay,plan_id=body['plan_id'])
            device=Device();device.aircraft.call.side_effect=[{'live_continuous_supported':True},{'landed':2}]
            obs=dict(state=np.zeros(14),image={},prompt='cube')
            with patch('real_robot.live_h45.StreamHeartbeat'),patch('real_robot.live_h45.model_observation',return_value=obs),patch('real_robot.live_h45.save_capture',side_effect=lambda folder,*args:folder.mkdir()),patch('real_robot.live_h45.pipeline_prefix_context',wraps=pipeline_prefix_context) as prefix:
                result=execute_live(device,policy,'cube',0,out)
            self.assertEqual(result['paired_commits'],3)
            self.assertEqual(pi.infer_prepared.call_count,2)
            # 0.7 s Pi arrival plus one 15 Hz dispatch period -> 12 steps.
            # The previous 8 s Astra request must never enter this estimate.
            self.assertEqual(prefix.call_count,1)
            self.assertEqual(prefix.call_args.args[4],12)
            self.assertEqual(pi.prepare_input.call_args_list[0].kwargs['rtc']['delay_steps'],0)

    def test_observe_move_handoff_pi_preserves_one_session(self):
        evidence={key:dict(status='supported',evidence='Synthetic comparison for transport test.',reference_ids=['test-demo'])
                  for key in ('target_visible','base_placement_supported','arm_start_supported')}
        decisions=iter([dict(decision='observe'),dict(decision='move',robot='uav',direction='up',distance_m=.39),
                        dict(decision='handoff_pi'),dict(decision='handoff_pi',handoff_evidence=evidence)])
        pi=Mock();pi.infer_prepared.return_value=np.ones((45,14))*.01
        class Planner:
            reference_ids={'test-demo'}
            def __init__(self):self.tracker=ActionFeedback();self.feedback_seen=[]
            @property
            def last_execution_feedback(self):return self.tracker.latest
            def observe_packet(self,p):self.tracker.observe(p)
            def record_execution(self,**receipt):self.tracker.submission(**receipt)
            def decide(self,obs,check_wait=None):
                self.feedback_seen.append(self.tracker.latest)
                return next(decisions)
        planner=Planner();policy=HybridPolicy(planner,pi)
        chunks=[];captures=[];events=[]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            class Device:
                paired=0;plane_stop=None
                aircraft=Mock()
                def wait_for_operator_ready(self):pass
                def abort(self):events.append('operator_stop')
                def capture(self):
                    i=len(captures);captures.append(i)
                    state=np.zeros(17);state[6]=1
                    return dict(stamp=float(i+10),state_stamps=[float(i+10)]*2,state=state.tolist(),observation_id=str(i),
                        p450_packet=dict(stamp=float(i+100),pose=dict(x=0,y=0,z=1+i*.1,yaw=0)),waiting_for_prediction=bool(chunks))
                def call(self,endpoint,body=None):
                    events.append(endpoint)
                    if endpoint=='/reset':return {'session':'one'}
                    if endpoint=='/status':return {'stream':{'waiting_for_prediction':True}}
                    if endpoint=='/stream/prepare':
                        self.body=body;chunks.append(body)
                        return {'accepted':True,'plan_id':str(len(chunks))}
                    if endpoint=='/stream/commit':
                        self.paired+=1
                        if self.paired==2:(out/'operator_completed.request').write_text('done')
                        return dict(accepted=True,end_at=None,origin_at=self.body['origin_at'],scheduled_start=self.body['origin_at']+.1,plan_id=body['plan_id'])
            device=Device();device.aircraft.call.side_effect=[{'live_continuous_supported':True},{'landed':2}]
            original_call=device.call
            def call(endpoint,body=None):
                result=original_call(endpoint,body)
                if endpoint=='/status':result['capabilities']=['h45_live_continuous_v1']
                return result
            device.call=call
            obs=dict(state=np.zeros(14),image={},prompt='cube')
            with patch('real_robot.live_h45.StreamHeartbeat'),patch('real_robot.live_h45.model_observation',return_value=obs),patch('real_robot.live_h45.save_capture',side_effect=lambda folder,*args:folder.mkdir()):
                result=execute_live(device,policy,'cube',0,out)
            self.assertEqual(result['paired_commits'],2)
            self.assertEqual(events.count('/reset'),1)
            self.assertEqual(len(captures),5)
            self.assertEqual(planner.feedback_seen[2]['status'],'measured_after_submission')
            self.assertAlmostEqual(planner.feedback_seen[2]['measured_change']['uav']['body_xyz_m'][2],.1)
            self.assertIsNone(planner.feedback_seen[2]['arrival_confirmed'])
            self.assertAlmostEqual(chunks[0]['actions'][-1][12],.39)
            self.assertTrue(chunks[0]['bootstrap'])
            self.assertTrue(chunks[1]['resume_from_wait'])
            last=np.asarray(chunks[1]['actions'])
            self.assertTrue(np.all(last==.01))
            self.assertEqual(events[-1],'operator_stop')
            self.assertEqual(policy.phase,'grasp')

if __name__=='__main__':unittest.main()
