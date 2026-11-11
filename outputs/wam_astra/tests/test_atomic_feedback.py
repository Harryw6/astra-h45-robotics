import math
import unittest
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'h45_takeover/staged/client'),str(PROJECT/'src')]
from wam_astra.atomic_feedback import ActionFeedback


def packet(t,x=0,y=0,yaw=0):
    return dict(stamp=t,state_stamps=[t,t],observation_id=str(t),waiting_for_prediction=True,
        state=[x,y,0,0,0,math.sin(yaw),math.cos(yaw),0,0,0,0,0,0,0,0,0,.03],
        p450_packet=dict(stamp=t+100,pose=dict(x=0,y=0,z=1,yaw=0)))


class FeedbackTests(unittest.TestCase):
    def test_new_camera_stamp_with_reused_native_state_is_not_fresh_motion(self):
        tracker=ActionFeedback();self.submit(tracker,packet(1))
        reused=packet(5,x=.3);reused['state_stamps']=[1,5]
        tracker.observe(reused)
        self.assertEqual(tracker.latest['status'],'awaiting_new_measurement')
        self.assertIsNone(tracker.latest['measured_change']['go2'])
    def submit(self,tracker,initial,accepted=True):
        tracker.submission(packet=initial,action_source='astra_atomic',
            decision=dict(decision='move',robot='go2',direction='forward',distance_m=.3),
            prepared=dict(accepted=True,plan_id='p'),committed=dict(accepted=accepted,plan_id='p'),
            inference_s=8.,transport_s=.1)

    def test_acceptance_never_claims_arrival_and_zero_motion_is_visible(self):
        tracker=ActionFeedback();self.submit(tracker,packet(1))
        self.assertEqual(tracker.latest['status'],'submitted')
        self.assertIsNone(tracker.latest['arrival_confirmed'])
        tracker.observe(packet(5))
        self.assertEqual(tracker.latest['status'],'measured_after_submission')
        self.assertEqual(tracker.latest['measured_change']['go2']['body_xyz_m'],[0.,0.,0.])
        self.assertIsNone(tracker.latest['arrival_confirmed'])

    def test_heading_is_taken_from_original_native_anchor(self):
        tracker=ActionFeedback();self.submit(tracker,packet(1,yaw=math.pi/2))
        tracker.observe(packet(5,y=.3,yaw=math.pi/2))
        self.assertAlmostEqual(tracker.latest['measured_change']['go2']['body_xyz_m'][0],.3)

    def test_rejection_and_pi_do_not_create_executed_action(self):
        tracker=ActionFeedback();self.submit(tracker,packet(1),False)
        tracker.observe(packet(5,x=1))
        self.assertEqual(tracker.latest['status'],'not_submitted')
        tracker.submission(packet=packet(6),action_source='pi_original_h45',decision=None,
            prepared={},committed={},inference_s=0,transport_s=0)
        self.assertEqual(tracker.latest['status'],'not_submitted')

    def test_stale_feedback_is_unknown_and_consumed_feedback_not_reattributed(self):
        tracker=ActionFeedback();self.submit(tracker,packet(1))
        tracker.observe(packet(1,x=5))
        self.assertEqual(tracker.latest['status'],'awaiting_new_measurement')
        self.assertIsNone(tracker.latest['measured_change']['go2'])
        tracker.observe(packet(5,x=.3))
        first=tracker.latest
        tracker.observe(packet(9,x=1))
        self.assertEqual(tracker.latest,first)

if __name__=='__main__':unittest.main()
