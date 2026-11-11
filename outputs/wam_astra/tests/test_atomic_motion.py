import unittest
import numpy as np
from wam_astra.atomic_motion import atomic_chunk, HybridPolicy

class AtomicTests(unittest.TestCase):
 def test_body_axes_and_arm_hold(self):
  state=np.zeros(14);state[9]=.04
  for robot,col in [('go2',1),('uav',11)]:
   chunk=atomic_chunk(dict(decision='move',robot=robot,direction='left',distance_m=.3),state)
   self.assertAlmostEqual(chunk[-1,col],.3)
   self.assertTrue(np.all(chunk[:,3:9]==0))
   self.assertTrue(np.all(chunk[:,9]==.04))
   self.assertAlmostEqual(np.diff(np.r_[0,chunk[:,col]]).sum(),.3)
 def test_observe_never_executes(self):
  self.assertIsNone(atomic_chunk(dict(decision='observe'),np.zeros(14)))
 def test_uav_vertical(self):
  chunk=atomic_chunk(dict(decision='move',robot='uav',direction='up',distance_m=.39),np.zeros(14))
  self.assertAlmostEqual(chunk[-1,12],.39)
  with self.assertRaises(ValueError):
   atomic_chunk(dict(decision='move',robot='go2',direction='up',distance_m=.39),np.zeros(14))
 def test_observe_is_not_a_stop_exception(self):
  policy=HybridPolicy(lambda obs:dict(decision='observe',reason='need view'),None)
  self.assertIsNone(policy.infer_prepared(policy.prepare_input({'state':np.zeros(14)})))
  self.assertEqual(policy.phase,'positioning')
 def test_handoff_requires_new_observation(self):
  evidence={key:dict(status='supported',evidence='Synthetic current-to-demo comparison.',reference_ids=['synthetic-demo'])
            for key in ('target_visible','base_placement_supported','arm_start_supported')}
  policy=HybridPolicy(lambda obs:dict(decision='handoff_pi',handoff_evidence=evidence),None)
  self.assertIsNone(policy.infer_prepared(policy.prepare_input({'state':np.zeros(14)})))
  self.assertEqual(policy.phase,'grasp')
 def test_handoff_preserves_original_pi_output(self):
  class Pi:
   def prepare_input(self,obs,rtc=None):return obs
   def infer_prepared(self,*args,**kwargs):return np.ones((45,14))
  policy=HybridPolicy(lambda obs:None,Pi())
  policy.begin_grasp()
  chunk=policy.infer_prepared(policy.prepare_input({'state':np.zeros(14)}))
  self.assertTrue(np.all(chunk==1))

if __name__=='__main__':unittest.main()
