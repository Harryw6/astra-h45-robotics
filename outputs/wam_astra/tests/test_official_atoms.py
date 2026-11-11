import unittest
from unittest.mock import Mock
from wam_astra.official_atoms import OfficialAtoms


class OfficialAtomTests(unittest.TestCase):
    def setUp(self):
        self.dog=Mock(); self.dog.MoveToIncrementPosition.return_value=0
        self.plane=Mock()
        self.adapter=OfficialAtoms(self.dog,self.plane,last_command_id=0)

    def test_caller_must_supply_controller_command_id(self):
        with self.assertRaises(TypeError):OfficialAtoms(self.dog,self.plane)
        for invalid in [True,-1,1.5,None,2**32]:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                OfficialAtoms(self.dog,self.plane,last_command_id=invalid)

    def test_handoff_continues_previous_controller_sequence(self):
        adapter=OfficialAtoms(self.dog,self.plane,last_command_id=837)
        a=dict(decision='move',robot='uav',direction='forward',distance_m=.2)
        self.assertEqual(adapter.dispatch('first',a)['command_id'],838)
        adapter.dispatch('first',a)
        self.assertEqual(adapter.dispatch('second',a)['command_id'],839)
        self.assertEqual(self.plane.call_count,2)

    def test_uint32_sequence_is_never_wrapped(self):
        adapter=OfficialAtoms(self.dog,self.plane,last_command_id=2**32-1)
        with self.assertRaisesRegex(ValueError,'exhausted'):
            adapter.dispatch('full',dict(decision='move',robot='uav',direction='up',distance_m=.2))
        self.plane.assert_not_called()

    def test_uncertain_uav_submission_consumes_id_without_replay(self):
        self.plane.side_effect=TimeoutError('publication result unknown')
        a=dict(decision='move',robot='uav',direction='left',distance_m=.2)
        with self.assertRaises(TimeoutError):self.adapter.dispatch('lost',a)
        retry=self.adapter.dispatch('lost',a)
        self.assertEqual(retry['status'],'unknown')
        self.assertEqual(retry['command_id'],1)
        self.assertEqual(self.plane.call_count,1)

    def test_go2_increment_is_sent_once_and_not_claimed_arrived(self):
        a=dict(decision='move',robot='go2',direction='left',distance_m=.3)
        first=self.adapter.dispatch('a',a)
        self.assertEqual(first['status'],'submitted')
        self.assertFalse(first['arrival_confirmed'])
        self.assertEqual(first,self.adapter.dispatch('a',a))
        self.dog.MoveToIncrementPosition.assert_called_once_with(0.,.3,0.)
        with self.assertRaises(ValueError):
            self.adapter.dispatch('a',dict(a,distance_m=.4))

    def test_uav_uses_official_body_position_message(self):
        a=dict(decision='move',robot='uav',direction='up',distance_m=.2)
        self.adapter.dispatch('b',a)
        self.plane.assert_called_once_with(command_id=1,body_xyz=(0.,0.,.2))

    def test_uncertain_sdk_call_is_not_retried(self):
        self.dog.MoveToIncrementPosition.side_effect=TimeoutError('reply lost')
        a=dict(decision='move',robot='go2',direction='forward',distance_m=.2)
        with self.assertRaises(TimeoutError):self.adapter.dispatch('c',a)
        self.assertEqual(self.adapter.dispatch('c',a)['status'],'unknown')
        self.assertEqual(self.dog.MoveToIncrementPosition.call_count,1)

    def test_invalid_and_non_motion_decisions_never_call_sdk(self):
        for distance in [True,float('nan'),float('inf'),-.1]:
            with self.assertRaises(ValueError):
                self.adapter.dispatch(str(distance),dict(decision='move',robot='go2',direction='forward',distance_m=distance))
        self.adapter.dispatch('wait',dict(decision='observe'))
        self.dog.MoveToIncrementPosition.assert_not_called()
        self.plane.assert_not_called()

if __name__=='__main__':unittest.main()
