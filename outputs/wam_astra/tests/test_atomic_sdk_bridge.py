"""Use actual native converters with no network or hardware objects."""
import importlib
import math
from pathlib import Path
import sys
import threading
import types
import unittest
from collections import namedtuple
from unittest.mock import Mock
import numpy as np
from wam_astra.atomic_motion import atomic_chunk

ROOT=Path(__file__).resolve().parents[1]/'h45_takeover/staged'
for name,path in [('atomic_dog_native',ROOT/'dog/real_robot'),('atomic_plane_native',ROOT/'p450')]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
dog=importlib.import_module('atomic_dog_native.goal_distance_gateway')
plane=importlib.import_module('atomic_plane_native.p450_online_h45')
gateway=importlib.import_module('atomic_dog_native.gateway')


class NativeSDKTests(unittest.TestCase):
    def test_plain_pi_request_is_rejected_by_legacy_review_configuration(self):
        service=gateway.Gateway.__new__(gateway.Gateway)
        service.session='s';service.review_epoch=0;service.review_phase='open'
        service.review_requested=threading.Event();service.review_deadline=None
        service.review_config={'require_deadline_binding':True};service.resume_monitoring=False
        with self.assertRaisesRegex(gateway.ReviewRejected,'Wrong/missing review epoch'):
            service._check_review_credentials({'session':'s'})
        service.review_config=None
        service._check_review_credentials({'session':'s'})

    def test_go2_displacement_is_converted_to_velocity_not_sent_as_velocity(self):
        state=np.zeros(14);state[9]=.03
        action=dict(decision='move',robot='go2',direction='forward',distance_m=.3)
        decoded=dog.decode_goals(atomic_chunk(action,state),state[3:10])
        np.testing.assert_allclose(decoded[:,0],.1)
        np.testing.assert_allclose(decoded[:,3:9],0)
        np.testing.assert_allclose(decoded[:,9],.03)
        self.assertAlmostEqual(float(decoded[:,0].sum()/15),.3)

    def test_uav_heading_and_vertical_use_native_world_transform(self):
        Pose=namedtuple('Pose','x y z yaw');anchor=Pose(1,2,3,math.pi/2)
        for direction,expected in [('forward',(1,2.3,3)),('left',(.7,2,3)),('up',(1,2,3.3))]:
            chunk=atomic_chunk(dict(decision='move',robot='uav',direction=direction,distance_m=.3),np.zeros(14))
            target=plane.world_target(Pose,anchor,chunk[-1,10:14])
            np.testing.assert_allclose(target[:3],expected)

    def test_legacy_finish_enters_review_but_atomic_config_uses_hold(self):
        # Reproduce the deployment behavior that differs from plain wait/resume.
        service=gateway.Gateway.__new__(gateway.Gateway)
        service.review_config={'require_deadline_binding':True,'max_review_s':180}
        service.review_lock=threading.RLock();service.dispatch_lock=threading.RLock()
        service.review_requested=threading.Event();service.closed=threading.Event()
        service.fault=None;service.worker=None;service.session='s';service.review_epoch=0
        service.pause_review=Mock();service.hardware=Mock()
        service._finish_hold()
        service.pause_review.assert_called_once()
        service.hardware.hold.assert_not_called()
        service.review_config=None
        service._finish_hold()
        service.hardware.hold.assert_called_once()

if __name__=='__main__':unittest.main()
