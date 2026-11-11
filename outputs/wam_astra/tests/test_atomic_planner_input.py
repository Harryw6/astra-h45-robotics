"""Astra must receive the checkpoint's measured-state field names and units."""
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from wam_astra.atomic_planner import AtomicPlannerClient


def test_live_request_labels_measured_state_without_changing_values(tmp_path):
    client=AtomicPlannerClient.__new__(AtomicPlannerClient)
    client.output=tmp_path;client.count=0;client.packet={}
    client.feedback=SimpleNamespace(latest=None)
    client.last_handoff_assessment=None;client.reference_ids={'demo'}
    client.process=SimpleNamespace(stdin=io.StringIO())
    client._receive=Mock(return_value={'id':1,'action':{'decision':'observe'}})
    state=np.arange(14,dtype=np.float32)/10
    client.decide({'state':state,'prompt':'cube','image':{
        name:np.zeros((2,2,3),dtype=np.uint8)
        for name in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}})
    observation=json.loads(client.process.stdin.getvalue())['observation']
    assert observation['source']=='live'
    assert observation['state14']==state.tolist()
    fields=list(zip(observation['state14_names'],observation['state14_units']))
    assert fields[:3]==[('go2.vx_heading','m/s'),('go2.vy_heading','m/s'),('go2.yaw_rate','rad/s')]
    assert fields[3:9]==[('piper.q'+str(i),'rad') for i in range(1,7)]
    assert fields[9]==('piper.gripper','m')
    assert fields[10:]==[('uav.vx_heading','m/s'),('uav.vy_heading','m/s'),('uav.vz','m/s'),('uav.yaw_rate','rad/s')]
    assert 'measured' in observation['state14_semantics']
