"""Native memory preparation with synthetic live transport; no real model calls."""
import base64
from io import BytesIO
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

from test_memory_request import memory_case


def api():
    assert importlib.util.find_spec('wam_astra.live_review_job') is not None,'Live memory review job is missing'
    from wam_astra import live_review_job
    return live_review_job


def packet():
    raw=BytesIO();Image.new('RGB',(640,480),'blue').save(raw,format='PNG')
    encoded=base64.b64encode(raw.getvalue()).decode()
    state=[0.]*17;state[6]=1.;state[10:]=[.25]*6+[.035]
    return dict(observation_id='obs-live',session='s',stamp=99.9,state=state,
        images={'front':encoded,'wrist':encoded},image_stamps={'front':99.8,'wrist':99.81},
        p450_packet=dict(stamp=99.83,state4=[0.]*4,images={'uav':encoded},image_stamps={'uav':99.82}))


CONTROL=dict(schema_version=1,request_id='r-live',session_id='s',generation=1)
MAPPING={k:dict(offset_s=0.,uncertainty_s=.001,checked_host_s=99.5) for k in ('unitree','p450')}


def job(memory,tmp_path):
    return api().LiveMemoryReviewJob(memory,'cube',tmp_path/'live-jobs',
        clock_snapshot=lambda:MAPPING,clock=lambda:100.,max_capture_age_s=.5,max_clock_age_s=5.,
        model_timeout_s=60.,effort='low',episode_id='synthetic-fresh-session')


def test_live_job_builds_native_reference_request_with_three_real_payload_images(memory_case,tmp_path):
    memory,_,_,_,_=memory_case
    prepared=job(memory,tmp_path).prepare(packet=packet(),control=CONTROL,event={'reason':'test'},
        deadline_s=170.,history=[],candidate=None)
    request=json.loads(Path(prepared['request_json']).read_text(encoding='utf-8'))
    assert request['runtime_control']==CONTROL and request['task_id']=='cube'
    current={k:v for k,v in request['available_frames'].items() if v.get('evidence_role')!='historical_reference'}
    assert {v['camera'] for v in current.values()}=={'front','wrist','uav'}
    assert all(v['time_s']<=100. for v in current.values())
    assert request['observation']['state']==[.25]*6+[.035]
    assert request['memory_context']['reference_block_ids']


def test_stale_or_future_capture_never_starts_native_model_job(memory_case,tmp_path,monkeypatch):
    memory,_,_,_,_=memory_case
    module=api();called=[];monkeypatch.setattr(module,'run_review',lambda *a,**kw:called.append(kw))
    value=packet();value['image_stamps']['front']=98.
    with pytest.raises(ValueError,match='capture'):
        job(memory,tmp_path)(packet=value,control=CONTROL,event={'reason':'test'},deadline_s=170.,history=[],candidate=None)
    assert not called


def test_job_uses_remaining_deadline_and_actual_native_runner(memory_case,tmp_path,monkeypatch):
    memory,_,_,_,_=memory_case
    module=api();seen=[]
    def provider(request,destination,**kw):
        seen.append(kw)
        actual=json.loads(Path(request).read_text(encoding='utf-8'))
        assert actual['runtime_control']==CONTROL
        raise RuntimeError('test stops before any real provider request')
    monkeypatch.setattr(module,'run_review',provider)
    with pytest.raises(RuntimeError,match='test stops'):
        job(memory,tmp_path)(packet=packet(),control=CONTROL,event={'reason':'test'},deadline_s=112.,history=[],candidate=None)
    assert len(seen)==1 and 0<seen[0]['timeout_s']<=12.


def test_custom_raw_source_root_is_rejected_before_writing_case(memory_case,tmp_path):
    memory,_,_,_,raw=memory_case
    with pytest.raises(ValueError,match='raw|source'):
        api().LiveMemoryReviewJob(memory,'cube',raw/'new-job',clock_snapshot=lambda:MAPPING,
            max_capture_age_s=.5,max_clock_age_s=5.,model_timeout_s=60.,effort='low')
    assert not (raw/'new-job').exists()


def test_live_checkpoint_preserves_bound_candidate_for_native_review(memory_case,tmp_path):
    from test_action_checkpoints import candidate
    memory,_,_,_,_=memory_case
    c=candidate()
    prepared=job(memory,tmp_path).prepare(packet=packet(),control=CONTROL,
        event=dict(event_id='closing',reason='close check',checkpoint_kinds=['gripper_pre_close']),
        deadline_s=170.,history=[],candidate=c)
    request=json.loads(Path(prepared['request_json']).read_text(encoding='utf-8'))
    assert request['action_checkpoint']['candidate']==c
    assert request['action_checkpoint']['checkpoint_kinds']==['gripper_pre_close']
