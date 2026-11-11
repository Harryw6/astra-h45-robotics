import json
from pathlib import Path

import pytest


def case_fixture(tmp_path):
    from PIL import Image
    source=tmp_path/'case';source.mkdir()
    Image.new('RGB',(8,8),'blue').save(source/'front.png')
    (source/'input.json').write_text(json.dumps({'model':'gpt-6-astra','instruction':'Review cube grasp.',
            'content':['frame f0 at1second',{'image':'front.png','label':'f0'}]}))
    request={'schema_version':1,'case_id':'case_1','phase':'causal_review','decision_time_s':1.,
             'task_instruction':'Grasp cube inside elevated box.','input_manifest':'input.json',
             'available_frames':{'f0':{'camera':'front','time_s':1.}},'observation':{}}
    path=source/'request.json';path.write_text(json.dumps(request))
    return path,request


def test_invalid_runtime_identity_rejected_before_provider_start(tmp_path):
    from wam_astra.astra import prepare_review
    path,request=case_fixture(tmp_path)
    request['runtime_control']=dict(schema_version=1,request_id='r',session_id='s',generation=True)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match='runtime_control'):
        prepare_review(path,tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_native_worker_selects_explicit_runtime_output_contract_only_when_requested():
    from wam_astra import native_review_worker as worker
    assert hasattr(worker,'review_prompt_contract')
    instructions,schema=worker.review_prompt_contract({})
    from wam_astra.review_contract import model_schema,review_instructions
    assert instructions==review_instructions() and schema==model_schema()
    instructions,schema=worker.review_prompt_contract(dict(runtime_control=dict(
        schema_version=1,request_id='r',session_id='s',generation=1)))
    assert 'replan_policy' in instructions
    assert 'runtime_control' in schema['properties']['arguments']['required']


def test_preflight_rejects_future_images_before_any_model_process(tmp_path):
    from wam_astra.astra import prepare_review
    path,case=case_fixture(tmp_path)
    case['available_frames']['f0']['time_s']=1.01;path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='future'):
        prepare_review(path,tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_preflight_only_accepts_requested_astra_and_image_manifest(tmp_path):
    from wam_astra.astra import prepare_review
    path,_=case_fixture(tmp_path)
    manifest=path.parent/'input.json';value=json.loads(manifest.read_text());value['model']='other-model'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='gpt-6-astra'):
        prepare_review(path,tmp_path/'result')


def test_raw_source_or_existing_result_never_overwritten(tmp_path):
    from wam_astra.astra import prepare_review
    path,_=case_fixture(tmp_path)
    with pytest.raises(ValueError,match='source|outside'):
        prepare_review(path,'E:/WAM/astrarun')
    with pytest.raises(ValueError,match='source|outside'):
        prepare_review(path,path.parent/'nested')
    existing=tmp_path/'existing';existing.mkdir()
    with pytest.raises(FileExistsError):prepare_review(path,existing)


def test_preflight_records_real_images_finite_json_and_source_hashes(tmp_path):
    from wam_astra.astra import prepare_review
    path,_=case_fixture(tmp_path)
    result=prepare_review(path,tmp_path/'result')
    assert result['model']=='gpt-6-astra'
    assert len(result['images'])==1
    assert result['images'][0]['sha256']
    assert result['request_sha256']
    assert result['model_calls']==0


def test_incomplete_native_turn_is_not_reported_as_success(tmp_path):
    from wam_astra.astra import load_review_result
    (tmp_path/'native_receipt.json').write_text('{"status":"failed","model":"gpt-6-astra"}')
    (tmp_path/'reply.json').write_text('{"name":"review_action_stream","arguments":{}}')
    with pytest.raises(RuntimeError,match='completed'):
        load_review_result(tmp_path)


def test_unregistered_image_cannot_hide_outside_causal_frame_catalog(tmp_path):
    from wam_astra.astra import prepare_review
    path,_=case_fixture(tmp_path)
    manifest=path.parent/'input.json';value=json.loads(manifest.read_text())
    value['content'].append({'image':'front.png','label':'hidden_future'})
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='frame|registered'):
        prepare_review(path,tmp_path/'result')


def test_registered_frame_path_must_match_actual_input_image(tmp_path):
    from wam_astra.astra import prepare_review
    path,case=case_fixture(tmp_path)
    case['available_frames']['f0']['path']=str(path.parent/'different.png');path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='path|image'):
        prepare_review(path,tmp_path/'result')


def test_completed_archive_cannot_bypass_reply_validation(tmp_path):
    from wam_astra.astra import load_review_result
    from wam_astra.contexts import UPSTREAM_COMMIT
    (tmp_path/'native_receipt.json').write_text(json.dumps({'status':'completed','model':'gpt-6-astra','upstream_commit':UPSTREAM_COMMIT}))
    (tmp_path/'reply.json').write_text('{"name":"review_action_stream","arguments":{}}')
    (tmp_path/'validation.json').write_text('{"executable":true}')
    (tmp_path/'usage.jsonl').write_text('{"status":"completed","model":"gpt-6-astra"}\n')
    with pytest.raises((ValueError,RuntimeError),match='request|contract|artifact'):
        load_review_result(tmp_path)


def test_windows_cwd_bridge_is_preserved_for_native_recovery_factory():
    from types import SimpleNamespace
    from wam_astra.native_review_worker import configure_native_cwd
    session=SimpleNamespace(client=SimpleNamespace(project_root=Path('/wrong')),
                            _new_client=lambda:SimpleNamespace(project_root=Path('/wrong')))
    configure_native_cwd(session,'C:/Users/1/review')
    assert str(session.client.project_root).replace('\\','/')=='C:/Users/1/review'
    assert str(session._new_client().project_root).replace('\\','/')=='C:/Users/1/review'


def test_completed_usage_must_match_the_recorded_native_thread_and_turn(tmp_path):
    from wam_astra.astra import load_review_result
    from wam_astra.contexts import UPSTREAM_COMMIT
    from wam_astra.data import write_json,file_sha256
    from wam_astra.review_contract import validate_review_reply
    request={'case_id':'c','decision_time_s':1.,'phase':'causal_review',
             'available_frames':{'f0':{'camera':'front','time_s':1.}}}
    reply={'name':'review_action_stream','arguments':{'schema_version':1,'case_id':'c','decision_time_s':1.,
        'assessment':{'status':'uncertain','confidence':.2,'evidence':[],'limitations':['occluded']},
        'correction':{'strategy':'补充观察','steps':[{'kind':'observe','description':'观察','evidence_frame_ids':[]}],
                      'joint_proposal':None,'not_executable_reason':'physical limits unknown'}}}
    write_json(tmp_path/'request.json',request);write_json(tmp_path/'reply.json',reply)
    write_json(tmp_path/'validation.json',validate_review_reply(reply,case_id='c',decision_time_s=1.,available_frames=request['available_frames']))
    write_json(tmp_path/'native_receipt.json',{'status':'completed','model':'gpt-6-astra','upstream_commit':UPSTREAM_COMMIT,
        'thread_id':'thr1','turn_id':'turn1','reply_sha256':file_sha256(tmp_path/'reply.json'),
        'request_sha256':file_sha256(tmp_path/'request.json')})
    (tmp_path/'usage.jsonl').write_text('{"status":"completed","model":"gpt-6-astra","thread_id":"thr2","turn_id":"turn1"}\n')
    with pytest.raises(RuntimeError,match='usage|thread'):
        load_review_result(tmp_path)


def test_native_revision_check_accepts_crlf_checkout_but_rejects_staged_content_change(tmp_path):
    import subprocess
    from wam_astra.native_review_worker import verify_upstream
    def git(*args):
        return subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init');source=tmp_path/'client.py';source.write_bytes(b'original\n')
    git('-c','core.autocrlf=false','add','client.py');git('commit','-m','fixture')
    commit=git('rev-parse','HEAD')
    source.write_bytes(b'original\r\n')
    assert verify_upstream(tmp_path,expected_commit=commit)==commit
    source.write_bytes(b'changed\r\n');git('add','client.py')
    with pytest.raises(ValueError,match='modified'):
        verify_upstream(tmp_path,expected_commit=commit)


def test_actual_native_sampling_keeps_gripper_and_zero_telemetry_boundaries_without_editing_source():
    from copy import deepcopy
    import importlib.util
    from wam_astra.native_review_worker import compact_measured_prefix
    path=Path(__file__).resolve().parents[3]/'work/upstream/GPT-Policy/src/gpt_policy/input/action_sampling.py'
    spec=importlib.util.spec_from_file_location('_actual_review_sampling',path)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    rows=[{'video_relative_time_s':i/10,'source_row':i,'joint_measured_rad':[1.]*6 if i<5 else [0.]*6,
           'gripper_measured_m':0. if i==2 else .02} for i in range(10)]
    observation={'observation':{'measured_prefix_or_audit':rows}};before=deepcopy(observation)
    result=compact_measured_prefix(observation,compress_samples=native.compress_samples)
    assert observation==before
    assert [r['source_row'] for r in result['observation']['measured_prefix_or_audit']]==[0,1,2,3,4,5,9]
    assert result['observation']['measurement_context_sampling']['input_samples']==10
    assert result['observation']['measurement_context_sampling']['output_samples']==7


def test_native_event_journal_records_status_without_secrets_or_reasoning_text():
    from wam_astra.native_review_worker import event_summary
    assert event_summary({'id':1,'result':{'config':{'api_key':'secret'}}}) is None
    value=event_summary({'method':'item/completed','params':{'threadId':'t','turnId':'u',
        'item':{'type':'agentMessage','phase':'final_answer','text':'private content','secret':'secret'}}})
    assert value['item_type']=='agentMessage'
    assert 'private content' not in json.dumps(value) and 'secret' not in json.dumps(value)


def test_historical_reference_uses_its_own_clock_and_a_real_manifest_digest(tmp_path):
    from wam_astra.astra import prepare_review
    from wam_astra.data import file_sha256
    path,case=case_fixture(tmp_path)
    source_manifest=path.parent/'reference_manifest.json'
    source_manifest.write_text('{"schema_version":1,"episodes":[{"episode_id":"old"}]}')
    case['current_episode_id']='live'
    case['reference_library_manifest']='reference_manifest.json'
    case['available_frames']['f0'].update(time_s=147.,evidence_role='historical_reference',
        source_episode_id='old',availability_attested=True,reference_manifest_sha256=file_sha256(source_manifest))
    path.write_text(json.dumps(case))
    result=prepare_review(path,tmp_path/'result')
    assert result['reference_manifests'][0]['sha256']==file_sha256(source_manifest)
    case['available_frames']['f0']['source_episode_id']='live';path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='future'):
        prepare_review(path,tmp_path/'result')


def test_historical_attestation_digest_must_match_actual_reference_manifest(tmp_path):
    from wam_astra.astra import prepare_review
    path,case=case_fixture(tmp_path)
    (path.parent/'reference_manifest.json').write_text('{"schema_version":1,"episodes":[{"episode_id":"old"}]}')
    case.update(current_episode_id='live',reference_library_manifest='reference_manifest.json')
    case['available_frames']['f0'].update(time_s=.5,evidence_role='historical_reference',source_episode_id='old',
        availability_attested=True,reference_manifest_sha256='a'*64)
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='manifest|digest'):
        prepare_review(path,tmp_path/'result')


@pytest.mark.parametrize('field', ['current_episode_id', 'source_episode_id'])
def test_preflight_rejects_noncanonical_episode_ids_before_model_call(tmp_path, field):
    from wam_astra.astra import prepare_review
    from wam_astra.data import file_sha256
    path,case=case_fixture(tmp_path)
    source_manifest=path.parent/'reference_manifest.json'
    source_manifest.write_text('{"episodes":[{"episode_id":"old"}]}')
    case.update(current_episode_id='old',reference_library_manifest='reference_manifest.json')
    frame=case['available_frames']['f0']
    frame.update(time_s=147.,evidence_role='historical_reference',source_episode_id='old',
                 availability_attested=True,reference_manifest_sha256=file_sha256(source_manifest))
    if field=='current_episode_id':case[field]='old '
    else:frame[field]='old '
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='canonical'):
        prepare_review(path,tmp_path/'result')
    assert not (tmp_path/'result').exists()
