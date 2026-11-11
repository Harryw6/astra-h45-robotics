import hashlib
from pathlib import Path

import numpy as np
import pytest


def api():
    try:
        from wam_astra import data
    except ImportError:
        pytest.fail('WAM source adapter has not been implemented')
    return data


def test_output_inside_source_is_rejected(tmp_path):
    root=tmp_path/'raw';root.mkdir()
    with pytest.raises(ValueError):
        api().ensure_separate_output(root,root/'generated')


def test_csv_keeps_original_row_numbers_and_records_invalid_tail(tmp_path):
    f=tmp_path/'piper_state.csv'
    f.write_bytes(b'monotonic_ns,wall_time_ns,joint_1.pos,joint_2.pos,joint_3.pos,joint_4.pos,joint_5.pos,joint_6.pos,gripper.pos\n'
                 b'1000000000,1790000000000000001,1,2,3,4,5,6,.02\n'
                 b'1100000000,1790000000100000001,2,3,4,5,6,7,.03\n'+b'\0'*32)
    before=hashlib.sha256(f.read_bytes()).hexdigest()
    s=api().read_piper_csv(f)
    assert s['invalid_rows']==[2]
    assert s['source_row'].tolist()==[0,1]
    assert s['wall_ns'].tolist()==[1790000000000000001,1790000000100000001]
    assert s['q'].shape==(2,6)
    assert hashlib.sha256(f.read_bytes()).hexdigest()==before


def test_causal_sampling_never_uses_future_state_or_fills_stale_values():
    d=api()
    times=np.array([100,200],dtype=np.int64)
    values=np.array([[1.],[2.]])
    assert d.causal_at(times,values,99,max_age_ns=100) is None
    assert d.causal_at(times,values,150,max_age_ns=100)['value']==[1.]
    assert d.causal_at(times,values,301,max_age_ns=100) is None


def test_bad_timestamp_is_classified_without_losing_valid_integer_precision(tmp_path):
    f=tmp_path/'bad.csv'
    f.write_text('monotonic_ns,wall_time_ns,joint_1.pos,joint_2.pos,joint_3.pos,joint_4.pos,joint_5.pos,joint_6.pos\n'
                 'bad,1790000000000000000,0,0,0,0,0,0\n'
                 '1000000001,1790000000000000001,0,0,0,0,0,0\n')
    result=api().read_piper_csv(f)
    assert result['invalid_rows']==[0]
    assert result['wall_ns'].tolist()==[1790000000000000001]


def test_no_sent_command_means_unknown_dispatch_not_success():
    r=api().receipt_record(command_rows=[],measured_rows=[2,3],tracking=None)
    assert r['hardware_acknowledged'] is None
    assert r['task_success'] is None
    assert r['command_sent_rows']==[]
    assert r['measured_state_rows']==[2,3]


def test_measured_gripper_change_anchors_preserve_close_and_reopen_without_normalizing():
    signal={'t_ns':np.arange(5,dtype=np.int64)*100_000_000,
            'q':np.zeros((5,6)),'g':np.array([.02,.02,0.,.02,.02]),'source_row':np.arange(5)}
    action=api().measured_action(signal,0,500_000_000,0)
    assert action['preserve_sample_indices']==[1,2,3]
    assert [s['piper']['gripper_measured_m'] for s in action['samples']]==[.02,.02,0.,.02,.02]
    assert 'task_success' not in action


def test_reference_windows_are_deterministic_and_disjoint():
    d=api()
    a=d.fixed_windows(0,20_000_000_000,count=12,window_ns=1_000_000_000)
    b=d.fixed_windows(0,20_000_000_000,count=12,window_ns=1_000_000_000)
    assert a==b and len(a)==12
    assert a[0][0]==0 and a[-1][1]==20_000_000_000
    assert all(x[1]<=y[0] for x,y in zip(a,a[1:]))


def test_short_recording_is_retained_with_fewer_windows():
    a=api().fixed_windows(0,2_500_000_000,count=12,window_ns=1_000_000_000)
    assert len(a)==2 and a[0][0]==0 and a[-1][1]<=2_500_000_000


def test_inventory_discovers_auxiliary_only_episode(tmp_path):
    source=tmp_path/'raw'
    a=source/'2026-10-21/完整采集数据/原始数据/无人机/joint_20261021_120000_UTCp0800_abcd'
    a.mkdir(parents=True);(a/'metadata.yaml').write_text('test')
    episodes=api().discover_episodes(source)
    assert len(episodes)==1
    assert episodes[0]['robot_dir'] is None
    assert episodes[0]['drone_dir']==str(a)
    assert episodes[0]['included'] is True


def test_source_inventory_detects_changed_inputs(tmp_path):
    root=tmp_path/'raw';root.mkdir();f=root/'a.bin';f.write_bytes(b'one')
    old=api().inventory(root)
    f.write_bytes(b'longer')
    result=api().compare_inventory(old,api().inventory(root))
    assert result['unchanged'] is False
    assert result['changed']==['a.bin']
