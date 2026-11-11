import json
from pathlib import Path

import pytest


def raw_fixture(root):
    base=root/'2026-10-12'/'完整采集数据'
    episode='joint_20261012_155725_UTCp0800_faf6'
    robot=base/'原始数据'/'机器狗'/episode/episode
    (robot/'raw').mkdir(parents=True)
    header='monotonic_ns,wall_time_ns,joint_1.pos,joint_2.pos,joint_3.pos,joint_4.pos,joint_5.pos,joint_6.pos,gripper.pos'
    rows=[f'{1_000_000_000+i*100_000_000},{11_000_000_000+i*100_000_000},0,0,0,0,0,0,0.02' for i in range(41)]
    (robot/'raw/piper_state.csv').write_text(header+'\n'+'\n'.join(rows)+'\n')
    (robot/'summary.json').write_text('{"quality_ok":false}')
    auxiliary='joint_20261012_155726_UTCp0800_abcd'
    (base/'原始数据'/'无人机'/auxiliary).mkdir(parents=True)
    (base/'原始数据'/'无人机'/auxiliary/'meta.json').write_text('{}')
    return episode,auxiliary


def test_all_episodes_retained_without_quality_gate_no_source_writes(tmp_path):
    from wam_astra.builder import build_library
    from wam_astra.data import inventory
    raw=tmp_path/'raw';episode,auxiliary=raw_fixture(raw)
    before=inventory(raw)
    result=build_library(raw,tmp_path/'library',extract_images=False,compile_contexts=False)
    manifest=json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    assert {r['episode_id'] for r in manifest['episodes']}=={episode,auxiliary}
    assert next(r for r in manifest['episodes'] if r['episode_id']==episode)['quality_ok'] is False
    assert manifest['source_verification']['unchanged'] is True
    assert inventory(raw)==before
    blocks=[json.loads(s) for s in (tmp_path/'library/blocks.jsonl').read_text().splitlines()]
    assert blocks and all(r['episode_id']==episode for r in blocks)
    first=json.loads((tmp_path/'library'/blocks[0]['record_path']).read_text())
    assert first['O0']['piper']['joint_measured_rad']==[0]*6
    assert first['A']['samples']==[]
    assert first['receipt']['hardware_acknowledged'] is None
    assert first['receipt']['task_success'] is None


def test_builder_refuses_existing_output_and_raw_descendant(tmp_path):
    from wam_astra.builder import build_library
    raw=tmp_path/'raw';raw_fixture(raw)
    with pytest.raises(ValueError):
        build_library(raw,raw/'derived',extract_images=False,compile_contexts=False)
    existing=tmp_path/'existing';existing.mkdir()
    with pytest.raises(FileExistsError):
        build_library(raw,existing,extract_images=False,compile_contexts=False)


def test_receipt_state_evidence_never_includes_a_future_row(tmp_path):
    from wam_astra.builder import build_library
    import numpy as np
    raw=tmp_path/'raw';eid,_=raw_fixture(raw)
    state=next(raw.rglob('piper_state.csv'))
    with state.open('a') as stream:stream.write('5050000000,15050000000,0,0,0,0,0,0,0.02\n')
    root=tmp_path/'library';build_library(raw,root,extract_images=False,compile_contexts=False)
    with np.load(root/'episodes'/eid/'signals.npz',allow_pickle=False) as signals:
        times=signals['state_t_ns']
    for path in (root/'episodes'/eid/'blocks').glob('*.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        assert all(int(times[row])<=record['end_ns'] for row in record['receipt']['measured_state_rows'])


def test_unreadable_signal_layout_does_not_drop_episode_or_abort_catalog(tmp_path):
    from wam_astra.builder import build_library
    raw=tmp_path/'raw';eid,aux=raw_fixture(raw)
    next(raw.rglob('piper_state.csv')).write_text('malformed,columns\n1,2\n')
    root=tmp_path/'library';build_library(raw,root,extract_images=False,compile_contexts=False)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert {e['episode_id'] for e in manifest['episodes']}=={eid,aux}
    row=next(e for e in manifest['episodes'] if e['episode_id']==eid)
    assert any('parse_error' in issue for issue in row['issues'])
    assert manifest['source_verification']['unchanged'] is True
