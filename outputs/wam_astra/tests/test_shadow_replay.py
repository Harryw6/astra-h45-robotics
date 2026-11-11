"""Offline smoke test using real compiler, native monitor and native queue."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / 'work/wam_demo_library/vendor'))


def api():
    script = PROJECT / 'examples/shadow_replay.py'
    assert script.is_file(), 'Shadow replay example must exist'
    return runpy.run_path(str(script))['run_shadow_replay']


@pytest.fixture
def library(tmp_path):
    from wam_astra.contexts import compile_demo
    from wam_astra.retrieval import build_index
    root = tmp_path / 'library'
    episode = root / 'episodes/fixture'
    episode.mkdir(parents=True)
    Image.new('RGB', (12, 8), 'red').save(episode / 'front.png')
    Image.new('RGB', (12, 8), 'blue').save(episode / 'wrist.png')
    t = np.array([1_000_000_000, 1_100_000_000], dtype=np.int64)
    q = np.array([[.1] * 6, [.11] * 6])
    np.savez_compressed(episode / 'signals.npz', state_t_ns=t, state_q=q,
                        state_g=np.array([.035, .034]), state_source_row=np.array([4, 5]))
    source = root / 'source_demos/fixture/demo.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'schema_version': 1, 'source_robot': 'Piper',
        'numeric_reference_semantics': 'measured_states', 'keyframes': [
            {'t_s': 0., 'images': {'front': str(episode / 'front.png'), 'wrist': str(episode / 'wrist.png')},
             'state': {'piper': {'joint_measured_rad': [.1] * 6, 'gripper_measured_m': .035}}},
            {'t_s': .1, 'images': {'front': str(episode / 'front.png')},
             'state': {'piper': {'joint_measured_rad': [.11] * 6, 'gripper_measured_m': .034}}},
        ]}))
    context = compile_demo(source, root / 'contexts/fixture', 'Observe fixture only')
    context = {key: str(Path(path).relative_to(root)) for key, path in context.items()}
    record = {'schema_version': 1, 'episode_id': 'fixture', 'block_id': 'block-1',
        'kind': 'fixed', 'start_ns': int(t[0]), 'end_ns': int(t[1]), 'start_s': 0., 'end_s': .1,
        'O0': {'state_monotonic_ns': int(t[0]), 'source_row': 4,
            'piper': {'joint_measured_rad': [.1] * 6, 'gripper_measured_m': .035},
            'images': {'front': {'available': True, 'path': 'episodes/fixture/front.png'}}},
        'A': {'samples': [{'piper': {'recorded_joint_target': [999.] * 6}}]},
        'receipt': {'hardware_acknowledgement': None}, 'O1': {}}
    (episode / 'block.json').write_text(json.dumps(record))
    import copy
    no_images = copy.deepcopy(record)
    no_images['block_id'] = 'block-no-images'
    no_images['O0']['images'] = {}
    (episode / 'no-images.json').write_text(json.dumps(no_images))
    block = {'block_id': 'block-1', 'episode_id': 'fixture',
        'kind': 'fixed', 'start_s': 0., 'end_s': .1, 'q_start': [.1] * 6,
        'gripper_start_m': .035, 'record_path': 'episodes/fixture/block.json'}
    invalid_block = {**block, 'block_id': 'block-no-images', 'record_path': 'episodes/fixture/no-images.json'}
    (root / 'blocks.jsonl').write_text('\n'.join(json.dumps(x) for x in (invalid_block, block)) + '\n')
    (root / 'manifest.json').write_text(json.dumps({'block_count': 2, 'episodes': [
        {'episode_id': 'fixture', 'signals_path': 'episodes/fixture/signals.npz',
         'origin_monotonic_ns': int(t[0]), 'source_demo': 'source_demos/fixture/demo.json', 'context': context}]}))
    build_index(root)
    return root


def test_actual_native_shadow_replay_reports_fixture_outcomes_and_preserves_library(library, tmp_path):
    before = {p.relative_to(library).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in library.rglob('*') if p.is_file()}
    report = api()(library, tmp_path / 'SHADOW_REPLAY.json')
    assert report['schema_version'] == 1
    assert report['synthetic_review_responses'] is True
    assert report['model_calls'] == report['actuator_calls'] == 0
    assert report['fixture_prior_gripper_normalized'] == 0
    assert report['source']['measured_gripper_m'] == .035
    assert report['source']['state_source_row'] == 4
    assert report['source']['current_rgb_available'] is True
    assert report['source']['block_id'] == 'block-1'
    assert report['source']['current_rgb_images'][0]['size'] == [12, 8]
    assert Path(report['scenarios'][0]['request']['observation']['images'][0]['path']).is_file()
    assert Path(report['source']['native_context']['input_json']).is_file()
    assert report['monitor_event']['triggered'] is True
    assert report['monitor_event']['evidence']['model_identity'] == 'RoboICL.TrackingGuard'
    assert [(row['scenario'], row['receipt']['status']) for row in report['scenarios']] == [
        ('fixture_hold', 'hold'), ('fixture_constant_edit', 'accepted'),
        ('fixture_wrong_generation', 'rejected'), ('fixture_stale_request', 'rejected')]
    assert report['scenarios'][1]['queue_size_after'] == 3
    assert report['scenarios'][1]['response']['actions'][0] == [.1] * 6 + [0.]
    assert report['source']['recorded_targets_used_as_controls'] is False
    assert json.loads((tmp_path / 'SHADOW_REPLAY.json').read_text()) == report
    assert before == {p.relative_to(library).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in library.rglob('*') if p.is_file()}


def test_replay_refuses_to_write_inside_input_library(library):
    with pytest.raises(ValueError, match='outside'):
        api()(library, library / 'bad.json')
    assert not (library / 'bad.json').exists()
