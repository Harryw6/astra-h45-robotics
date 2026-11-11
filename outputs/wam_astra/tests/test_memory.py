"""Persistent overlay tests use actual retrieval, native compiler and archives."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / 'work/wam_demo_library/vendor'))


def api():
    assert importlib.util.find_spec('wam_astra.memory') is not None, 'Task memory adapter must exist'
    from wam_astra.memory import TaskMemory
    return TaskMemory


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def library(tmp_path):
    from mcap.writer import Writer
    from wam_astra.retrieval import build_index
    raw = tmp_path / 'raw'
    raw.mkdir()
    camera = raw / 'front.mcap'
    with camera.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('heterovla.raw_image.v1', 'heterovla-layout-v1', b'{}')
        channel = writer.register_channel('/camera/front', 'heterovla.raw_image.v1', schema)
        for i, t in enumerate((2_000_000_000, 2_100_000_000)):
            metadata = {'camera': 'front', 'monotonic_ns': t, 'wall_time_ns': t + 10_000_000_000,
                        'width': 2, 'height': 1, 'stride': 6, 'pixel_format': 'bgr8', 'dtype': 'uint8', 'seq': i}
            encoded = json.dumps(metadata).encode()
            writer.add_message(channel, t + 10_000_000_000,
                struct.pack('<I', len(encoded)) + encoded + bytes([0, 0, 255] * 2), t + 10_000_000_000, i)
        writer.finish()
    root = tmp_path / 'library'
    root.mkdir()
    a = root / 'episodes/a'
    a.mkdir(parents=True)
    times = np.arange(22, dtype=np.int64) * 100_000_000 + 1_000_000_000
    np.savez_compressed(a / 'signals.npz', state_t_ns=times, state_q=np.full((22, 6), .3),
                        state_g=np.full(22, .02), state_source_row=np.arange(22))
    episodes = [
        {'episode_id': 'a', 'quality_ok': False, 'origin_monotonic_ns': 1_000_000_000,
         'signals_path': 'episodes/a/signals.npz', 'camera_shards': {'front': [
            {'path': str(camera), 'first_monotonic_ns': 2_000_000_000,
             'last_monotonic_ns': 2_100_000_000, 'first_wall_time_ns': 12_000_000_000}]}},
        {'episode_id': 'b', 'quality_ok': True}, {'episode_id': 'auxiliary', 'state': {'present': False}},
        {'episode_id': 'unrelated', 'quality_ok': True},
    ]
    blocks = []
    for bid, eid, start, q in [('a-old', 'a', 2_000_000_000, .3),
                               ('a-future', 'a', 3_000_000_000, 0.),
                               ('b-near', 'b', 2_000_000_000, .1),
                               ('unrelated-nearest', 'unrelated', 2_000_000_000, 0.)]:
        relative = f'episodes/{eid}/{bid}.json'
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {'schema_version': 1, 'episode_id': eid, 'block_id': bid, 'kind': 'dense',
            'start_ns': start, 'end_ns': start + 100_000_000, 'start_s': start / 1e9 - 1., 'end_s': start / 1e9 - .9,
            'O0': {'piper': {'joint_measured_rad': [q] * 6, 'gripper_measured_m': .02}},
            'A': {'samples': [{'piper': {'recorded_joint_target': [999.] * 6}}]},
            'receipt': {'hardware_acknowledged': None}, 'O1': {'piper': None}}
        path.write_text(json.dumps(record))
        blocks.append({**{key: record[key] for key in ('episode_id', 'block_id', 'kind', 'start_ns', 'end_ns', 'start_s', 'end_s')},
                       'record_path': relative, 'q_start': [q] * 6, 'gripper_start_m': .02})
    (root / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'source_root': str(raw),
                                                  'episodes': episodes, 'block_count': len(blocks)}))
    (root / 'blocks.jsonl').write_text('\n'.join(json.dumps(row) for row in blocks) + '\n')
    build_index(root)
    return root, raw


def test_explicit_task_mapping_persists_all_qualities_and_auxiliary_catalog(library, tmp_path):
    root, raw = library
    before, source_before = hashes(root), hashes(raw)
    memory = api()(root, tmp_path / 'memory')
    mapping = memory.associate_task('cube', 'Grasp cube from box.', ['a', 'b', 'auxiliary'])
    assert mapping['episode_ids'] == ['a', 'auxiliary', 'b']
    reopened = api()(root, tmp_path / 'memory')
    catalog = reopened.catalog('cube')
    assert {e['episode_id'] for e in catalog['episodes']} == {'a', 'b', 'auxiliary'}
    assert next(e for e in catalog['episodes'] if e['episode_id'] == 'a')['quality_ok'] is False
    assert reopened.task('cube')['source'] == 'user'
    assert len(reopened.catalog()['episodes']) == 4
    assert hashes(root) == before
    assert hashes(raw) == source_before


def test_task_subset_topk_uses_real_per_episode_native_distances(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a', 'b', 'auxiliary'])
    found = memory.retrieve('cube', joint_positions=[0.] * 6, k=3)
    assert [row['block_id'] for row in found] == ['a-future', 'b-near', 'a-old']
    assert [row['distance'] for row in found] == sorted(row['distance'] for row in found)
    assert all(row['distance_metric'] == 'euclidean_unscaled_joint_radians' for row in found)
    assert all(row['task_outcome'] == 'unknown' for row in found)
    assert memory.retrieve('cube', episode_id='auxiliary') == []
    with pytest.raises(ValueError, match='associated'):
        memory.retrieve('cube', episode_id='unrelated')


def test_native_causal_filter_precedes_neighbor_ranking(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a', 'b'])
    found = memory.retrieve('cube', joint_positions=[0.] * 6, episode_id='a',
                            before_monotonic_ns=2_100_000_000, k=1)
    assert [row['block_id'] for row in found] == ['a-old']
    with pytest.raises(ValueError, match='episode'):
        memory.retrieve('cube', before_monotonic_ns=2_100_000_000)


def test_materialize_delegates_actual_rgb_selection_and_native_compiler(library, tmp_path):
    root, raw = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a'])
    records = memory.retrieve('cube', episode_id='a', before_monotonic_ns=2_100_000_000)
    before, raw_before = hashes(root), hashes(raw)
    exported = memory.materialize('cube', records, tmp_path / 'selection')
    assert exported['compiled_context_count'] == 1
    assert exported['image_count'] == 2
    entry = exported['episodes'][0]
    assert Path(entry['context']['input_json']).is_file()
    assert Path(entry['context']['app_server_items_json']).is_file()
    native_demo = json.loads(Path(entry['context']['demo_json']).read_text())
    assert native_demo['keyframes'][0]['state']['left']['joint_measured_rad'] == [.3] * 6
    assert 'recorded_joint_target' not in json.dumps(native_demo)
    assert hashes(root) == before
    assert hashes(raw) == raw_before


@pytest.mark.parametrize('place', ['library', 'raw', 'wam'])
def test_overlay_cannot_be_written_into_sources(library, tmp_path, place):
    root, raw = library
    location = root / 'memory' if place == 'library' else raw / 'memory' if place == 'raw' else 'E:/WAM/generated-memory'
    with pytest.raises(ValueError, match='outside'):
        api()(root, location)


def test_unknown_episode_and_instruction_relabel_are_rejected(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    with pytest.raises(ValueError, match='episode'):
        memory.associate_task('cube', 'Grasp cube.', ['unknown'])
    memory.associate_task('cube', 'Grasp cube.', ['a'])
    with pytest.raises(ValueError, match='instruction'):
        memory.associate_task('cube', 'Different task.', ['b'])


@pytest.fixture
def real_review_copy(tmp_path):
    source = PROJECT / 'astra_runs/candidate_2_causal_retry1'
    if not (source / 'native_receipt.json').is_file():
        pytest.skip('Saved real native Astra run is unavailable; no model call is made by this test')
    target = tmp_path / 'review_copy'
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def test_append_saved_real_review_is_provisional_bound_and_idempotent(library, tmp_path, real_review_copy):
    root, _ = library
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', request['task_instruction'], ['a', 'b'])
    note = memory.append_review(real_review_copy)
    assert note['task_id'] == 'cube'
    assert note['case_id'] == request['case_id']
    assert note['provisional'] is True
    assert note['task_outcome'] == 'unknown'
    assert note['cube_grasped'] is None
    assert note['stable_lift'] is None
    assert note['assessment_is_ground_truth'] is False
    assert note['source_artifacts']['reply.json']['sha256']
    assert note['source_artifacts']['native_input.json']['sha256']
    assert note['source_artifacts']['app_server_items.json']['sha256']
    assert note['model'] == 'gpt-6-astra'
    assert memory.append_review(real_review_copy)['note_id'] == note['note_id']
    assert len(memory.reviews('cube')) == 1
    assert api()(root, tmp_path / 'memory').reviews('cube') == [note]


def test_corrupted_or_incomplete_review_never_enters_memory(library, tmp_path, real_review_copy):
    root, _ = library
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', request['task_instruction'], ['a'])
    (real_review_copy / 'reply.json').write_text('{}')
    with pytest.raises((ValueError, RuntimeError)):
        memory.append_review(real_review_copy)
    assert memory.reviews('cube') == []


def test_review_is_not_semantically_guessed_into_a_different_task(library, tmp_path, real_review_copy):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('other', 'A superficially similar cube task.', ['a'])
    with pytest.raises(ValueError, match='instruction|task'):
        memory.append_review(real_review_copy)
    assert memory.reviews('other') == []


def test_incomplete_archive_rejected_without_a_memory_note(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a'])
    pending = tmp_path / 'pending'
    pending.mkdir()
    (pending / 'native_receipt.json').write_text('{"status":"running"}')
    with pytest.raises(RuntimeError, match='incomplete|completed'):
        memory.append_review(pending)
    assert memory.reviews('cube') == []


def test_different_association_source_cannot_be_labeled_as_original_user(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a'], source='user')
    with pytest.raises(ValueError, match='source'):
        memory.associate_task('cube', 'Grasp cube.', ['b'], source='unverified_inference')
    assert memory.task('cube')['episode_ids'] == ['a']


def test_overlay_database_cannot_alias_the_library_index_via_hardlink(library, tmp_path):
    root, _ = library
    before = hashes(root)
    memory_root = tmp_path / 'aliased-memory'
    memory_root.mkdir()
    os.link(root / 'retrieval.sqlite3', memory_root / 'task_memory.sqlite3')
    with pytest.raises(ValueError, match='alias|outside'):
        api()(root, memory_root)
    assert hashes(root) == before


@pytest.mark.parametrize('method', ['task', 'retrieve', 'catalog', 'catalog_all', 'reviews'])
def test_library_manifest_drift_rejects_reads_without_rebinding(library, tmp_path, method):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a'])
    path = root / 'manifest.json'
    value = json.loads(path.read_text())
    value['episodes'][0]['quality_ok'] = True
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='snapshot|manifest|drift'):
        if method == 'catalog_all':
            memory.catalog()
        else:
            getattr(memory, method)('cube')
    assert path.read_bytes() == before


def test_reopening_overlay_does_not_approve_a_changed_manifest(library, tmp_path):
    root, _ = library
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', 'Grasp cube.', ['a'])
    path = root / 'manifest.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='snapshot|manifest|drift'):
        api()(root, tmp_path / 'memory').retrieve('cube')


def test_review_provenance_and_context_exclusion_bind_to_actual_archived_request(library, tmp_path, real_review_copy):
    root, _ = library
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', request['task_instruction'], ['a'])
    note = memory.append_review(real_review_copy)
    assert note['source_episode_id'] == request['observation']['source_episode']
    assert note['clock_domain'] == request.get('clock_domain', 'unspecified')
    assert note['latest_input_frame_time_s'] == max(frame['time_s'] for frame in request['available_frames'].values())
    cited = {e['frame_id'] for e in note['validated_arguments']['assessment']['evidence']}
    cited.update(frame for step in note['validated_arguments']['correction']['steps'] for frame in step['evidence_frame_ids'])
    assert note['latest_cited_evidence_time_s'] == max(request['available_frames'][frame]['time_s'] for frame in cited)
    assert note['evidence_mode'] == 'causal'
    assert memory.reviews_for_context('cube', current_case_id=note['case_id'], current_episode_id='other') == []
    assert memory.reviews_for_context('cube', current_case_id='earlier-case', current_episode_id=note['source_episode_id']) == []
    eligible = memory.reviews_for_context('cube', current_case_id='new-case', current_episode_id='other')
    assert len(eligible) == 1
    assert eligible[0]['source_artifact_hashes_revalidated'] is True
    assert eligible[0]['provisional'] is True
    assert eligible[0]['task_outcome'] == 'unknown'


def test_legacy_unknown_episode_note_stays_readable_but_is_excluded_from_context(library, tmp_path, real_review_copy):
    root, _ = library
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', request['task_instruction'], ['a'])
    note = memory.append_review(real_review_copy)
    for key in ('source_episode_id', 'clock_domain', 'latest_input_frame_time_s',
                'latest_cited_evidence_time_s', 'evidence_mode'):
        note.pop(key, None)
    with sqlite3.connect(memory.database_path) as db:
        db.execute('UPDATE review_notes SET note_json=? WHERE note_id=?', (json.dumps(note), note['note_id']))
    assert memory.reviews('cube') == [note]
    assert memory.reviews_for_context('cube', current_case_id='new', current_episode_id='different') == []
    assert memory.reviews('cube') == [note]


def test_context_read_rejects_changed_archived_payload(library, tmp_path, real_review_copy):
    root, _ = library
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    memory = api()(root, tmp_path / 'memory')
    memory.associate_task('cube', request['task_instruction'], ['a'])
    memory.append_review(real_review_copy)
    with (real_review_copy / 'request.json').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    with pytest.raises(RuntimeError, match='hash|artifact'):
        memory.reviews_for_context('cube', current_case_id='new', current_episode_id='different')


@pytest.mark.parametrize('reference_domain', [None, 'recorded_episode_elapsed_s'])
def test_mixed_clock_provenance_keeps_current_max_and_each_reference_clock(reference_domain):
    from wam_astra.memory import _review_provenance
    from wam_astra.review_contract import validate_review_reply
    frames = {
        'live': {'camera': 'front', 'time_s': 4.8},
        'old': {'camera': 'front', 'time_s': 147., 'evidence_role': 'historical_reference',
                'source_episode_id': 'old-episode', 'availability_attested': True,
                'reference_manifest_sha256': 'a' * 64},
    }
    if reference_domain is not None:
        frames['old']['clock_domain'] = reference_domain
    request = {'case_id': 'mixed', 'decision_time_s': 5., 'current_episode_id': 'live-episode',
        'clock_domain': 'current_boot_monotonic_s', 'observation': {'source_episode': 'live-episode'},
        'available_frames': frames}
    reply = {'name': 'review_action_stream', 'arguments': {'schema_version': 1,
        'case_id': 'mixed', 'decision_time_s': 5., 'assessment': {'status': 'misaligned', 'confidence': .6,
            'evidence': [{'frame_id': 'live', 'camera': 'front', 'time_s': 4.8, 'observation': '当前可见偏差。'},
                         {'frame_id': 'old', 'camera': 'front', 'time_s': 147., 'observation': '旧示范接近方向。'}],
            'limitations': ['没有接触力证据。']},
        'correction': {'strategy': '补充观察并重新对齐。',
            'steps': [{'kind': 'align', 'description': '根据旧示范关系检查当前对齐。', 'evidence_frame_ids': ['old']}],
            'joint_proposal': None, 'not_executable_reason': '未验证机械约束。'}}}
    validation = validate_review_reply(reply, case_id='mixed', decision_time_s=5.,
                                      current_episode_id='live-episode', available_frames=frames)
    provenance = _review_provenance(request, validation)
    assert provenance['latest_input_frame_time_s'] == 4.8
    assert provenance['latest_cited_evidence_time_s'] == 4.8
    assert provenance['clock_domain'] == 'current_boot_monotonic_s'
    assert provenance['historical_reference_times'] == [
        {'id': 'old', 'source_episode_id': 'old-episode',
         'clock_domain': reference_domain or 'unspecified', 'time_s': 147.}]


def test_legacy_note_time_provenance_keeps_its_exact_existing_shape(real_review_copy):
    from wam_astra.memory import _review_provenance
    request = json.loads((real_review_copy / 'request.json').read_text(encoding='utf-8'))
    validation = json.loads((real_review_copy / 'validation.json').read_text(encoding='utf-8'))
    provenance = _review_provenance(request, validation)
    assert 'historical_reference_times' not in provenance
    assert provenance['latest_input_frame_time_s'] == max(f['time_s'] for f in request['available_frames'].values())
