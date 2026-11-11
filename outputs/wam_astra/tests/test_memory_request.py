"""Real MCAP/native context preparation; no model or robot calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import csv
import subprocess

import numpy as np
from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parents[1] / 'work/wam_demo_library/vendor'))
TASK = '目标物体在桌上盒子里，是里面的方块，起点在地面上，桌子有一定高度'


def api():
    assert importlib.util.find_spec('wam_astra.memory_request') is not None, 'Memory request adapter must exist'
    from wam_astra.memory_request import prepare_memory_review
    return prepare_memory_review


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def memory_case(tmp_path, request):
    from mcap.writer import Writer
    from wam_astra.memory import TaskMemory
    from wam_astra.retrieval import build_index
    single = getattr(request, 'param', {}).get('single_capture', False)
    raw = tmp_path / 'raw'
    raw.mkdir()
    camera = raw / 'front.mcap'
    captures = [147_920_000_000, 148_020_000_000] + ([] if single else [148_120_000_000]) + [148_220_000_000]
    with camera.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('heterovla.raw_image.v1', 'heterovla-layout-v1', b'{}')
        channel = writer.register_channel('/camera/front', 'heterovla.raw_image.v1', schema)
        for i, timestamp in enumerate(captures):
            meta = {'camera': 'front', 'monotonic_ns': timestamp, 'wall_time_ns': timestamp + 10_000_000_000,
                    'width': 2, 'height': 1, 'stride': 6, 'pixel_format': 'bgr8', 'dtype': 'uint8', 'seq': i}
            encoded = json.dumps(meta).encode()
            writer.add_message(channel, timestamp + 10_000_000_000,
                struct.pack('<I', len(encoded)) + encoded + bytes([0, 0, 255] * 2), timestamp + 10_000_000_000, i)
        writer.finish()
    library = tmp_path / 'library'
    episode = library / 'episodes/old-demo'
    episode.mkdir(parents=True)
    times = np.array([147_900_000_000, 148_000_000_000, 148_100_000_000], dtype=np.int64)
    np.savez_compressed(episode / 'signals.npz', state_t_ns=times, state_q=np.full((3, 6), .25),
                        state_g=np.full(3, .035), state_source_row=np.arange(3))
    end = 148_050_000_000 if single else 148_140_000_000
    record = {'schema_version': 1, 'episode_id': 'old-demo', 'block_id': 'old-block', 'kind': 'dense',
        'start_ns': 148_040_000_000, 'end_ns': end, 'start_s': 147.04, 'end_s': (end - 1_000_000_000) / 1e9,
        'O0': {'piper': {'joint_measured_rad': [.25] * 6, 'gripper_measured_m': .035}},
        'A': {'samples': []}, 'receipt': {'hardware_acknowledged': None}, 'O1': {'piper': None}}
    (episode / 'block.json').write_text(json.dumps(record))
    indexed = {key: record[key] for key in ('episode_id', 'block_id', 'kind', 'start_ns', 'end_ns', 'start_s', 'end_s')}
    indexed.update(record_path='episodes/old-demo/block.json', q_start=[.25] * 6, gripper_start_m=.035)
    (library / 'blocks.jsonl').write_text(json.dumps(indexed) + '\n')
    (library / 'manifest.json').write_text(json.dumps({'source_root': str(raw), 'block_count': 1, 'episodes': [
        {'episode_id': 'old-demo', 'quality_ok': False, 'origin_monotonic_ns': 1_000_000_000,
         'signals_path': 'episodes/old-demo/signals.npz', 'camera_shards': {'front': [
             {'path': str(camera), 'first_monotonic_ns': captures[0], 'last_monotonic_ns': captures[-1],
              'first_wall_time_ns': captures[0] + 10_000_000_000}]}}]}))
    build_index(library)
    memory = TaskMemory(library, tmp_path / 'memory')
    memory.associate_task('cube', TASK, ['old-demo'])
    case = tmp_path / 'current'
    case.mkdir()
    Image.new('RGB', (4, 3), 'blue').save(case / 'live.png')
    (case / 'input.json').write_text(json.dumps({'instruction': TASK, 'model': 'gpt-6-astra',
        'content': ['Current observation text.', {'image': 'live.png', 'label': 'live_front current RGB'}]}))
    current = {'schema_version': 1, 'case_id': 'live-case', 'current_episode_id': 'new-live-boot',
        'phase': 'causal_review', 'decision_time_s': 5., 'task_instruction': TASK,
        'input_manifest': 'input.json', 'available_frames': {'live_front': {
            'camera': 'front', 'time_s': 4.8, 'path': str(case / 'live.png')}},
        'observation': {'source_episode': 'new-live-boot', 'state': [.25] * 6 + [.035],
                        'state_joint_units': 'rad', 'state_gripper_units': 'm'}}
    request_path = case / 'request.json'
    request_path.write_text(json.dumps(current))
    return memory, request_path, current, library, raw


def test_actual_native_references_preserve_147_second_clock_for_five_second_live_case(memory_case, tmp_path):
    from wam_astra.astra import prepare_review
    memory, request, original, library, raw = memory_case
    before = [hashes(root) for root in (request.parent, library, raw)]
    result = api()(memory, 'cube', request, tmp_path / 'prepared')
    prepared = json.loads(Path(result['request_json']).read_text(encoding='utf-8'))
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    assert prepared['available_frames']['live_front'] == original['available_frames']['live_front']
    assert manifest['content'][0] == 'Current observation text.'
    assert prepared['decision_time_s'] == 5.
    references = {key: frame for key, frame in prepared['available_frames'].items() if key != 'live_front'}
    assert len(references) == 2
    assert sorted(frame['time_s'] for frame in references.values()) == [147.02, 147.12]
    for frame in references.values():
        assert frame['source_episode_id'] == 'old-demo'
        assert frame['evidence_role'] == 'historical_reference'
        assert frame['availability_attested'] is True
        assert frame['reference_manifest_sha256'] == hashlib.sha256((library / 'manifest.json').read_bytes()).hexdigest()
        assert Path(frame['path']).is_file()
    preflight = prepare_review(result['request_json'], tmp_path / 'not_called_model_output')
    assert preflight['model_calls'] == 0
    assert len(preflight['images']) == 3
    preparation = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert preparation['retrieval_mode'] == 'joint_q6_radians'
    assert preparation['native_text_budget']['text_chars'] > 0
    assert preparation['model_calls'] == preparation['actuator_calls'] == 0
    assert [hashes(root) for root in (request.parent, library, raw)] == before


@pytest.mark.parametrize('recorded_id', ['old-demo', 'old-demo '])
def test_recorded_episode_replay_requires_explicit_custom_clock_mapping_else_rejected(memory_case, tmp_path, recorded_id):
    memory, request, current, _, _ = memory_case
    current['current_episode_id'] = current['observation']['source_episode'] = recorded_id
    request.write_text(json.dumps(current))
    with pytest.raises(ValueError, match='episode|replay'):
        api()(memory, 'cube', request, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('k', [0, 4, True])
def test_reference_count_is_explicitly_bounded(memory_case, tmp_path, k):
    memory, request, _, _, _ = memory_case
    with pytest.raises(ValueError, match='k'):
        api()(memory, 'cube', request, tmp_path / 'invalid', k=k)


def test_native_text_budget_rejects_oversized_current_input_without_truncation(memory_case, tmp_path):
    memory, request, _, _, _ = memory_case
    path = request.parent / 'input.json'
    manifest = json.loads(path.read_text())
    manifest['content'][0] = 'x' * 1_048_577
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='large|limit|budget'):
        api()(memory, 'cube', request, tmp_path / 'oversized')
    assert not (tmp_path / 'oversized').exists()
    assert len(json.loads(path.read_text())['content'][0]) == 1_048_577


@pytest.mark.parametrize('memory_case', [{'single_capture': True}], indirect=True)
def test_duplicate_historical_capture_has_one_stable_id_and_one_image_item(memory_case, tmp_path):
    memory, request, _, _, _ = memory_case
    result = api()(memory, 'cube', request, tmp_path / 'deduplicated')
    prepared = json.loads(Path(result['request_json']).read_text(encoding='utf-8'))
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    assert len(prepared['available_frames']) == 2
    assert len([part for part in manifest['content'] if isinstance(part, dict)]) == 2
    assert json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))['deduplicated_historical_images'] == 1


def test_missing_measured_state_uses_labeled_metadata_retrieval(memory_case, tmp_path):
    memory, request, current, _, _ = memory_case
    current['observation'].pop('state')
    request.write_text(json.dumps(current))
    result = api()(memory, 'cube', request, tmp_path / 'metadata-only')
    assert json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))['retrieval_mode'] == 'metadata_only'


def test_same_case_review_note_does_not_enter_prepared_model_context(memory_case, tmp_path):
    memory, request, current, _, _ = memory_case
    run = PROJECT / 'astra_runs/candidate_2_causal_retry1'
    if not (run / 'native_receipt.json').is_file():
        pytest.skip('Saved actual model archive absent; no model call is fabricated')
    note = memory.append_review(run)
    current['case_id'] = note['case_id']
    request.write_text(json.dumps(current))
    result = api()(memory, 'cube', request, tmp_path / 'without-hindsight')
    preparation = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert preparation['included_note_ids'] == []


def test_preparation_cannot_write_into_current_case_or_library(memory_case):
    memory, request, _, library, _ = memory_case
    for destination in (request.parent / 'nested', library / 'nested', 'E:/WAM/prepared'):
        with pytest.raises(ValueError, match='outside'):
            api()(memory, 'cube', request, destination)


def test_future_current_frame_is_rejected_before_reference_materialization(memory_case, tmp_path):
    memory, request, current, _, _ = memory_case
    current['available_frames']['live_front']['time_s'] = 5.01
    request.write_text(json.dumps(current))
    with pytest.raises(ValueError, match='future'):
        api()(memory, 'cube', request, tmp_path / 'future-input')
    assert not (tmp_path / 'future-input').exists()


def test_other_episode_note_is_compact_provisional_text_without_native_run_payloads(memory_case, tmp_path):
    memory, request, _, _, _ = memory_case
    run = PROJECT / 'astra_runs/candidate_2_causal_retry1'
    if not (run / 'native_receipt.json').is_file():
        pytest.skip('Saved actual model archive absent; no model call is fabricated')
    note = memory.append_review(run)
    result = api()(memory, 'cube', request, tmp_path / 'with-prior-note')
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    text = '\n'.join(part for part in manifest['content'] if isinstance(part, str))
    assert 'PROVISIONAL PRIOR-EPISODE MODEL NOTES' in text
    assert note['note_id'] in text
    assert str(run) not in text
    assert '"run_dir"' not in text
    assert '"source_artifacts"' not in text
    assert 'data:image/' not in text
    preparation = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert preparation['included_note_ids'] == [note['note_id']]
    assert preparation['model_calls'] == 0


def multiview_inspection(tmp_path):
    """Actual FFmpeg fixture with a future UAV component on the newer row."""
    from wam_astra.data import file_sha256
    root = tmp_path / 'inspection'
    episode = root / 'old-demo'
    phase = episode / 'phase_candidates'
    phase.mkdir(parents=True)
    video = episode / 'fixture.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
        'color=red:s=16x16:r=30:d=0.1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)],
        check=True, capture_output=True, timeout=30)
    alignment = episode / 'alignment.json'
    alignment.write_text(json.dumps({'episode_id': 'old-demo', 'clock_model_validated': True,
        'common_interval': {'start_desktop_ns': 9_900_000_000}}))
    cameras = episode / '相机时间戳'
    cameras.mkdir()
    (cameras / 'front.csv').write_text('source_time_ns,desktop_time_ns,clock_estimated_error_ns\n'
        '148020000000,10000000000,1000000\n148120000000,10100000000,1000000\n')
    rows = []
    for number in (0, 1):
        row = {'output_frame': str(number), 'output_time_s': str(number / 30)}
        for camera in ('front', 'wrist', 'drone'):
            row[camera + '_source_frame'] = str(number)
            row[camera + '_mapped_relative_time_s'] = '0' if number == 0 else ('.3' if camera == 'drone' else '.1')
            row[camera + '_clock_error_ms'] = '1'
        rows.append(row)
    index = episode / 'index.csv'
    with index.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (phase / 'provenance.json').write_text(json.dumps({'episode_id': 'old-demo', 'source_video': str(video),
        'alignment_path': str(alignment), 'alignment_sha256': file_sha256(alignment),
        'index_path': str(index), 'index_sha256': file_sha256(index)}))
    return root


def test_memory_review_uses_real_multiview_native_overlay_and_keeps_current(memory_case, tmp_path):
    memory, request, original, library, raw = memory_case
    inspection = multiview_inspection(tmp_path)
    before = [hashes(root) for root in (request.parent, library, raw, inspection)]
    result = api()(memory, 'cube', request, tmp_path / 'with-multiview', multiview_inspection_root=inspection)
    prepared = json.loads(Path(result['request_json']).read_text(encoding='utf-8'))
    assert prepared['available_frames']['live_front'] == original['available_frames']['live_front']
    frames = [frame for frame in prepared['available_frames'].values() if frame['camera'] == 'three_view']
    assert len(frames) == 1  # Same recorded composite is deduplicated across endpoints.
    frame = frames[0]
    assert frame['time_s'] == 0.
    assert frame['time_s_semantics'] == 'aligned_video_presentation_time'
    assert frame['clock_domain'] == 'aligned_desktop_common_interval'
    assert frame['source_episode_id'] == 'old-demo'
    assert set(frame['component_capture_bounds']) == {'front', 'wrist', 'drone'}
    assert frame['evidence_upper_desktop_ns'] <= frame['cutoff_lower_desktop_ns']
    assert sorted(f['time_s'] for f in prepared['available_frames'].values()
                  if f.get('evidence_role') == 'historical_reference' and f['camera'] == 'front') == [147.02, 147.12]
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    assert any('presentation time' in p and 'robot' in p for p in manifest['content'] if isinstance(p, str))
    receipt = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert receipt['multiview_enrichment'][0]['available_frame_count'] == 2
    assert len(receipt['reference_selection_receipts']) == 2
    assert receipt['native_text_budget']['validator'].endswith('validate_input_size')
    assert [hashes(root) for root in (request.parent, library, raw, inspection)] == before


def test_missing_multiview_sidecar_is_explicit_and_keeps_original_images(memory_case, tmp_path):
    memory, request, original, _, _ = memory_case
    result = api()(memory, 'cube', request, tmp_path / 'missing-multiview',
                   multiview_inspection_root=tmp_path / 'absent-inspection')
    prepared = json.loads(Path(result['request_json']).read_text(encoding='utf-8'))
    assert prepared['available_frames']['live_front'] == original['available_frames']['live_front']
    assert len(prepared['available_frames']) == 3
    receipt = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    enrichment = receipt['multiview_enrichment'][0]
    assert enrichment['status'] == 'unavailable'
    assert enrichment['available_frame_count'] == 0
    assert all(not frame['available'] for frame in json.loads(Path(enrichment['provenance_json']).read_text(encoding='utf-8'))['keyframes'])


@pytest.mark.parametrize('bad', ['future_component', 'missing_bound', 'image_hash', 'provenance_hash', 'negative_keyframe'])
def test_three_view_registration_rejects_unproved_or_future_receipts(tmp_path, bad):
    from wam_astra.data import file_sha256
    from wam_astra.memory_request import _three_view_metadata
    image = tmp_path / 'rgb.png'
    Image.new('RGB', (4, 3), 'red').save(image)
    components = {camera: {'capture_desktop_ns': 100, 'clock_error_ns': 1, 'upper_desktop_ns': 101}
                  for camera in ('front', 'wrist', 'drone')}
    proof = {'keyframe': 0, 'available': True, 'output_frame': 0,
        'alignment_index_row': {'output_time_s': '0'}, 'components': components,
        'evidence_upper_desktop_ns': 101, 'cutoff_lower_desktop_ns': 199,
        'requested_monotonic_ns': 1000, 'clock_anchor': {'source_time_ns': 999,
            'desktop_time_ns': 200, 'clock_estimated_error_ns': 1}, 'image_sha256': file_sha256(image)}
    if bad == 'future_component':
        components['drone']['capture_desktop_ns'] = 200
        components['drone']['upper_desktop_ns'] = proof['evidence_upper_desktop_ns'] = 201
    elif bad == 'missing_bound':
        components['drone'].pop('clock_error_ns')
    elif bad == 'image_hash':
        proof['image_sha256'] = '0' * 64
    elif bad == 'negative_keyframe':
        proof['keyframe'] = -1
    source = tmp_path / 'demo.json'
    frame = {'alignment': {'requested_monotonic_ns': 1000, 'camera_receipts': {'three_view': proof}}}
    source.write_text(json.dumps({'keyframes': [frame]}))
    provenance = tmp_path / 'provenance.json'
    provenance.write_text(json.dumps({'enriched_source_sha256': file_sha256(source), 'keyframes': [proof]}))
    entry = {'source_demo': str(source), 'multiview_enrichment': {'provenance_json': str(provenance),
        'sha256': file_sha256(provenance) if bad != 'provenance_hash' else '0' * 64}}
    with pytest.raises(ValueError, match='causal|proof|hash|bound'):
        _three_view_metadata(frame, image, entry)


def test_cli_multiview_option_runs_real_pipeline_with_explicit_missing_coverage(memory_case, tmp_path, monkeypatch, capsys):
    from wam_astra.cli import main
    memory, request, _, library, _ = memory_case
    target = tmp_path / 'cli-prepared'
    monkeypatch.setattr(sys, 'argv', ['wam-astra', 'prepare-memory-review', '--library', str(library),
        '--memory', str(memory.memory_root), '--task-id', 'cube', '--current-request', str(request),
        '--output', str(target), '--multiview-inspection-root', str(tmp_path / 'no-sidecars')])
    main()
    result = json.loads(capsys.readouterr().out)
    receipt = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert receipt['multiview_enrichment'][0]['status'] == 'unavailable'
    assert len(result['reference_selection_receipts']) == 2


def test_multiview_preserves_compact_prior_episode_notes(memory_case, tmp_path):
    memory, request, _, _, _ = memory_case
    run = PROJECT / 'astra_runs/candidate_2_causal_retry1'
    if not (run / 'native_receipt.json').is_file():
        pytest.skip('Saved actual model archive absent; no model call is fabricated')
    note = memory.append_review(run)
    result = api()(memory, 'cube', request, tmp_path / 'notes-and-multiview',
        multiview_inspection_root=tmp_path / 'missing-sidecars')
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    text = '\n'.join(part for part in manifest['content'] if isinstance(part, str))
    assert 'PROVISIONAL PRIOR-EPISODE MODEL NOTES' in text
    assert note['note_id'] in text
    assert str(run) not in text
    assert 'data:image/' not in text
    receipt = json.loads(Path(result['preparation_json']).read_text(encoding='utf-8'))
    assert receipt['included_note_ids'] == [note['note_id']]
