import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def source_demo(tmp_path):
    for name, color in [('front.png', 'red'), ('wrist.png', 'blue')]:
        Image.new('RGB', (12, 8), color).save(tmp_path / name)
    demo = {
        'schema_version': 1, 'source_robot': 'Piper',
        'numeric_reference_semantics': 'measured_states',
        'keyframes': [
            {'t_s': 0.0, 'images': {'front': 'front.png', 'wrist': 'wrist.png'},
             'state': {'piper': {'joint_measured_rad': [0., .1, .2, .3, .4, .5], 'gripper_measured_m': .02}},
             'action': {'samples': [
                 {'t_s': 0., 'piper': {'joint_measured_rad': [0.] * 6, 'gripper_measured_m': .02}},
                 {'t_s': .1, 'piper': {'joint_measured_rad': [.1] * 6}},
                 {'t_s': .2, 'piper': {'joint_measured_rad': [.2] * 6, 'gripper_measured_m': .0}},
             ]}},
            {'t_s': 1., 'images': {'front': 'front.png'},
             'state': {'piper': {'joint_measured_rad': [.2] * 6}}},
        ],
    }
    path = tmp_path / 'demo.json'
    path.write_text(json.dumps(demo), encoding='utf-8')
    return path, demo


def test_real_native_compiler_preserves_piper_and_exports_native_items(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    result = compile_demo(source, tmp_path.parent / (tmp_path.name + '-compiled'), 'Move using current observations.')
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    assert manifest['model'] == 'gpt-6-astra'
    assert manifest['instruction'] == 'Move using current observations.'
    text = '\n'.join(x for x in manifest['content'] if isinstance(x, str))
    assert 'HISTORICAL DEMONSTRATION' in text
    assert 'not issued controls' in text
    assert 'piper' in text
    assert 'Move using current observations.' in text
    actions = [json.loads(line) for line in Path(result['source_actions_jsonl']).read_text().splitlines()]
    assert actions[0]['action']['samples'] == demo['keyframes'][0]['action']['samples']
    assert 'gripper_measured_m' not in actions[0]['action']['samples'][1]['piper']
    exported = json.loads(Path(result['demo_json']).read_text())
    # Native sampling retains loss/recovery boundaries for missing feedback.
    assert len(exported['keyframes'][0]['action']['samples']) == 3
    assert exported['source_robot']['bimanual'] is False
    assert exported['source_robot']['arms'] == {'left': 'the sole physical Piper arm'}
    assert 'right' not in exported['keyframes'][0]['state']
    assert exported['keyframes'][0]['state']['left'] == demo['keyframes'][0]['state']['piper']
    for sample in actions[0]['action']['samples']:
        assert not {'left', 'right', 'command_sent', 'ack', 'acknowledged'} & sample.keys()
        assert not {'joint_target_rad', 'gripper_command'} & sample['piper'].keys()
    provenance = json.loads(Path(result['provenance_json']).read_text())
    assert provenance['adapter']['upstream_commit'] == '2ca201638e1139fe35973c8a2d209a894b389db1'
    assert provenance['adapter']['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance['input_action_samples'] == 3
    assert provenance['output_action_samples'] == 3
    assert provenance['adapter']['receipt_semantics'] == 'unknown unless explicitly present in source'
    items = json.loads(Path(result['app_server_items_json']).read_text())
    images = [x for x in items if x['type'] == 'image']
    assert len(images) == 2
    for item in images:
        pixels = Image.open(io.BytesIO(base64.b64decode(item['url'].split(',', 1)[1])))
        assert pixels.size == (12, 8)
    assert any(item.get('text') == manifest['instruction'] for item in items)
    assert source.read_text(encoding='utf-8') == json.dumps(demo)


def test_output_under_source_tree_is_rejected_before_creation(source_demo):
    from wam_astra.contexts import compile_demo
    source, _ = source_demo
    with pytest.raises(ValueError, match='source'):
        compile_demo(source, source.parent / 'nested', 'Do task')
    assert not (source.parent / 'nested').exists()


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_measured_values_rejected(source_demo, tmp_path, bad):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    demo['keyframes'][0]['state']['piper']['joint_measured_rad'][0] = bad
    source.write_text(json.dumps(demo))
    with pytest.raises(ValueError, match='finite|JSON'):
        compile_demo(source, tmp_path.parent / (tmp_path.name + '-bad-output'), 'Do task')


def test_missing_fields_stay_absent_and_images_are_byte_identical(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    result = compile_demo(source, tmp_path.parent / (tmp_path.name + '-bundle'), 'Observe task')
    exported = json.loads(Path(result['demo_json']).read_text())
    original = json.loads(Path(result['source_demo_json']).read_text())
    assert original == demo
    assert exported['keyframes'][1]['state']['left'] == demo['keyframes'][1]['state']['piper']
    assert 'action' not in exported['keyframes'][1]
    for frame in exported['keyframes']:
        for camera, image_path in frame['images'].items():
            assert (Path(result['destination']) / image_path).read_bytes() == (source.parent / demo['keyframes'][0]['images'][camera]).read_bytes()


def test_output_within_wam_is_rejected_without_writes():
    from wam_astra.contexts import compile_demo
    with pytest.raises(ValueError, match='E:/WAM|source'):
        compile_demo('E:/WAM/demo.json', 'E:/WAM/generated', 'Do task')


def test_plain_channels_use_real_native_periodic_compression(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    samples = [
        {'t_s': i * .1, 'piper': {'joint_measured_rad': [i * .01] * 6, 'gripper_measured_m': .02}}
        for i in range(5)
    ]
    demo['keyframes'][0]['action']['samples'] = samples
    source.write_text(json.dumps(demo))
    result = compile_demo(source, tmp_path.parent / (tmp_path.name + '-compressed'), 'Observe task')
    native = json.loads(Path(result['demo_json']).read_text())
    action = native['keyframes'][0]['action']
    assert [sample['t_s'] for sample in action['samples']] == [0., .4]
    assert action['compression']['method'] == 'periodic_and_gripper_events'
    assert action['compression']['input_samples'] == 5
    assert action['compression']['output_samples'] == 2
    raw = [json.loads(line) for line in Path(result['actions_jsonl']).read_text().splitlines()]
    assert len(raw[0]['action']['samples']) == 5
    assert set(raw[0]['action']['samples'][0]) == {'t_s', 'left'}
    assert 'gripper_command' not in raw[0]['action']['samples'][0]['left']


def test_schema_version_bool_is_rejected_before_native_execution(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    demo['schema_version'] = True
    source.write_text(json.dumps(demo))
    destination = tmp_path.parent / (tmp_path.name + '-schema')
    with pytest.raises(ValueError, match='schema_version'):
        compile_demo(source, destination, 'Observe task')
    assert not destination.exists()


def test_absolute_windows_image_paths_compile_through_actual_wsl(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, demo = source_demo
    for frame in demo['keyframes']:
        frame['images'] = {camera: str((source.parent / image).resolve())
                           for camera, image in frame['images'].items()}
    source.write_text(json.dumps(demo), encoding='utf-8')
    result = compile_demo(source, tmp_path.parent / (tmp_path.name + '-absolute-images'), 'Observe task')
    manifest = json.loads(Path(result['input_json']).read_text())
    image_parts = [item for item in manifest['content'] if isinstance(item, dict) and 'image' in item]
    assert len(image_parts) == 2
    assert all((Path(result['destination']) / item['image']).is_file() for item in image_parts)
    original = json.loads(Path(result['source_demo_json']).read_text())
    assert original == demo


def test_multiline_user_instruction_stays_literal_through_native_bridge(source_demo, tmp_path):
    from wam_astra.contexts import compile_demo
    source, _ = source_demo
    instruction = '无人机先观察目标\n低位遮挡正常，保留字面 "$((3+4))" 和 `printf marker`。\nUse real images.'
    result = compile_demo(source, tmp_path.parent / (tmp_path.name + '-literal-instruction'), instruction)
    manifest = json.loads(Path(result['input_json']).read_text(encoding='utf-8'))
    assert manifest['instruction'] == instruction
