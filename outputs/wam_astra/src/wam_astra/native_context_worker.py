"""Subprocess entry point; imports native GPT-Policy only inside Linux/WSL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import tempfile


UPSTREAM_COMMIT = '2ca201638e1139fe35973c8a2d209a894b389db1'


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _validate_piper(value):
    """Reject malformed observed vectors, leaving absent fields absent."""
    if not isinstance(value, dict):
        raise ValueError('piper data must be an object')
    q = value.get('joint_measured_rad')
    if q is not None and (not isinstance(q, list) or len(q) != 6 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in q)):
        raise ValueError('Piper joint_measured_rad requires six finite numeric values')
    g = value.get('gripper_measured_m')
    if g is not None and (isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g)):
        raise ValueError('Piper gripper_measured_m requires finite numeric metres')


def _alias(container):
    if not isinstance(container, dict):
        raise ValueError('State/sample must be an object')
    if 'left' in container or 'right' in container:
        raise ValueError('Canonical Piper source must not contain left/right arm namespaces')
    if 'piper' in container:
        _validate_piper(container['piper'])
        container['left'] = container.pop('piper')


def compile_native(source: Path, destination: Path, instruction: str, upstream_root: Path):
    commit = subprocess.run(['git', '-c', f'safe.directory={upstream_root}', '-C', str(upstream_root), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f'GPT-Policy must be pinned to {UPSTREAM_COMMIT}')
    # These imports are the real upstream implementation, not adapter copies.
    from gpt_policy.input.demonstration import prepare_demonstration
    from gpt_policy.input.request import RunInput
    from gpt_policy.input.manifest import TextPart
    from gpt_policy.harness.input_content import to_app_server_items, validate_input_size

    original = source.read_bytes()
    demo = json.loads(original.decode('utf-8-sig'))
    mapped = copy.deepcopy(demo)
    mapped['source_robot'] = {'model': 'AgileX Piper', 'arms': {'left': 'the sole physical Piper arm'}, 'bimanual': False}
    frames = mapped.get('keyframes')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Reviewed source requires keyframes')
    for frame in frames:
        if 'state' in frame:
            _alias(frame['state'])
        for sample in frame.get('action', {}).get('samples', []):
            _alias(sample)
        for camera, image in frame.get('images', {}).items():
            windows = PureWindowsPath(image)
            if windows.drive:
                if len(windows.drive) != 2 or windows.drive[1] != ':' or not windows.is_absolute():
                    raise ValueError('Image requires an absolute local drive path; UNC and drive-relative paths are unsupported')
                path = Path('/mnt', windows.drive[0].lower(), *windows.parts[1:]) if os.name != 'nt' else Path(image)
            else:
                path = Path(image.replace('\\', '/'))
            if not path.is_absolute():
                path = source.parent / path
            frame['images'][camera] = str(path.resolve(strict=True))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.piper-context-', dir=destination.parent))
    try:
        native_source = temporary / 'native_demo.json'
        _save(native_source, mapped)
        bundle = temporary / 'bundle'
        alias_note = (
            'Piper compatibility note: canonical piper fields are named left in this historical '
            'context solely for the GPT-Policy single-arm input namespace. left denotes the sole '
            'physical Piper arm; no right arm exists. gripper_measured_m is measured aperture in '
            'metres, not normalized opening or a command. Missing feedback, sent-command receipts '
            'and hardware acknowledgements remain unknown unless recorded explicitly.'
        )
        run = RunInput(instruction, 'gpt-6-astra', (TextPart(alias_note), TextPart(instruction)))
        prepared, metadata = prepare_demonstration(run, native_source, 'video+action', bundle)
        validate_input_size(prepared.content)
        _save(bundle / 'app_server_items.json', to_app_server_items(prepared.content))
        (bundle / 'source_demo.json').write_bytes(original)
        with (bundle / 'source_actions.jsonl').open('w', encoding='utf-8') as stream:
            for i, frame in enumerate(demo['keyframes']):
                if 'action' in frame:
                    stream.write(json.dumps({'keyframe': i, 'action': frame['action']}, ensure_ascii=False, allow_nan=False) + '\n')
        metadata['adapter'] = {
            'upstream': 'https://github.com/cheng-haha/GPT-Policy', 'upstream_commit': commit,
            'source': str(source), 'source_sha256': hashlib.sha256(original).hexdigest(),
            'native_input_sha256': hashlib.sha256(native_source.read_bytes()).hexdigest(),
            'namespace_mapping': {'piper': 'left'}, 'bimanual': False,
            'namespace_semantics': 'left is a compatibility alias for the sole physical Piper arm; no right arm exists',
            'receipt_semantics': 'unknown unless explicitly present in source',
            'gripper_semantics': 'gripper_measured_m remains metres, never normalized; native gripper event detector does not recognize this field',
            'sampling': 'unchanged GPT-Policy compress_samples; periodic endpoints and recognized native events only',
            'source_robot_original': demo.get('source_robot'),
            'model_calls': 0,
        }
        _save(bundle / 'provenance.json', metadata)
        if source.read_bytes() != original:
            raise RuntimeError('Source demonstration changed during compilation')
        if destination.exists():
            raise ValueError('Destination already exists')
        os.rename(bundle, destination)
    finally:
        shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'destination', 'instruction', 'upstream-root'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    compile_native(Path(args.source), Path(args.destination), args.instruction, Path(args.upstream_root))


if __name__ == '__main__':
    main()
