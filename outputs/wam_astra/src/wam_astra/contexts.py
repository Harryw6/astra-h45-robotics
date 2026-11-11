"""Offline bridge to pinned, unmodified GPT-Policy demonstration compilation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys


UPSTREAM_COMMIT = '2ca201638e1139fe35973c8a2d209a894b389db1'


def _workspace() -> Path:
    return Path(__file__).resolve().parents[4]


def _wsl_path(path: Path) -> str:
    """Translate a resolved local drive path; UNC/network shares are unsupported."""
    if os.name != 'nt':
        return str(path)
    local = PureWindowsPath(path)
    if len(local.drive) != 2 or local.drive[1] != ':':
        raise ValueError('WSL adapter requires an absolute local drive path')
    return '/mnt/' + local.drive[0].lower() + '/' + '/'.join(local.parts[1:])


def _under_wam(path: str | Path) -> bool:
    text = str(path).replace('\\', '/').rstrip('/').casefold()
    return text == 'e:/wam' or text.startswith('e:/wam/') or text == '/mnt/e/wam' or text.startswith('/mnt/e/wam/')


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Source JSON numbers must be finite')
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def compile_demo(
    demo_json: str | Path,
    destination: str | Path,
    instruction: str,
    *,
    upstream_root: str | Path | None = None,
    vendor_root: str | Path | None = None,
    distro: str = 'Ubuntu',
    timeout_s: float = 120.0,
) -> dict[str, str]:
    """Compile reviewed Piper keyframes through actual GPT-Policy code, offline.

    ``destination`` must not exist or be inside E:/WAM or the demo source
    directory. Linux runs directly; Windows uses WSL Ubuntu. Dependencies are
    provided separately, never installed implicitly. No model or robot calls.
    Native left means the sole physical Piper arm, never a second arm.
    """
    if _under_wam(destination):
        raise ValueError('Output must be outside E:/WAM source data')
    source = Path(demo_json).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if _under_wam(target):
        raise ValueError('Output must be outside E:/WAM source data')
    if target == source.parent or source.parent in target.parents:
        raise ValueError('Output must be outside the source directory')
    if target.exists():
        raise ValueError(f'Output already exists: {target}')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be nonempty text')
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout_s must be finite and positive')
    demo = json.loads(source.read_text(encoding='utf-8-sig'))
    _finite_json(demo)
    if not isinstance(demo, dict) or type(demo.get('schema_version')) is not int or demo['schema_version'] != 1:
        raise ValueError('Source JSON requires schema_version 1')
    repo = Path(upstream_root or _workspace() / 'work/upstream/GPT-Policy').resolve()
    vendor = Path(vendor_root or _workspace() / 'work/wsl_vendor').resolve()
    worker = Path(__file__).with_name('native_context_worker.py').resolve()
    args = [
        _wsl_path(worker), '--source', _wsl_path(source), '--destination', _wsl_path(target),
        '--instruction', instruction.strip(), '--upstream-root', _wsl_path(repo),
    ]
    pythonpath = ':'.join((_wsl_path(repo / 'src'), _wsl_path(vendor))) if os.name == 'nt' else os.pathsep.join((str(repo / 'src'), str(vendor)))
    if os.name == 'nt':
        # --exec bypasses the default Linux shell: user instructions must stay
        # literal, including newlines, quotes, dollars and backticks.
        command = ['wsl.exe', '-d', distro, '--exec', 'env', 'PYTHONPATH=' + pythonpath, 'python3', *args]
        env = None
    else:
        command = [sys.executable, *args]
        env = {**os.environ, 'PYTHONPATH': pythonpath}
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Native context compilation timed out') from exc
    if completed.returncode:
        detail = completed.stderr.decode('utf-8', errors='replace').replace('\x00', '').strip()
        raise RuntimeError(f'Native GPT-Policy compilation failed: {detail[-4000:]}')
    required = {
        'input_json': 'input.json', 'actions_jsonl': 'actions.jsonl',
        'provenance_json': 'provenance.json', 'demo_json': 'demo.json',
        'source_demo_json': 'source_demo.json', 'source_actions_jsonl': 'source_actions.jsonl',
        'app_server_items_json': 'app_server_items.json',
    }
    for filename in required.values():
        if not (target / filename).is_file():
            raise RuntimeError(f'Native compiler did not produce {filename}')
    return {'destination': str(target), **{key: str(target / value) for key, value in required.items()}}
