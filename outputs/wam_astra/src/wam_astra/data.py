"""Read-only adapters for WAM source records; no robot control or relabeling."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = [f'joint_{i}.pos' for i in range(1, 7)]
EPISODE_RE = re.compile(r'^joint_\d{8}_\d{6}_UTCp0800_[0-9a-f]+$')


def ensure_separate_output(source_root, output):
    source, target = Path(source_root).resolve(), Path(output).resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError('Generated output must be outside the source dataset')
    return target


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path, default=None):
    if not Path(path).is_file():
        return {} if default is None else default
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inventory(root):
    root = Path(root).resolve()
    rows = []
    for parent, directories, names in os.walk(root):
        directories.sort()
        for name in sorted(names):
            path = Path(parent) / name
            stat = path.stat()
            rows.append({'path': path.relative_to(root).as_posix(), 'size': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns})
    return sorted(rows, key=lambda row: row['path'])


def compare_inventory(before, after):
    left = {r['path']: (r['size'], r['mtime_ns']) for r in before}
    right = {r['path']: (r['size'], r['mtime_ns']) for r in after}
    changed = sorted(k for k in left.keys() & right.keys() if left[k] != right[k])
    added, removed = sorted(right.keys()-left.keys()), sorted(left.keys()-right.keys())
    return {'unchanged': not (changed or added or removed), 'changed': changed,
            'added': added, 'removed': removed,
            'verification': 'path, size, and mtime_ns for every source file; parsed-file SHA256 is recorded separately'}


def discover_episodes(root):
    root = Path(root).resolve()
    found = {}
    for day in sorted(p for p in root.iterdir() if p.is_dir()):
        base = day / '完整采集数据'
        for kind, parent in [('robot', base/'原始数据'/'机器狗'),
                             ('drone', base/'原始数据'/'无人机'),
                             ('joint', base/'joint_metadata')]:
            if not parent.is_dir():
                continue
            for directory in sorted(parent.iterdir()):
                if not directory.is_dir() or not EPISODE_RE.fullmatch(directory.name):
                    continue
                key = directory.name
                row = found.setdefault(key, {'episode_id': key, 'date': day.name, 'included': True,
                                             'robot_dir': None, 'drone_dir': None, 'joint_dir': None,
                                             'alignment_path': None})
                actual = directory / key if kind == 'robot' and (directory/key).is_dir() else directory
                row[kind+'_dir'] = str(actual)
        for key, row in found.items():
            if row['date'] == day.name:
                path = base/'对齐数据'/key/'alignment.json'
                if path.is_file():
                    row['alignment_path'] = str(path)
    return [found[key] for key in sorted(found)]


def read_piper_csv(path, *, command=False):
    path = Path(path)
    if not path.is_file():
        return None
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = ['monotonic_ns', 'wall_time_ns', 'seq', *JOINTS, 'gripper.pos']
    if command:
        wanted += ['command_sent', 'arm_connected', 'arm_enabled', 'gamepad_connected']
    use = [key for key in wanted if key in header]
    if 'monotonic_ns' not in use or not all(key in use for key in JOINTS):
        raise ValueError(f'Missing time/joint columns in {path}')
    time_types = {key: 'string' for key in ('monotonic_ns', 'wall_time_ns', 'seq') if key in use}
    frame = pd.read_csv(path, usecols=use, dtype=time_types)
    def integer_or_na(value):
        try:
            integer=int(value)
            return integer if -(2**63)<=integer<2**63 else pd.NA
        except (ValueError,TypeError,OverflowError):
            return pd.NA
    for key in time_types:
        frame[key]=pd.array([integer_or_na(v) for v in frame[key]],dtype='Int64')
    times = frame['monotonic_ns']
    valid_time = times.notna() & (times > 0)
    q = frame[JOINTS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    valid_q = np.isfinite(q).all(axis=1)
    valid = valid_time.to_numpy(dtype=bool) & valid_q
    indices = np.flatnonzero(valid)
    result = {'path': str(path), 'sha256': file_sha256(path), 'columns': header,
              'source_row': indices.astype(np.int64), 'q': q[valid],
              't_ns': times[valid].to_numpy(dtype=np.int64),
              'invalid_rows': np.flatnonzero(~valid).tolist(), 'source_rows': len(frame)}
    if 'wall_time_ns' in frame:
        result['wall_ns'] = frame.loc[valid, 'wall_time_ns'].fillna(-1).to_numpy(dtype=np.int64)
    else:
        result['wall_ns'] = np.full(len(indices), -1, dtype=np.int64)
    result['g'] = (pd.to_numeric(frame.loc[valid, 'gripper.pos'], errors='coerce').to_numpy(dtype=float)
                   if 'gripper.pos' in frame else np.full(len(indices), np.nan))
    for key in ['command_sent', 'arm_connected', 'arm_enabled', 'gamepad_connected']:
        if key in frame:
            result[key] = pd.to_numeric(frame.loc[valid,key], errors='coerce').fillna(-1).to_numpy(dtype=np.int8)
    # Source ordering is evidence. Do not sort a time reset into a false trajectory.
    result['clock_reset_indices'] = (np.flatnonzero(np.diff(result['t_ns']) <= 0)+1).tolist()
    return result


def causal_at(times, values, query_ns, *, max_age_ns=250_000_000):
    if len(times) == 0:
        return None
    index = int(np.searchsorted(times, query_ns, side='right') - 1)
    if index < 0 or query_ns-int(times[index]) > max_age_ns:
        return None
    value = np.asarray(values[index], dtype=float)
    if not np.isfinite(value).all():
        return None
    return {'index': index, 'value': value.tolist(), 'age_ns': int(query_ns)-int(times[index])}


def fixed_windows(start_ns, end_ns, *, count=12, window_ns=1_000_000_000):
    if window_ns <= 0 or count < 1:
        raise ValueError('Window length and count must be positive')
    available = (int(end_ns)-int(start_ns)) // window_ns
    size = min(count, max(0, available))
    if size == 0:
        return []
    if size == 1:
        return [(int(start_ns), int(start_ns)+window_ns)]
    last = int(end_ns)-window_ns
    starts = [int(start_ns) + ((last-int(start_ns))*i)//(size-1) for i in range(size)]
    return [(value, value+window_ns) for value in starts]


def receipt_record(*, command_rows, measured_rows, tracking):
    return {'kind': 'recorded_evidence_not_hardware_ack', 'command_sent_rows': list(command_rows),
            'measured_state_rows': list(measured_rows), 'tracking': tracking,
            'hardware_acknowledged': None, 'task_success': None,
            'interpretation': 'Recorder command_sent is not an actuator acknowledgement or grasp-success label.'}


def save_signals(path, state, commands):
    values = {}
    for prefix, signal in [('state', state), ('command', commands)]:
        if signal is None:
            continue
        for key, value in signal.items():
            if isinstance(value, np.ndarray):
                values[prefix+'_'+key] = value
    np.savez_compressed(path, **values)


def observation_at(state, query_ns, *, max_age_ns=250_000_000):
    sample = causal_at(state['t_ns'], state['q'], query_ns, max_age_ns=max_age_ns)
    if sample is None:
        return {'requested_monotonic_ns': int(query_ns), 'piper': None, 'issue': 'missing_or_stale_state'}
    index = sample['index']
    g = float(state['g'][index])
    return {'requested_monotonic_ns': int(query_ns), 'state_monotonic_ns': int(state['t_ns'][index]),
            'source_row': int(state['source_row'][index]), 'age_ns': sample['age_ns'],
            'piper': {'joint_measured_rad': sample['value'],
                      'gripper_measured_m': g if np.isfinite(g) else None}}


def action_segment(commands, start_ns, end_ns):
    if commands is None:
        return {'kind': 'unrecorded_commands', 'samples': [], 'command_sent_rows': []}
    lo = int(np.searchsorted(commands['t_ns'], start_ns, side='left'))
    hi = int(np.searchsorted(commands['t_ns'], end_ns, side='left'))
    rows, sent = [], []
    flags = commands.get('command_sent')
    for index in range(lo, hi):
        flag = int(flags[index]) if flags is not None else -1
        if flag != 1:
            continue
        source_row = int(commands['source_row'][index]); sent.append(source_row)
        g = float(commands['g'][index])
        rows.append({'monotonic_ns': int(commands['t_ns'][index]), 'source_row': source_row,
                     'piper': {'recorded_joint_target': commands['q'][index].tolist(),
                               'recorded_gripper_target': g if np.isfinite(g) else None},
                     'command_sent': True})
    return {'kind': 'recorded_teleop_targets', 'samples': rows, 'command_sent_rows': sent,
            'target_units': 'preserved_source_values; verify recorder-specific target units before using as motion commands'}


def measured_action(state, start_ns, end_ns, origin_ns):
    """Archive measurements; ask native GPT-Policy to keep every aperture change.

    No metre-to-normalized conversion or grasp/event classifier is used. Even
    tiny recorded changes are preserved, including possible measurement noise.
    """
    lo,hi=np.searchsorted(state['t_ns'],[start_ns,end_ns],side='left')
    samples=[]
    for index in range(int(lo),int(hi)):
        g=float(state['g'][index])
        piper={'joint_measured_rad':state['q'][index].tolist()}
        if np.isfinite(g):piper['gripper_measured_m']=g
        samples.append({'t_s':(int(state['t_ns'][index])-int(origin_ns))/1e9,
                        'source_row':int(state['source_row'][index]),'piper':piper})
    keep=set()
    for index in range(1,len(samples)):
        if samples[index-1]['piper'].get('gripper_measured_m') != samples[index]['piper'].get('gripper_measured_m'):
            keep.update((index-1,index))
    return {'samples':samples,'preserve_sample_indices':sorted(keep),
            'semantics':'historical_measured_states_not_issued_controls',
            'anchor_semantics':'both sides of every recorded metre-valued gripper change, including noise; not grasp labels'}
