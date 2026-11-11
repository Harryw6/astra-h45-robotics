"""16.7 coordinator: three live cameras -> V3 policy -> bounded three-robot chunks."""
import argparse
import base64
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import threading
import time
import urllib.request
import urllib.error

import numpy as np

from .core import DT, SLOTS, decode_observation, finite, device_names, masking_metadata
from .relative14 import check_policy_metadata, policy_observation
from . import displacement14


class Device:
    def __init__(self, url, token, timeout=2):
        self.url, self.token, self.timeout = url.rstrip('/'), token, timeout
        self.offset, self.uncertainty = 0., float('inf')
        self.response_timeouts = {}
        self.clock_checked = 0.
        # Local SSH tunnels must never be sent through HTTP_PROXY.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, endpoint, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self.url+endpoint, data=data,
            headers={'Authorization': 'Bearer '+self.token, 'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.response_timeouts.get(endpoint, self.timeout)) as response:
                payload = response.read(8*1024*1024+1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self.url+': '+exc.read(4096).decode(errors='replace')) from exc
        if len(payload) > 8*1024*1024:
            raise ValueError('Oversized observation response')
        return json.loads(payload)

    def calibrate(self):
        candidates = []
        for _ in range(5):
            before = time.monotonic()
            status = self.call('/status')
            after = time.monotonic()
            candidates.append(((after-before)/2, status['monotonic']-(before+after)/2))
        self.uncertainty, self.offset = min(candidates)
        if not np.isfinite([self.uncertainty, self.offset]).all() or self.uncertainty > .04:
            raise RuntimeError('Gateway clock uncertainty %.1f ms exceeds 40 ms '
                               '(best roundtrip %.1f ms): %s' %
                               (self.uncertainty*1000, self.uncertainty*2000, self.url))
        self.clock_checked = time.monotonic()
        return self.call('/status')


class Policy:
    def __init__(self, url, timeout=120):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.codec, self.packer, self.timeout = msgpack_numpy, msgpack_numpy.Packer(), timeout
        self.connection = connect(url, compression=None, max_size=8*1024*1024,
                                  proxy=None, ping_interval=None, open_timeout=10)
        self.metadata = msgpack_numpy.unpackb(self.connection.recv(timeout=10))
        if (self.metadata.get('physical_action_dim') != 14 or self.metadata.get('action_horizon') != 15
                or self.metadata.get('fps') != 15 or self.metadata.get('vit_lora_rank') != 8
                or self.metadata.get('vit_lora_alpha') != 16):
            self.close()
            raise ValueError('Server is not the V3 H15/14D ViT-LoRA r8/a16 policy')
        try:
            width = self.metadata.get('physical_state_dim', 30)
            if self.metadata.get('schema_id') == displacement14.SCHEMA_ID:
                displacement14.check_policy_metadata(self.metadata)
            elif width == 14:
                check_policy_metadata(self.metadata)
            elif width != 30:
                raise ValueError('Unsupported policy state dimension')
        except Exception:
            self.close()
            raise

    def prepare_observation(self, observation):
        if self.metadata.get('schema_id') == displacement14.SCHEMA_ID:
            return displacement14.policy_observation(observation, self.metadata)
        return policy_observation(observation, self.metadata)

    def infer(self, observation):
        observation = self.prepare_observation(observation)
        self.connection.send(self.packer.pack(observation))
        response = self.connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError('Policy server error: '+response)
        self.last_policy_actions = finite(self.codec.unpackb(response)['actions'], (15, 14)).copy()
        if self.metadata.get('schema_id') == displacement14.SCHEMA_ID:
            return displacement14.execution_actions(self.last_policy_actions, self.metadata)
        return self.last_policy_actions.copy()

    def close(self):
        self.connection.close()


def parallel(devices, method, bodies=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(device.call, method[name] if isinstance(method, dict) else method,
                                    None if bodies is None else bodies[name])
                   for name, device in devices.items()}
        return {name: future.result() for name, future in futures.items()}


def displacement_policy(policy):
    return getattr(policy, 'metadata', {}).get('schema_id') == displacement14.SCHEMA_ID


def prediction_metadata(policy):
    if not displacement_policy(policy):
        return {}
    return dict(policy_actions=policy.last_policy_actions.tolist(),
                policy_action_encoding=displacement14.OUTPUT_ENCODING,
                predicted_actions_encoding='physical_joint_targets_and_heading_velocities')


def execution_devices(devices, targets=None):
    targets = tuple(devices) if targets is None else tuple(targets)
    if targets not in (('unitree', 'p450'), ('unitree',)):
        raise ValueError('Execution targets must be both gateways or unitree only')
    return {name: devices[name] for name in targets}


def observer_status(devices, selected):
    observers = {name: device for name, device in devices.items() if name not in selected}
    statuses = parallel(observers, '/status')
    for name, status in statuses.items():
        if status['execution_enabled'] or status['active'] or status['execution'] is not None:
            raise RuntimeError('Observation-only gateway must have execution disabled and no plan: '+name)
    return statuses


def stop_all(devices):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {name: pool.submit(device.call, '/stop', {}) for name, device in devices.items()}
        for name, future in pending.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {'error': str(exc)}
    return results


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    os.replace(temporary, path)


def save_observations(folder, packets, extra=None):
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, packet in packets.items():
        metadata[name] = {k: v for k, v in packet.items() if k != 'images'}
        for camera, data in packet['images'].items():
            path = folder/(camera+'.jpg')
            temp = path.with_suffix('.tmp')
            temp.write_bytes(base64.b64decode(data, validate=True))
            os.replace(temp, path)
    atomic_json(folder/'observation.json', dict(devices=metadata, **(extra or {})))


def run_cycle(devices, policy, config, output, execute=False, cycle=0, targets=None):
    selected = execution_devices(devices, targets)
    try:
        return _run_cycle(devices, policy, config, output, execute, cycle, tuple(selected))
    except BaseException:
        if execute:
            stop_all(selected)
        raise


def _run_cycle(devices, policy, config, output, execute=False, cycle=0, targets=None):
    selected = execution_devices(devices, targets)
    observer_status(devices, selected)
    endpoints = {name: '/observation' if name in selected else '/diagnostic' for name in devices}
    packets = parallel(devices, endpoints)
    clocks = {name: (d.offset, d.uncertainty) for name, d in devices.items()}
    if any(time.monotonic()-d.clock_checked > 30 for d in devices.values()):
        raise RuntimeError('Clock calibration expired')
    mask_info = masking_metadata(config)
    if mask_info and getattr(policy, 'metadata', {}).get('physical_state_dim', 30) != 30:
        raise ValueError('Unitree-only state fill is calibrated for the state30 checkpoint')
    observation = decode_observation(packets['unitree'], packets.get('p450'), clocks, config['prompt'],
                                     p450_state_fill=config.get('p450_state_fill'))
    save_observations(output/'latest', packets, {'cycle': cycle, 'phase': 'before_inference', **mask_info})
    started = time.monotonic()
    actions = finite(policy.infer(observation), (15, 14))
    elapsed = time.monotonic()-started
    record = dict(cycle=cycle, inference_s=elapsed, predicted_actions=actions.tolist(), executed=False,
                  execution_targets=list(selected),
                  observation_only=[name for name in devices if name not in selected],
                  observer_readiness_issues={name: packet.get('readiness_issues', [])
                      for name, packet in packets.items() if name not in selected})
    record.update(mask_info)
    record.update(prediction_metadata(policy))
    record['unitree_action_chunk'] = actions[:, :10].tolist()
    record['observations_before'] = {name:{k:v for k,v in packet.items() if k != 'images'} for name,packet in packets.items()}
    atomic_json(output/'latest/actions.json', record)
    if not execute:
        return record
    steps = config.get('execute_steps', 15)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 15:
        raise ValueError('execute_steps must be 1..15')
    if sum(d.uncertainty for d in selected.values()) > .04:
        raise RuntimeError('Clock uncertainty too large for a coordinated start')
    observer_status(devices, selected)
    prepared = parallel(selected, '/prepare', {
        name: dict(session=packets[name]['session'], observation_id=packets[name]['observation_id'],
                   actions=(actions[:steps, :10] if name == 'unitree' else actions[:steps, 10:]).tolist())
        for name in selected})
    start = time.monotonic()+.2
    commits = parallel(selected, '/commit', {
        name: dict(plan_id=prepared[name]['plan_id'], start_at=start+devices[name].offset)
        for name in selected})
    deadline = start+steps*DT+1.
    while True:
        statuses = parallel(selected, '/status')
        if any(s['fault'] for s in statuses.values()):
            raise RuntimeError('A device faulted: '+json.dumps(statuses))
        if all(s['execution'] and s['execution']['status'] == 'done' for s in statuses.values()):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Execution acknowledgement deadline exceeded')
        time.sleep(.02)
    starts = [statuses[name]['execution']['actual_start']-devices[name].offset for name in selected]
    if max(starts)-min(starts)+sum(d.uncertainty for d in selected.values()) > .05:
        raise RuntimeError('Cross-device execution skew exceeds 50 ms')
    # Request a frame acquired after both chunks completed, not a replayed pre-action image.
    finished = max(statuses[name]['execution']['finished_at']-devices[name].offset
                   for name in selected)
    frame_deadline = time.monotonic()+.5
    while True:
        after = parallel(devices, endpoints)
        if all(min(after[name]['image_stamps'].values())-devices[name].offset > finished
               for name in devices):
            break
        if time.monotonic() > frame_deadline:
            raise RuntimeError('No new camera frames after execution')
        time.sleep(.02)
    decode_observation(after['unitree'], after.get('p450'), clocks, config['prompt'],
                       p450_state_fill=config.get('p450_state_fill'))
    record.update(executed=True, executed_steps=steps, execution= statuses,
                  observation_only_status=observer_status(devices, selected),
                  applied_actions={name: value['applied_actions'] for name, value in prepared.items()})
    record['observations_after'] = {name:{k:v for k,v in packet.items() if k != 'images'} for name,packet in after.items()}
    save_observations(output/'latest', after, {'cycle': cycle, 'phase': 'after_execution', **mask_info})
    atomic_json(output/'latest/actions.json', record)
    return record


class StreamHeartbeat:
    """Keep waiting-for-inference distinct from a disconnected controller."""
    def __init__(self, devices, sessions):
        self.devices = devices
        self.bodies = {name: dict(session=sessions[name]) for name in devices}
        self.cancel = threading.Event()
        self.error = None
        self.thread = None

    def start(self):
        if not self.devices:
            return
        parallel(self.devices, '/stream/heartbeat', self.bodies)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.cancel.wait(.15):
            try:
                parallel(self.devices, '/stream/heartbeat', self.bodies)
            except Exception as exc:
                self.error = str(exc)
                return

    def check(self):
        if self.error:
            raise RuntimeError('Inference controller heartbeat failed: '+self.error)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=3)


def run_action_stream(devices, policy, config, output, cycles, targets=None):
    """Infer serially while device-local executors consume timestamped H15 buffers."""
    selected = execution_devices(devices, targets)
    endpoints = {name: '/observation' if name in selected else '/diagnostic' for name in devices}
    initial = parallel(selected, '/status')
    if any('stream_actions_v1' not in s.get('capabilities', []) for s in initial.values()):
        raise RuntimeError('Deploy stream_actions_v1 gateways before streaming execution')
    holding = {name: device for name, device in selected.items()
               if name == 'unitree' and not displacement_policy(policy)}
    if any('stream_hold_heartbeat_v1' not in initial[n].get('capabilities', []) for n in holding):
        raise RuntimeError('Deploy stream_hold_heartbeat_v1 before last-action holding')
    heartbeats = StreamHeartbeat(holding, {n: initial[n]['session'] for n in holding})
    records = []
    try:
        heartbeats.start()
        for cycle in range(cycles):
            heartbeats.check()
            observer_status(devices, selected)
            if any(time.monotonic()-d.clock_checked > 10 for d in devices.values()):
                for device in devices.values():
                    device.calibrate()
            if sum(d.uncertainty for d in selected.values()) > .04:
                raise RuntimeError('Clock uncertainty too large for coordinated streaming')
            waiting_since = time.monotonic()
            while True:
                try:
                    packets = parallel(devices, endpoints)
                    clocks = {name: (d.offset, d.uncertainty) for name, d in devices.items()}
                    observation = decode_observation(packets['unitree'], packets.get('p450'), clocks,
                        config['prompt'], p450_state_fill=config.get('p450_state_fill'))
                    candidate_origin = min(p['stamp']-devices[name].offset
                                           for name, p in packets.items())
                    if records and candidate_origin <= records[-1]['prediction_origin']:
                        # An inference faster than the cameras must wait for a
                        # genuinely new observation instead of replacing the
                        # buffer with another prediction of the same frame.
                        time.sleep(.002)
                        continue
                    break
                except (RuntimeError, OSError) as exc:
                    states = parallel(selected, '/status')
                    if any(s['fault'] for s in states.values()):
                        raise RuntimeError('Action stream faulted while waiting for model input: '+str(exc)) from exc
                    if records and not any(s['active'] for s in states.values()):
                        raise RuntimeError('Action buffer exhausted while waiting for model input') from exc
                    if not records and time.monotonic()-waiting_since > 3:
                        raise RuntimeError('No valid initial model observation') from exc
                    # No replacement/zero command is sent: the onboard buffer
                    # consumes future steps. Velocity policies may hold the last
                    # action; displacement policies expire at the horizon.
                    heartbeats.check()
                    time.sleep(.01)
            save_observations(output/'latest', packets,
                              {'cycle': cycle, 'phase': 'before_inference', **masking_metadata(config)})
            started = time.monotonic()
            actions = finite(policy.infer(observation), (15, 14))
            heartbeats.check()
            inference_s = time.monotonic()-started
            origin = min(packet['stamp']-devices[name].offset for name, packet in packets.items())
            prepared = parallel(selected, '/stream/prepare', {
                name: dict(session=packets[name]['session'], observation_id=packets[name]['observation_id'],
                           origin_at=origin+devices[name].offset,
                           hold_last_action=name == 'unitree' and not displacement_policy(policy),
                           actions=(actions[:, :10] if name == 'unitree' else actions[:, 10:]).tolist())
                for name in selected})
            lead = float(config.get('stream_start_lead_s', .05))
            if not np.isfinite(lead) or not .025 <= lead <= .2:
                raise ValueError('Stream scheduling lead must be 25..200 ms')
            lead += 2*max(d.uncertainty for d in selected.values())
            start = time.monotonic()+lead
            commits = parallel(selected, '/stream/commit', {
                name: dict(plan_id=prepared[name]['plan_id'], start_at=start+devices[name].offset)
                for name in selected})
            record = dict(cycle=cycle, execution_mode='stream', submitted=True, executed=False,
                inference_s=inference_s, prediction_origin=origin, predicted_actions=actions.tolist(),
                unitree_action_chunk=actions[:, :10].tolist(), submitted_steps=15,
                execution_targets=list(selected), observation_only=[n for n in devices if n not in selected],
                prepared=prepared, commits=commits,
                applied_actions={name: p['applied_actions'] for name, p in prepared.items()},
                observations_before={name: {k: v for k, v in packet.items() if k != 'images'}
                                     for name, packet in packets.items()})
            record.update(masking_metadata(config))
            record.update(prediction_metadata(policy))
            records.append(record)
            atomic_json(output/'latest/actions.json', record)
            with (output/'cycles.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            print(json.dumps(dict(cycle=cycle, inference_s=inference_s, submitted_steps=15,
                                  scheduling_lead_s=lead)), flush=True)
        parallel(selected, '/stream/finish', {name: {} for name in selected})
        deadline = time.monotonic()+1.5
        while True:
            statuses = parallel(selected, '/status')
            if any(s['fault'] for s in statuses.values()):
                raise RuntimeError('Action stream faulted: '+json.dumps(statuses))
            if all(not s['active'] and s['stream']['finished'] for s in statuses.values()):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Action stream did not finish within its prediction horizon')
            time.sleep(.01)
        for record in records:
            record['prediction_execution'] = {}
            for name in selected:
                plan_id = record['prepared'][name]['plan_id']
                prediction = next(p for p in statuses[name]['stream']['predictions']
                                  if p['plan_id'] == plan_id)
                record['prediction_execution'][name] = prediction
            record['executed'] = any(p['executed_steps'] > 0
                                     for p in record['prediction_execution'].values())
            starts = [p['actual_start']-devices[name].offset
                      for name, p in record['prediction_execution'].items()
                      if p['executed_steps'] > 0]
            if len(starts) > 1 and max(starts)-min(starts)+sum(
                    d.uncertainty for d in selected.values()) > .05:
                raise RuntimeError('Cross-device streaming execution skew exceeds 50 ms')
        (output/'cycles.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
        result = dict(complete=True, predictions=len(records), execution_mode='stream',
                      execution=statuses, robot_commands_sent=True)
        if records:
            records[-1]['execution'] = statuses
            records[-1]['observation_only_status'] = observer_status(devices, selected)
            finished = max(s['execution']['finished_at']-devices[name].offset
                           for name, s in statuses.items())
            try:
                after = parallel(devices, endpoints)
                if not all(min(p['image_stamps'].values())-devices[name].offset > finished
                           for name, p in after.items()):
                    raise RuntimeError('No new camera frame after stream completion')
                records[-1]['observations_after'] = {
                    name: {k: v for k, v in p.items() if k != 'images'} for name, p in after.items()}
                save_observations(output/'latest', after,
                    {'cycle': cycles-1, 'phase': 'after_execution', **masking_metadata(config)})
            except (RuntimeError, OSError) as exc:
                result['post_observation_error'] = str(exc)
            atomic_json(output/'latest/actions.json', records[-1])
        atomic_json(output/'stream_summary.json', result)
        return result
    except BaseException:
        stop_all(selected)
        raise
    finally:
        heartbeats.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/real_robot'))
    parser.add_argument('--cycles', type=int, default=1)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--observe', action='store_true', help='Return live camera previews without model inference')
    modes.add_argument('--execute', action='store_true', help='Execute bounded chunks on the selected gateways')
    parser.add_argument('--execution-target', choices=('both', 'unitree'), default='both',
                        help='unitree executes Go2 and Piper only; P450 supplies diagnostic observations')
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('--cycles must be positive')
    config = json.loads(args.config.read_text())
    token = args.token_file.read_text().strip()
    devices = {name: Device(config[name], token) for name in device_names(config)}
    targets = ('unitree',) if args.execution_target == 'unitree' else None
    selected = execution_devices(devices, targets)
    output = args.output.expanduser()/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    # The static page only displays images and JSON; it has no actuator controls.
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="1">'
        '<title>HeteroVLA live observations</title><h1>Live camera observations</h1>'
        +''.join('<figure style="display:inline-block"><figcaption>'+n+'</figcaption><img width="400" src="latest/'+n+'.jpg"></figure>'
                 for n in SLOTS if n != 'uav' or 'p450' in devices)
        +'<p><a href="latest/observation.json">Observation</a> · <a href="latest/actions.json">Actions</a></p>')
    policy = None
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.observe:
            for cycle in range(args.cycles):
                packets = parallel(devices, '/preview')
                save_observations(output/'latest', packets, {'cycle': cycle, 'phase': 'preview'})
                print(json.dumps(dict(cycle=cycle, preview_saved=str(output/'latest'))), flush=True)
                time.sleep(.1)
            return
        observer_status(devices, selected)
        for name, device in devices.items():
            status = device.calibrate()
            if args.execute and name in selected and not status['execution_enabled']:
                raise RuntimeError('Execution gateway must be started with --execute: '+name)
        parallel(devices, '/reset', {name: {} for name in devices})
        policy = Policy(config.get('policy', 'ws://127.0.0.1:8000'))
        # Discard the first inference because cold JAX compilation can take tens of seconds.
        warm = run_cycle(devices, policy, config, output, execute=False, cycle=-1, targets=targets)
        print(json.dumps(dict(warmup_s=warm['inference_s'], executed=False)), flush=True)
        if args.execute and config.get('execution_mode', 'stream') == 'stream':
            run_action_stream(devices, policy, config, output, args.cycles, targets)
            return
        for cycle in range(args.cycles):
            if any(time.monotonic()-d.clock_checked > 10 for d in devices.values()):
                for device in devices.values():
                    device.calibrate()
            record = run_cycle(devices, policy, config, output, args.execute, cycle, targets=targets)
            with (output/'cycles.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            print(json.dumps({k: record[k] for k in ('cycle', 'inference_s', 'executed')}), flush=True)
    finally:
        if args.execute:
            atomic_json(output/'stop.json', stop_all(selected))
        if policy is not None:
            policy.close()
        print('Run output:', output, flush=True)


if __name__ == '__main__':
    main()
