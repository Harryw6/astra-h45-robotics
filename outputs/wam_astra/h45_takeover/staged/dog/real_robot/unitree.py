"""Unitree SDK2/Piper SDK/V4L2 adapter. Import has no hardware side effects."""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from .core import Sensors, piper_sdk_target, command_limits
from .review_pause import settings as review_settings, check_arm, _stamp, base_sample


def go2_normal_state_codes(config):
    """Explicit per-device compatibility; unspecified devices require code zero."""
    codes = config.get('go2_normal_state_codes', [0])
    if (not isinstance(codes, list) or not codes or
            any(type(code) is not int or not 0 <= code <= 0xffffffff for code in codes)):
        raise ValueError('go2_normal_state_codes must be a nonempty list of uint32 values')
    return frozenset(codes)


class UnitreeHardware:
    def __init__(self, config, execute=False):
        import fcntl  # Linux-only hardware dependency; local source review can import elsewhere.
        self.config, self.execute = config, execute
        self.piper_move_speed_percent = config.get('piper_move_speed_percent', 100)
        if (type(self.piper_move_speed_percent) is not int
                or not 0 <= self.piper_move_speed_percent <= 100):
            raise ValueError('piper_move_speed_percent must be a supported SDK percentage (0..100)')
        self.go2_normal_codes = go2_normal_state_codes(config)
        self.check_go2_error_code = config.get('go2_check_error_code', True)
        if type(self.check_go2_error_code) is not bool:
            raise ValueError('go2_check_error_code must be a boolean')
        self.sensors = Sensors('unitree')
        self.closed = threading.Event()
        self.io_lock = threading.RLock()
        self.processes, self.logs, self.cameras = [], [], {}
        self.piper = self.guard = None
        self.reader = None
        self.fault = None
        self.sensor_diagnostics = dict(idle_gap_events=0, last_error=None)
        self._previous_sensor_error = None
        self.run = Path(tempfile.mkdtemp(prefix='heterovla-v3-'))
        self.motion_socket = self.run/'motion.sock'
        self.lock_file = open('/tmp/heterovla-v3-unitree.lock', 'a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if execute:
                conflicts = ('piper_chunk_loop.py', 'piper_stream_loop.py', 'hetero_teleop_loop.py',
                             'hetero_teach_loop.py', 'piper_gamepad_teleop.py', 'go2_velocity_bridge',
                             'go2_replay_bridge', 'replay_episode.py')
                for process in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        command = process.read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    except OSError:
                        continue
                    if any(name in command for name in conflicts):
                        raise RuntimeError('Another controller is running: PID '+process.parent.name)
            sys.path.insert(0, str(Path(__file__).parent/'vendor'))
            from hetero_pkl_recorder import CameraReader, feedback_piper
            from piper_safety import monitored_piper, SafetyGuard
            from piper_sdk import C_PiperInterface_V2
            self.piper = feedback_piper(monitored_piper(C_PiperInterface_V2))(
                config.get('can', 'can0'), start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
            self.piper.ConnectPort(piper_init=False)
            if execute:
                self.guard = SafetyGuard(self.piper)
                self.guard.acknowledge()
                self.guard.start()
            for name, serial in (config['cameras'].items() if config.get('capture_live_cameras', True) else ()):
                camera = CameraReader(name, serial, 640, 480, 30)
                self.cameras[name] = camera
                camera.start()
            self._spawn('v3_go2_state', [config.get('interface', 'eth0'), str(self.run)])
            if execute:
                limits = command_limits('unitree', config.get('velocity_limits'))
                self._spawn('v3_go2_motion', [config.get('interface', 'eth0'), str(self.motion_socket),
                                             *map(str, limits), '200', '3'])
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        except Exception:
            self.close()
            raise

    def _spawn(self, binary, args):
        folder = Path(self.config.get('build_dir', str(Path(__file__).parent/'build'))).expanduser()
        log = (self.run/(binary+'.log')).open('ab', buffering=0)
        self.logs.append(log)
        self.processes.append(subprocess.Popen([str(folder/binary), *args], stdout=log, stderr=log))

    def _read(self):
        while not self.closed.wait(.01):
            self._read_once()
            self._update_sensor_fault()

    def _update_sensor_fault(self):
        error = self.sensors.error
        active = self.execute and bool(getattr(getattr(self, 'guard', None), 'active', False))
        if error and error != self._previous_sensor_error:
            if not active:
                self.sensor_diagnostics['idle_gap_events'] += 1
            self.sensor_diagnostics['last_error'] = dict(monotonic=time.monotonic(),
                                                       error=error, executing=active)
        self._previous_sensor_error = error
        # Standby has no outstanding motion. Its transient missing feedback
        # blocks readiness until fresh data returns, without a permanent latch.
        # An active control tick and SafetyGuard retain their original checks.
        if active and error and not self.fault:
            self.fault = 'Execution inhibited after sensor error: ' + error
            try:
                self.halt()
            except Exception as exc:
                self.fault += '; stop request failed: ' + str(exc)

    def _read_once(self):
        # A disabled robot or missing arm feedback must not hide live cameras.
        # Execution still rejects any error; preview reports each stream as received.
        errors = []
        try:
            if any(p.poll() is not None for p in self.processes):
                raise RuntimeError('Go2 bridge exited; inspect '+str(self.run))
            data = json.loads((self.run/'go2_snapshot.json').read_text())['sport_mode_state']
            self.go2_feedback_code = data['error_code']
            if not data['valid']:
                raise RuntimeError('Missing Go2 DDS feedback')
            g = np.array([*data['position'], data['rpy'][2], data['body_height'], *data['rpy'][:2]])
            self.sensors.put('go2', data['monotonic_ns']/1e9, g)
            if getattr(self, 'check_go2_error_code', True) and data['error_code'] not in self.go2_normal_codes:
                raise RuntimeError('Go2 DDS error_code='+str(data['error_code']))
        except Exception as exc:
            errors.append('go2: '+str(exc))
        try:
            arm, stamp = self.arm_feedback()
            self.sensors.put('piper', stamp, arm)
        except Exception as exc:
            errors.append('piper: '+str(exc))
        input_errors = {}
        for name, camera in self.cameras.items():
            try:
                if camera.error:
                    raise RuntimeError(camera.error)
                frame = camera.snapshot()
                if frame is not None:
                    self.sensors.put(name, frame['monotonic_ns']/1e9, frame['image'])
            except Exception as exc:
                input_errors[name] = str(exc)
        self.sensors.input_errors = input_errors
        self.sensors.error = '; '.join(errors) or None

    def arm_feedback(self):
        joints, gripper, _, stamp, _ = self.piper.feedback_snapshot(timeout_ns=75_000_000)
        arm = [math.radians(float(getattr(joints.joint_state, 'joint_'+str(i)))/1000.) for i in range(1, 7)]
        arm.append(float(gripper.gripper_state.grippers_angle)/1_000_000.)
        return np.array(arm), stamp[0]/1e9

    def _go2(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(.08)
            client.connect(str(self.motion_socket))
            client.sendall((command+'\n').encode())
            result = client.makefile('r').readline(512).split()
        if len(result) < 2 or result[:2] != ['OK', '0']:
            raise RuntimeError('Go2 command failed: '+str(result))
        if len(result) != 12 or result[2] not in ('0', '1'):
            raise RuntimeError('Go2 response lacks measured state: '+str(result))
        values = [float(value) for value in result[4:]]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('Nonfinite Go2 response')
        return dict(valid=result[2] == '1', monotonic_ns=int(result[3]),
                    position=values[:3], yaw=values[3], velocity=values[4:6],
                    yaw_speed=values[6], body_height=values[7])

    def check_execution(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.sensors.error:
            raise RuntimeError(self.sensors.error)
        if not self.execute or self.guard is None or self.guard.inhibited:
            raise RuntimeError('Piper execution unavailable or safety fault latched')
        from .piper_safety import FEEDBACK_IDS, ENABLE_IDS
        if not self.piper.feedback_fresh(FEEDBACK_IDS+ENABLE_IDS, .25):
            raise RuntimeError('Missing Piper feedback/enable status')
        status = self.piper.GetArmStatus().arm_status
        if (status.arm_status != 0 or status.ctrl_mode != 1 or status.err_code != 0
                or not all(self.piper.GetArmEnableStatus())):
            raise RuntimeError('Piper must already be enabled, normal, and in CAN mode')
        with self.sensors.lock:
            _, go = self.sensors.streams['go2'].past(time.monotonic(), .075)
            if go[4] < .23:
                raise RuntimeError('Go2 is not standing')
        if not self.motion_socket.exists():
            raise RuntimeError('Go2 SportClient lease/socket not ready')
        self.arm_feedback()

    def _review_targets(self, arm, config):
        arm = check_arm(arm, config)
        joints, _ = piper_sdk_target(arm)
        # The installed SDK silently clips. Reject before calling JointCtrl if
        # its actual configured integer bounds would alter a measured target.
        for index, target in enumerate(joints):
            bounds = self.piper.GetSDKJointLimitParam('j'+str(index+1))
            low, high = [round(math.degrees(float(q))*1000) for q in bounds]
            if not all(math.isfinite(float(q)) for q in bounds) or not low <= target <= high:
                raise RuntimeError('Measured target would be clipped by SDK joint limits')
        return joints

    def review_feedback(self, settings):
        """Read measured state and keep the existing guard continuously active."""
        config = review_settings(settings)
        from .piper_safety import FEEDBACK_IDS, ENABLE_IDS
        with self.io_lock, self.guard.lock:
            if self.fault or not self.execute or self.guard.inhibited:
                raise RuntimeError(self.fault or 'Review execution unavailable or safety fault latched')
            if config['monitor_period_s'] >= self.guard.loop_timeout:
                raise RuntimeError('Review monitor exceeds active watchdog deadline')
            if self.guard.errors.generation != self.guard.seen_error:
                raise RuntimeError('Unacknowledged SDK/CAN error during review')
            if not self.piper.feedback_fresh(FEEDBACK_IDS+ENABLE_IDS, config['max_feedback_age_s']):
                raise RuntimeError('Missing fresh Piper feedback and enable state')
            status = self.piper.GetArmStatus().arm_status
            enabled = self.piper.GetArmEnableStatus()
            if (status.arm_status != 0 or status.ctrl_mode != 1 or status.err_code != 0
                    or len(enabled) != 6 or not all(enabled)):
                raise RuntimeError('Review requires already enabled, normal CAN mode and zero err_code')
            arm, stamp = self.arm_feedback()
            self._review_targets(arm, config)
            base = self._go2('STATE')
            snapshot = json.loads((self.run/'go2_snapshot.json').read_text())['sport_mode_state']
            if snapshot.get('error_code') not in config['go2_normal_state_codes']:
                raise RuntimeError('Unverified Go2 DDS error_code')
            snapshot = dict(snapshot, yaw=snapshot['rpy'][2])
            now = time.monotonic()
            _stamp(stamp, now, config)
            for measured in (base, snapshot):
                base_sample(measured, now, config)
                if not math.isfinite(measured['body_height']) or measured['body_height'] < config['go2_min_body_height_m']:
                    raise RuntimeError('Go2 below configured standing height')
            self.guard.active = True
            self.guard.last_tick = now
            return dict(base=base, go2_snapshot=snapshot, arm=arm.tolist(),
                        arm_stamp=stamp, readiness_verified=True)

    def review_hold(self, settings):
        """STOP then measured joint hold. Never enable or write the gripper."""
        config = review_settings(settings)
        with self.io_lock:
            stop_response = self._go2('STOP')
            frame = self.review_feedback(config)
            joints = self._review_targets(frame['arm'], config)
            with self.guard.lock:
                if self.guard.inhibited:
                    raise RuntimeError('Safety fault latched during review entry')
                self.guard.active = True
                self.guard.last_tick = time.monotonic()
                self.guard.call('JointCtrl', *joints)
            return dict(stop_response=stop_response, initial_feedback=frame,
                        gripper_command_sent=False, physical_hold_proven=False)

    def send(self, row):
        with self.io_lock:
            self.check_execution()
            with self.guard.lock:
                if self.guard.inhibited:
                    raise RuntimeError(self.guard.fault)
                self.guard.active = True
                self.guard.last_tick = time.monotonic()
                joints, grip = piper_sdk_target(row[3:])
                self.guard.call('ModeCtrl', 0x01, 0x01, self.piper_move_speed_percent, 0)
                self.guard.call('JointCtrl', *joints)
                self.guard.call('GripperCtrl', grip, 1000, 0x01, 0)
            self._go2('MOVE %.17g %.17g %.17g' % tuple(row[:3]))

    def hold(self):
        with self.io_lock:
            self._go2('STOP')
            arm, _ = self.arm_feedback()
            joints, grip = piper_sdk_target(arm)
            with self.guard.lock:
                if self.guard.inhibited:
                    raise RuntimeError(self.guard.fault)
                self.guard.call('JointCtrl', *joints)
                self.guard.call('GripperCtrl', grip, 1000, 0x01, 0)
                self.guard.active = False

    def halt(self):
        if self.guard is not None:
            self.guard.trip('V3 execution stopped; inspect and restart gateway before resuming')
        if self.motion_socket.exists():
            self._go2('STOP')

    def close(self):
        self.closed.set()
        if self.guard is not None:
            self.guard.close()
        if self.motion_socket.exists():
            try:
                self._go2('QUIT')
            except Exception:
                pass
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.reader is not None and self.reader is not threading.current_thread():
            self.reader.join(timeout=1)
        for camera in self.cameras.values():
            camera.stop()
        if self.piper is not None:
            self.piper.DisconnectPort()
        for log in self.logs:
            log.close()
        self.lock_file.close()
