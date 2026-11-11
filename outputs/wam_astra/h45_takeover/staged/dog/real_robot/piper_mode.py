"""Explicit operator command for CAN mode; default operation is passive feedback only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def passive_status(channel='can0', duration=1.):
    import can
    ids = [0x2A1, *range(0x261, 0x267)]
    frames = {}
    bus = can.interface.Bus(channel=channel, interface='socketcan',
        can_filters=[dict(can_id=i, can_mask=0x7ff, extended=False) for i in ids],
        receive_own_messages=False)
    try:
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            frame = bus.recv(.05)
            if frame is not None and frame.arbitration_id in ids and len(frame.data) == 8:
                frames[frame.arbitration_id] = (bytes(frame.data), time.monotonic())
        now = time.monotonic()
        missing = [hex(i) for i in ids if i not in frames or now-frames[i][1] > .25]
        if missing:
            raise RuntimeError('Missing/stale Piper feedback: '+', '.join(missing))
        status = frames[0x2A1][0]
        return dict(ctrl_mode=status[0], arm_status=status[1], teach_status=status[3],
                    err_code=int.from_bytes(status[6:8], 'big'),
                    enabled=[bool(frames[i][0][5] & 0x40) for i in range(0x261, 0x267)],
                    passive=True)
    finally:
        bus.shutdown()


def snapshot(interface):
    from piper_safety import FEEDBACK_IDS, ENABLE_IDS
    if not interface.feedback_fresh(FEEDBACK_IDS+ENABLE_IDS, .25):
        raise RuntimeError('Missing/stale Piper feedback or enable flags')
    joints, _, status, _, _ = interface.feedback_snapshot(timeout_ns=75_000_000)
    arm = status.arm_status
    values = [int(getattr(joints.joint_state, 'joint_'+str(i))) for i in range(1, 7)]
    return dict(ctrl_mode=int(arm.ctrl_mode), arm_status=int(arm.arm_status),
                err_code=int(arm.err_code), enabled=[bool(x) for x in interface.GetArmEnableStatus()],
                joints_mdeg=values)


def require_normal_enabled(state):
    if state['arm_status'] != 0 or state['err_code'] != 0:
        raise RuntimeError('Piper is not normal; no recovery commands sent: '+json.dumps(state))
    if len(state['enabled']) != 6 or not all(state['enabled']):
        raise RuntimeError('Piper motors must already be enabled; no enable commands sent: '+json.dumps(state))
    if state['ctrl_mode'] not in (0, 1, 2):
        raise RuntimeError('Unsupported Piper control mode: '+json.dumps(state))


def switch_to_can(interface, send, timeout=2.):
    """Exit teaching if needed, select CAN/MOVE_J and hold the measured joint pose."""
    before = snapshot(interface)
    require_normal_enabled(before)
    if before['ctrl_mode'] == 1:
        return dict(before=before, after=before, changed=False)
    if before['ctrl_mode'] == 2:
        send('MotionCtrl_1', 0x00, 0x00, 0x02)
    # Refresh the measured target immediately before selecting joint control.
    target = snapshot(interface)
    require_normal_enabled(target)
    send('ModeCtrl', 0x01, 0x01, 100, 0x00)
    send('JointCtrl', *target['joints_mdeg'])
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        after = snapshot(interface)
        require_normal_enabled(after)
        if after['ctrl_mode'] == 1:
            return dict(before=before, after=after, changed=True,
                        hold_joint_targets_mdeg=target['joints_mdeg'])
        time.sleep(.02)
    raise RuntimeError('Piper did not confirm CAN control mode: '+json.dumps(after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can', default='can0')
    parser.add_argument('--apply', action='store_true', help='Operator action: exit teaching and hold current pose in CAN mode')
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps(passive_status(args.can)), flush=True)
        return
    from .service import identity
    record = Path.home()/'.config/heterovla-live/service_unitree.json'
    if record.is_file():
        service = json.loads(record.read_text())
        if identity(service['pid']) == service['identity']:
            raise RuntimeError('Stop the Unitree gateway before changing Piper mode')
    with open('/tmp/heterovla-v3-unitree.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conflicts = ('piper_gamepad_teleop.py', 'hetero_teleop_loop.py', 'piper_stream_loop.py',
                     'piper_chunk_loop.py', 'replay_episode.py', 'Gamepad_PiPER/main.py')
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            if int(path.parent.name) == os.getpid():
                continue
            try:
                command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
            except OSError:
                continue
            if any(name in command for name in conflicts):
                raise RuntimeError('Another Piper controller is running: PID '+path.parent.name)
        sys.path.insert(0, str(Path(__file__).parent/'vendor'))
        from hetero_pkl_recorder import feedback_piper
        from piper_safety import monitored_piper, SafetyGuard
        from piper_sdk import C_PiperInterface_V2
        interface = feedback_piper(monitored_piper(C_PiperInterface_V2))(
            args.can, start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
        interface.ConnectPort(piper_init=False)
        guard = None
        try:
            deadline = time.monotonic()+3.
            while True:
                try:
                    snapshot(interface)
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            guard = SafetyGuard(interface)
            guard.acknowledge()
            with guard.lock:
                attempted = []
                def send(method, *values):
                    attempted.append(method)
                    return guard.call(method, *values)
                try:
                    report = switch_to_can(interface, send)
                except Exception:
                    if attempted:
                        try:
                            guard.call('MotionCtrl_1', 0x01, 0, 0)
                        except Exception:
                            pass
                    raise
            print(json.dumps(report), flush=True)
        finally:
            if guard is not None:
                guard.close()
            interface.DisconnectPort()


if __name__ == '__main__':
    main()
