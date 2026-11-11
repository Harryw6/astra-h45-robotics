"""Go2-only sequence validation; no Piper interfaces or control-mode changes."""
import numpy as np

from .core import finite


VELOCITY_LIMITS = np.array([.25, .25, .5])


def continue_prediction_chunks(predictions, valid_prediction):
    """Use the latest H15 chunk's time-aligned future step until a new chunk arrives."""
    predictions = np.asarray(predictions, dtype=np.float64)
    valid = np.asarray(valid_prediction, dtype=bool)
    if (predictions.ndim != 3 or predictions.shape[1:] != (15, 3)
            or not len(predictions) or valid.shape != (len(predictions),)):
        raise ValueError('Expected Nx15x3 predictions and N validity flags')
    if not np.isfinite(predictions[valid]).all():
        raise ValueError('Nonfinite valid Go2 prediction chunk')
    velocities = np.zeros((len(predictions), 3))
    sources = np.full(len(predictions), -1, dtype=np.int64)
    steps = np.full(len(predictions), -1, dtype=np.int64)
    last = -1
    for frame in range(len(predictions)):
        if valid[frame]:
            last = frame
        if last >= 0:
            step = frame-last
            if step >= 15:
                raise ValueError('Prediction horizon exhausted during missing model input')
            velocities[frame] = predictions[last, step]
            sources[frame], steps[frame] = last, step
    return velocities, sources, steps


def hold_last_velocity(predictions, valid_prediction):
    """Hold the last valid vx/vy/yaw_rate on missing recorded inputs.

    Leading missing inputs remain stationary. Valid zero-speed predictions are
    applied normally. Returned indices identify the prediction used per tick;
    -1 means no valid prediction has arrived. This is offline replay planning,
    not permission to ignore stale live robot feedback or the command watchdog.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    valid = np.asarray(valid_prediction, dtype=bool)
    if (predictions.ndim != 2 or predictions.shape[1] != 3 or
            not len(predictions) or valid.shape != (len(predictions),)):
        raise ValueError('Expected Nx3 predictions and N validity flags')
    if not np.isfinite(predictions[valid]).all():
        raise ValueError('Nonfinite valid Go2 prediction')
    velocities = np.zeros_like(predictions)
    sources = np.full(len(predictions), -1, dtype=np.int64)
    last = -1
    for frame in range(len(predictions)):
        if valid[frame]:
            last = frame
        if last >= 0:
            velocities[frame] = predictions[last]
            sources[frame] = last
    return velocities, sources


def validate_velocities(values, limits=VELOCITY_LIMITS):
    """Validate samples without changing them; allow a checked run-specific envelope."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError('Expected a nonempty sequence of vx/vy/yaw_rate')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite Go2 velocity')
    limits = finite(limits, (3,))
    if np.any(limits <= 0):
        raise ValueError('Go2 velocity limits must be positive')
    if np.any(np.abs(values) > limits):
        raise ValueError('Go2 sequence exceeds hardware velocity limits')
    return values


def validate_feedback(packet, now, normal_codes=(0, 100), max_age=.075):
    state = packet['sport_mode_state']
    if state.get('valid') is not True:
        raise RuntimeError('Missing Go2 DDS feedback')
    stamp = float(state['monotonic_ns']) / 1e9
    if not np.isfinite([now, stamp]).all() or not 0 <= now-stamp <= max_age:
        raise RuntimeError('Go2 DDS feedback is stale or future dated')
    if state['error_code'] not in normal_codes:
        raise RuntimeError('Go2 DDS error_code='+str(state['error_code']))
    if not np.isfinite(state['body_height']) or state['body_height'] < .23:
        raise RuntimeError('Go2 is not standing')
    finite(state['position'], (3,))
    finite(state['rpy'], (3,))
    finite(state['velocity'], (3,))
    return state


def parse_response(line):
    fields = line.split()
    if len(fields) < 12 or fields[0] != 'OK':
        raise RuntimeError('Invalid Go2 bridge response')
    if int(fields[1]) != 0:
        raise RuntimeError('Go2 SDK command failed: '+fields[1])
    values = finite([float(x) for x in fields[3:12]], (9,))
    return dict(sdk_return=0, state_valid=bool(int(fields[2])),
                monotonic_ns=int(fields[3]), position=values[1:4].tolist(),
                yaw=float(values[4]), velocity=values[5:7].tolist(),
                yaw_rate=float(values[7]), body_height=float(values[8]))
