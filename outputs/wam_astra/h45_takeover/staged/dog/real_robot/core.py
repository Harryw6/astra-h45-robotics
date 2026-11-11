"""V3 units, causal state construction, freshness and bounded action plans."""
import base64
import math
import threading
import time
from collections import deque

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None  # Camera entry points import their real dependency when used.

# Small camera frames do not benefit from a machine-wide OpenCV worker pool.
# Hundreds of image workers can starve the gateway's deadline/watchdog threads.
if cv2 is not None:
    cv2.setNumThreads(1)

DT = 1.0 / 15
SLOTS = {'front': 'base_0_rgb', 'wrist': 'left_wrist_0_rgb', 'uav': 'right_wrist_0_rgb'}
JOINT_LIMITS = np.array([[-2.6179, 2.6179], [0., 3.14], [-2.967, 0.],
                         [-1.745, 1.745], [-1.22, 1.22], [-2.09439, 2.09439]])
COMMAND_LIMITS = {'unitree': np.array([1., 1., 3.5]),
                  'p450': np.array([.25, .25, .15, .4])}


def command_limits(kind, values=None):
    if kind == 'go2':
        kind = 'unitree'
    if kind == 'unitree' and values is False:
        # Explicit H45 setting: retain finite command validation, with no
        # application-level Go2 velocity envelope or output rescaling.
        return np.full(3, np.inf)
    limits = finite(COMMAND_LIMITS[kind] if values is None else values,
                    (3 if kind == 'unitree' else 4,))
    if np.any(limits <= 0):
        raise ValueError('Command rejection limits must be positive')
    return limits


def finite(value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Expected finite array %s; got %s' % (shape, value.shape))
    return value


def heading(xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = xy
    return np.array([c*x+s*y, -s*x+c*y])


def world_velocity(body, yaw):
    return heading(body, -yaw)


def angle_delta(a, b):
    return math.atan2(math.sin(b-a), math.cos(b-a))


class History:
    def __init__(self, capacity=300):
        self.rows = deque(maxlen=capacity)

    def add(self, stamp, value):
        if not math.isfinite(stamp):
            raise ValueError('Nonfinite sensor timestamp')
        if self.rows and stamp <= self.rows[-1][0]:
            return
        self.rows.append((stamp, value))

    def past(self, stamp, max_age):
        for t, value in reversed(self.rows):
            if t <= stamp:
                if stamp - t > max_age:
                    break
                return t, value
        raise RuntimeError('Missing/stale causal sensor sample')

    def motion(self, stamp, yaw_index, include_z=False):
        ta, a = self.past(stamp-DT, .075)
        tb, b = self.past(stamp, .075)
        times = [t for t, _ in self.rows if ta <= t <= tb]
        if tb <= ta or any(y-x > .12 for x, y in zip(times, times[1:])):
            raise RuntimeError('Pose history has a gap')
        xy = heading((b[:2]-a[:2])/(tb-ta), b[yaw_index])
        rate = angle_delta(a[yaw_index], b[yaw_index])/(tb-ta)
        return [*xy, (b[2]-a[2])/(tb-ta), rate] if include_z else [*xy, rate]


class Sensors:
    """All timestamps are the device's monotonic seconds; callbacks own no actuators."""
    def __init__(self, kind):
        self.kind = kind
        self.lock = threading.RLock()
        names = ['go2', 'piper', 'front', 'wrist'] if kind == 'unitree' else ['pose', 'flags', 'landed', 'uav']
        self.streams = {name: History(12 if name in SLOTS else 300) for name in names}
        self.origin = None
        self.error = None
        self.input_errors = {}

    def put(self, name, stamp, value):
        with self.lock:
            self.streams[name].add(stamp, value)

    def reset(self):
        with self.lock:
            name = 'go2' if self.kind == 'unitree' else 'pose'
            _, pose = self.streams[name].past(time.monotonic(), .075)
            self.origin = np.array(pose, copy=True)

    def preview(self):
        """Read camera images while landed/disabled; never fabricate model state."""
        import cv2
        names = ('front', 'wrist') if self.kind == 'unitree' else ('uav',)
        images, times = {}, {}
        with self.lock:
            now = time.monotonic()
            for name in names:
                stamp, bgr = self.streams[name].past(now, .15)
                ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError('Preview encoding failed')
                images[name] = base64.b64encode(data).decode('ascii')
                times[name] = stamp
            health = {name: dict(samples=len(history.rows),
                      age_s=None if not history.rows else now-history.rows[-1][0])
                      for name, history in self.streams.items()}
            raw = {name: dict(stamp=history.rows[-1][0],
                             value=np.asarray(history.rows[-1][1]).tolist())
                   for name, history in self.streams.items() if name not in SLOTS and history.rows}
        return dict(kind=self.kind, stamp=min(times.values()), images=images,
                    image_stamps=times, health=health, raw_sensors=raw, sensor_error=self.error)

    def execution_snapshot(self):
        return self.snapshot(execution_only=True)

    def snapshot(self, diagnostic=False, execution_only=False):
        import cv2
        with self.lock:
            issues = [self.error] if self.error else []
            if self.error and not diagnostic:
                raise RuntimeError(self.error)
            if self.origin is None:
                raise RuntimeError('Call /reset to establish the episode origin')
            names = ('front', 'wrist') if self.kind == 'unitree' else ('uav',)
            if execution_only:
                names = ()
            main = ('go2', 'piper') if self.kind == 'unitree' else ('pose',)
            missing = [n for n in (*names, *main) if not self.streams[n].rows]
            if missing:
                raise RuntimeError('Waiting for live sensors: '+', '.join(missing))
            stamp = min(self.streams[n].rows[-1][0] for n in (*names, *main))
            now = time.monotonic()
            if not 0 <= now-stamp <= .15:
                raise RuntimeError('Observation timeline is stale')
            images, times = {}, {}
            for name in names:
                t, bgr = self.streams[name].past(stamp, .15)
                images[name] = bgr.copy()
                times[name] = t
            if times and max(times.values())-min(times.values()) > .1:
                raise RuntimeError('Camera skew exceeds 100 ms')
            if self.kind == 'unitree':
                tg, g = self.streams['go2'].past(stamp, .075)
                tp, p = self.streams['piper'].past(stamp, .075)
                g, p = finite(g, (7,)), finite(p, (7,))
                if g[4] < .23:
                    issues.append('Go2 is not standing')
                    if not diagnostic:
                        raise RuntimeError('Go2 must already be standing; policy has no stand-up action')
                xy = heading(g[:2]-self.origin[:2], self.origin[3])
                dyaw = g[3]-self.origin[3]
                state = [*xy, g[4], g[5], g[6], math.sin(dyaw), math.cos(dyaw),
                         *self.streams['go2'].motion(stamp, 3), *p]
                state_times = [tg, tp]
            else:
                tu, u = self.streams['pose'].past(stamp, .075)
                tf, flags = self.streams['flags'].past(stamp, 1.5 if diagnostic else .15)
                tl, landed = self.streams['landed'].past(stamp, 1.5 if diagnostic else .25)
                u, flags = finite(u, (6,)), finite(flags, (3,))
                if not np.all(flags == 1):
                    issues.append('P450 connected/armed/odom_valid='+str(flags.tolist()))
                    if not diagnostic:
                        raise RuntimeError('P450 connected/armed/odom_valid must all be true (V3 training contract)')
                if stamp-tf > .15 or stamp-tl > .25:
                    issues.append('Status message age exceeds execution limits')
                xy = heading(u[:2]-self.origin[:2], self.origin[5])
                dyaw = u[5]-self.origin[5]
                state = [*xy, u[2], u[3], u[4], math.sin(dyaw), math.cos(dyaw),
                         *self.streams['pose'].motion(stamp, 5, True), flags[1], float(landed)]
                state_times = [tu, tf, tl]
        encoded = {}
        for name, bgr in images.items():
            ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            encoded[name] = base64.b64encode(data).decode('ascii')
        finite(state, (17 if self.kind == 'unitree' else 13,))
        return dict(kind=self.kind, stamp=stamp, image_stamps=times,
                    state_stamps=state_times, state=state, images=encoded,
                    diagnostic_only=diagnostic, readiness_issues=issues)


def device_names(config):
    mode = config.get('observation_mode', 'all')
    if mode == 'all':
        return ('unitree', 'p450')
    if mode != 'unitree_only':
        raise ValueError('Unknown observation_mode: '+str(mode))
    finite(config.get('p450_state_fill'), (13,))
    return ('unitree',)


def masking_metadata(config):
    if device_names(config) != ('unitree',):
        return {}
    return dict(observation_mode='unitree_only', missing_devices=['p450'],
                masked_image_slots=[SLOTS['uav']], state_fill_indices=list(range(17, 30)),
                state_fill=config['p450_state_fill'], state_fill_is_measured=False,
                state_dimension_mask_supported=False, ignored_action_indices=list(range(10, 14)),
                normalization_sha256=config.get('normalization_sha256'))


def model_observation(unitree_state, p450_state, images, prompt, p450_state_fill=None):
    """Keep the checkpoint shape; an absent UAV is explicit, never a fake sensor packet."""
    images = dict(images)
    expected = {SLOTS['front'], SLOTS['wrist']}
    if p450_state is not None:
        expected.add(SLOTS['uav'])
    if set(images) != expected:
        raise ValueError('Missing or unexpected live camera slots')
    masks = {key: np.bool_(True) for key in images}
    if p450_state is None:
        p450_state = finite(p450_state_fill, (13,))
        images[SLOTS['uav']] = np.zeros((480, 640, 3), dtype=np.uint8)
        masks[SLOTS['uav']] = np.bool_(False)
    state = np.concatenate([finite(unitree_state, (17,)), finite(p450_state, (13,))]).astype(np.float32)
    return dict(state=state, image=images, image_mask=masks, prompt=prompt)


def decode_observation(unitree, p450, clock_maps, prompt, now=None, *, p450_state_fill=None):
    """Map each device clock to the controller clock with bounded RTT uncertainty."""
    import cv2
    now = time.monotonic() if now is None else now
    images, times, uncertainties = {}, [], []
    packets = [('unitree', unitree)] if p450 is None else [('unitree', unitree), ('p450', p450)]
    for name, packet in packets:
        if packet['kind'] != name:
            raise ValueError('Wrong gateway kind')
        offset, uncertainty = clock_maps[name]
        if not np.isfinite([offset, uncertainty, packet['stamp']]).all() or not 0 <= uncertainty <= .04:
            raise RuntimeError('Clock uncertainty exceeds 40 ms')
        age = now-(packet['stamp']-offset)
        if age < -uncertainty or age+uncertainty > .25:
            raise RuntimeError('Observation expired in transport')
        for camera, stamp in packet['image_stamps'].items():
            if not math.isfinite(stamp) or stamp > packet['stamp']:
                raise ValueError('Invalid/noncausal camera timestamp')
            times.append(stamp-offset)
            uncertainties.append(uncertainty)
            data = np.frombuffer(base64.b64decode(packet['images'][camera], validate=True), np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (480, 640, 3):
                raise ValueError('Expected a 640x480 live RGB camera frame')
            images[SLOTS[camera]] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(t+u for t, u in zip(times, uncertainties))-min(t-u for t, u in zip(times, uncertainties)) > .1:
        raise RuntimeError('Cross-device camera skew/clock uncertainty exceeds 100 ms')
    return model_observation(unitree['state'], None if p450 is None else p450['state'],
                             images, prompt, p450_state_fill)


def action_plan(actions, kind, current_arm=None, velocity_scale=1., velocity_limits=None,
                piper_rate_checks=True):
    actions = np.asarray(actions, dtype=float)
    width = {'unitree': 10, 'go2': 3, 'p450': 4}[kind]
    if actions.ndim != 2 or actions.shape[1] != width or not 1 <= len(actions) <= 15:
        raise ValueError('Invalid action chunk shape')
    if not np.isfinite(actions).all() or velocity_scale != 1.:
        raise ValueError('Raw action execution requires finite predictions and velocity_scale=1')
    rows = actions.copy()
    if kind == 'unitree':
        if np.any(rows[:, 3:9] < JOINT_LIMITS[:, 0]) or np.any(rows[:, 3:9] > JOINT_LIMITS[:, 1]):
            raise ValueError('Piper prediction outside physical joint limits')
        if np.any(rows[:, 9] < 0) or np.any(rows[:, 9] > .075):
            raise ValueError('Piper gripper prediction outside physical limits')
        if np.any(np.abs(rows[:, :3]) > command_limits(kind, velocity_limits)):
            raise ValueError('Go2 velocity exceeds execution limits')
        if piper_rate_checks:
            previous = finite(current_arm, (7,)).copy()
            if np.max(np.abs(rows[0, 3:9]-previous[:6])) > .35:
                raise ValueError('First Piper target differs from feedback by over 0.35 rad')
            previous = rows[0, 3:].copy()
            for row in rows[1:]:
                if np.any(np.abs(row[3:]-previous) > np.array([.35]*6+[.02])*DT+1e-12):
                    raise ValueError('Piper target exceeds joint/gripper rate limits; raw target was not clipped')
                previous = row[3:].copy()
    else:
        if np.any(np.abs(rows) > command_limits(kind, velocity_limits)):
            raise ValueError('%s velocity exceeds execution limits' % kind)
    return rows


def piper_sdk_target(arm):
    arm = finite(arm, (7,))
    return [int(round(math.degrees(q)*1000)) for q in arm[:6]], int(round(arm[6]*1_000_000))
