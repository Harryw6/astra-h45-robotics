"""Hardware-free RTC prefix alignment for cumulative H45 goals."""
import math
import numpy as np
from .core import DT

CAPABILITY = 'h45_rtc_endpoint_vjp_v1'
ARRIVAL_CAPABILITY = 'h45_rtc_arrival_switch_v1'


class ArrivalPrefixEstimator:
    """Estimate generation-time guidance, never impose an execution deadline.

    Samples include observation transport, encoding, inference and submission.
    The caller includes one controller period for the next dispatch tick.
    Warmup is deliberately excluded: it happens before the robot starts.
    """
    def __init__(self, initial_s):
        if not math.isfinite(initial_s) or not 0 < initial_s < 44*DT:
            raise ValueError('Invalid initial RTC prefix estimate')
        self.estimate_s = float(initial_s)
        self.samples = 0

    @property
    def steps(self):
        return max(1, min(43, int(math.ceil(self.estimate_s/DT-1e-9))))

    def observe(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured RTC latency')
        self.estimate_s = (.5*self.estimate_s+.5*seconds) if self.samples else float(seconds)
        self.samples += 1


def empty_context():
    return dict(previous_actions=np.zeros((45, 14), np.float32), delay_steps=0,
                overlap_steps=0, channels=[True]*10+[False]*4)


def choose_delay(old_origin, new_origin, requested_steps, latency_history):
    """Shorten a conservative budget after an observation gap, never below P95."""
    remaining = int(math.floor((45*DT-(new_origin-old_origin))/DT+1e-7))
    available = min(requested_steps, remaining-1)
    required = int(math.ceil((np.percentile(latency_history[-20:], 95)+.10)/DT))
    if available < required:
        raise RuntimeError('RTC old-plan coverage cannot cover measured inference latency')
    return available


def prefix_context(previous, old_origin, new_origin, old_anchor, new_anchor,
                   old_state, new_state, delay_steps, soft_steps=4):
    aligned, remaining = align_plan(previous, old_origin, new_origin, old_anchor, new_anchor, old_state, new_state)
    if not 0 < delay_steps < remaining:
        raise ValueError('RTC delay leaves no usable old-plan overlap')
    aligned[remaining:] = 0.
    return dict(previous_actions=aligned.astype(np.float32), delay_steps=int(delay_steps),
                overlap_steps=min(remaining, int(delay_steps)+soft_steps),
                channels=[True]*10+[False]*4)


def align_plan(previous, old_origin, new_origin, old_anchor, new_anchor, old_state, new_state):
    previous = np.asarray(previous, float)
    old_anchor, new_anchor = np.asarray(old_anchor, float), np.asarray(new_anchor, float)
    if (previous.shape != (45, 14) or old_anchor.shape != (4,) or new_anchor.shape != (4,)
            or not all(np.isfinite(x).all() for x in (previous, old_anchor, new_anchor, old_state, new_state))
            or new_origin < old_origin):
        raise ValueError('Invalid RTC previous plan or causal pose anchors')
    age = new_origin-old_origin
    remaining = min(45, int(math.floor((45*DT-age)/DT+1e-7)))
    times = np.arange(46)*DT
    queries = age + np.arange(1, 46)*DT
    start = np.zeros(14)
    start[9] = old_state[9]
    rows = np.vstack((start, previous))
    aligned = np.stack([np.interp(queries, times, rows[:, j]) for j in range(14)], axis=1)
    # Unwrap yaw before interpolation, including the zero-relative initial pose.
    aligned[:, 2] = np.interp(queries, times, np.unwrap(rows[:, 2]))
    c, s = np.cos(old_anchor[3]), np.sin(old_anchor[3])
    world_x = old_anchor[0] + c*aligned[:, 0] - s*aligned[:, 1]
    world_y = old_anchor[1] + s*aligned[:, 0] + c*aligned[:, 1]
    dx, dy = world_x-new_anchor[0], world_y-new_anchor[1]
    c, s = np.cos(new_anchor[3]), np.sin(new_anchor[3])
    aligned[:, 0], aligned[:, 1] = c*dx+s*dy, -s*dx+c*dy
    yaw = old_anchor[3] + aligned[:, 2]-new_anchor[3]
    aligned[:, 2] = np.arctan2(np.sin(yaw), np.cos(yaw))
    aligned[:, 3:9] += np.asarray(old_state)[3:9]-np.asarray(new_state)[3:9]
    aligned[:, 10:] = 0.  # No UAV prefix is constrained in this Unitree loopback.
    return aligned, remaining


class PrefixCoverageError(ValueError):
    pass


def pipeline_prefix_context(plans, new_origin, new_anchor, new_state, delay_steps, soft_steps=4,
                            allow_short_prefix=False):
    """Condition on the trajectory actually scheduled, including future switches."""
    previous = np.zeros((45, 14), np.float32)
    covered = np.zeros(45, bool)
    starts = new_origin+np.arange(45)*DT
    for plan in sorted(plans, key=lambda p: p['switch_at']):
        aligned, _ = align_plan(plan['actions'], plan['origin_at'], new_origin,
                                plan['anchor'], new_anchor, plan['state'], new_state)
        selected = starts >= plan['switch_at']-1e-7
        valid = starts+DT <= plan['origin_at']+45*DT+1e-7
        previous[selected] = aligned[selected]
        covered[selected] = valid[selected]
    missing = np.flatnonzero(~covered)
    remaining = int(missing[0]) if len(missing) else 45
    if allow_short_prefix:
        # Guidance can shrink to the available overlap. Lack of a conservative
        # latency budget must not suppress an otherwise useful inference.
        if remaining <= 0:
            raise PrefixCoverageError('No unexpired scheduled action remains')
        delay_steps = min(max(0, int(delay_steps)), remaining-1)
    elif not 0 < delay_steps < remaining:
        raise PrefixCoverageError('Scheduled actions cannot cover the requested RTC prefix')
    previous[remaining:] = 0.
    return dict(previous_actions=previous, delay_steps=int(delay_steps),
                overlap_steps=min(remaining, delay_steps+soft_steps),
                channels=[True]*10+[False]*4)
