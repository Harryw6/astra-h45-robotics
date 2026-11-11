"""Timestamped H15 action buffers; model inference never clocks the actuator loop."""
import math
import time

from .core import DT


class ActionBuffer:
    def __init__(self, hold_last=False):
        # The caller must independently enforce controller/feedback liveness.
        self.hold_last = hold_last
        self.current = None
        self.pending = None
        self.history = []
        self.latest_origin = -math.inf
        self.drain = False
        self.finished = False
        self.wait_on_exhaustion = False
        self.waiting_since = None
        self.wait_epoch = 0
        self.pipeline = False
        self.arrival_handoff = False
        self.queued = []
        self.paused_s = 0.

    def latest_plan(self):
        return self.queued[-1] if self.queued else self.pending or self.current

    def pause_duration(self, now):
        return (self.paused_s + (now-self.waiting_since if self.waiting_since is not None else 0.)) if self.pipeline else 0.

    def schedule(self, plan, replace_future=False):
        if plan['origin_at'] <= self.latest_origin:
            raise ValueError('Prediction is older than the latest action buffer')
        if self.drain:
            raise RuntimeError('Action stream is already draining')
        if replace_future:
            # A fresher arrival supersedes every not-yet-running plan. Keep the
            # current plan until sample() atomically switches on a control tick.
            for old in ([self.pending] if self.pending else [])+self.queued:
                old['status'] = 'superseded'
            self.pending = None
            self.queued = []
        if self.pipeline and self.pending is not None:
            if plan['scheduled_start'] <= self.latest_plan()['scheduled_start']:
                raise ValueError('RTC pipeline switch times must increase')
            self.queued.append(plan)
        else:
            if self.pending is not None:
                self.pending['status'] = 'superseded'
            self.pending = plan
        self.history.append(plan)
        self.latest_origin = plan['origin_at']

    def sample(self, now):
        while self.pending is not None and now >= self.pending['scheduled_start']:
            if self.current is not None:
                self.current['status'] = 'replaced'
            self.current = self.pending
            self.pending = self.queued.pop(0) if self.queued else None
            self.current['status'] = 'running'
            if self.waiting_since is not None:
                self.paused_s += now-self.waiting_since
            self.waiting_since = None
        if self.current is None:
            return None
        index = int(math.floor((now-self.current['origin_at'])/DT+1e-9))
        if index >= len(self.current['rows']):
            if self.drain and self.pending is None:
                self.current['status'] = 'done'
                self.finished = True
                return None
            if self.wait_on_exhaustion:
                if self.waiting_since is None:
                    self.waiting_since = now
                    self.wait_epoch += 1
                    self.current['status'] = 'waiting_for_prediction'
                return None
            if self.hold_last:
                self.current['status'] = 'holding_last_action'
                return self.current, len(self.current['rows'])-1
            raise RuntimeError('Action prediction horizon exhausted before a new prediction arrived')
        if index < 0:
            raise RuntimeError('Cannot execute an action before its prediction origin')
        return self.current, index

    def record_dispatch(self, plan, index, now, sampled_at=None):
        plan.setdefault('actual_start', now)
        plan.setdefault('first_sample_at', now if sampled_at is None else sampled_at)
        plan['executed_steps'] += 1
        plan.setdefault('executed_action_indices', []).append(index)
        plan['last_dispatch_at'] = now
        if now >= plan['origin_at']+len(plan['rows'])*DT:
            plan['held_last_action_ticks'] = plan.get('held_last_action_ticks', 0)+1
            plan['last_hold_at'] = now

    def status(self):
        now = time.monotonic()
        paused = self.pause_duration(now)
        keys = ('plan_id', 'status', 'origin_at', 'scheduled_start', 'actual_start', 'first_sample_at',
                'executed_steps', 'executed_action_indices', 'last_dispatch_at',
                'held_last_action_ticks', 'last_hold_at')
        return dict(finished=self.finished, draining=self.drain, hold_last=self.hold_last,
                    wait_on_exhaustion=self.wait_on_exhaustion,
                    waiting_for_prediction=self.waiting_since is not None,
                    waiting_since=self.waiting_since, wait_epoch=self.wait_epoch,
                    pipeline=self.pipeline, pending_plan_ids=[p['plan_id'] for p in ([self.pending] if self.pending else [])+self.queued],
                    arrival_handoff=self.arrival_handoff,
                    paused_s=paused,
                    source_elapsed_s=max(0., now-self.history[0]['origin_at']-paused) if self.history else 0.,
                    predictions=[{k: p[k] for k in keys if k in p} for p in self.history])
