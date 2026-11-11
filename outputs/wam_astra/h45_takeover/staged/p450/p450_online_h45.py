"""P450 H45 position executor. JSON requests over SSH, ROS on the aircraft."""
import base64
import argparse
from collections import deque
import json
import math
from pathlib import Path
from queue import Empty, Queue
import signal
import sys
import threading
import time
if __package__:
    from .h45_review_hold import ReviewHold, settings as review_settings, frame_evidence, source_age
    from .h45_review_deadline import DeadlineGrant
else:
    from h45_review_hold import ReviewHold, settings as review_settings, frame_evidence, source_age
    from h45_review_deadline import DeadlineGrant

DT = 1 / 15
LEASE = .65
TOPIC = '/uav1/p450_policy/command'
CONTROLLER = '/uav_control_main_1'
VERSION = 'p450_h45_position_stream_v6'
LIFTOFF_TIMEOUT_S = 8.
MIN_TAKEOFF_RISE_M = .20
MAX_PRESTART_DRIFT_M = .10
SMOOTH_VELOCITY_TAU_S = .10


class ReviewRejected(RuntimeError):
    """Obsolete protocol authority; retain the existing healthy hold."""


class GroundStart:
    """Accept on-ground or the controller's brief automatic liftoff."""
    def __init__(self, origin):
        self.origin = origin
        self.first_command_at = None
        self.airborne = False

    def check(self, state, now, *, check_timeout=True):
        if not (state['armed'] and state['mode'] == 'OFFBOARD' and state['control_state'] == 2):
            raise RuntimeError('Ground start requires armed OFFBOARD COMMAND_CONTROL')
        landed = state['landed']
        if self.airborne:
            if landed != 2:
                raise RuntimeError('P450 left the airborne state during policy execution')
        elif landed == 2:
            self.airborne = True
        elif landed not in (1, 3):
            raise RuntimeError('P450 landed state is invalid for ground start')
        if self.first_command_at is None and landed == 1:
            pose = state['pose']
            drift = math.sqrt(sum((getattr(pose, k)-getattr(self.origin, k))**2 for k in ('x','y','z')))
            if drift > MAX_PRESTART_DRIFT_M:
                raise RuntimeError('P450 moved before the first ground-start command')
        elif (self.first_command_at is not None and check_timeout and not self.airborne
              and now-self.first_command_at > LIFTOFF_TIMEOUT_S):
            raise RuntimeError('P450 did not report liftoff before the ground-start timeout')

    def check_plan(self, targets):
        if self.airborne:
            return
        if not targets or max(p.z for p in targets) < self.origin.z + MIN_TAKEOFF_RISE_M:
            raise RuntimeError('P450 ground-start prediction does not contain a takeoff')
        if min(p.z for p in targets) < self.origin.z - .03:
            raise RuntimeError('P450 ground-start prediction goes below the takeoff floor')

    def note_command(self, now):
        if self.first_command_at is None:
            self.first_command_at = now


def finite(values):
    values = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Expected finite values')
    return values


def world_target(pose_type, anchor, row):
    x, y, z, yaw = finite(row)
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    return pose_type(anchor.x + c*x - s*y, anchor.y + s*x + c*y,
                     anchor.z + z, anchor.yaw + yaw)


class Smoother:
    """The desktop replay limiter, retaining reference velocity across chunks."""
    def __init__(self, origin, caps):
        self.pose = origin
        self.velocity = (0., 0., 0.)
        self.yaw_rate = 0.
        self.caps = caps

    def step(self, target):
        pose = self.pose
        horizontal, vertical, yaw_max, accel, yaw_accel = self.caps
        vx, vy = 2*(target.x-pose.x), 2*(target.y-pose.y)
        norm = math.hypot(vx, vy)
        if norm > horizontal:
            vx, vy = vx*horizontal/norm, vy*horizontal/norm
        vz = max(-vertical, min(vertical, 2*(target.z-pose.z)))
        delta = tuple(want-old for want, old in zip((vx,vy,vz), self.velocity))
        norm = math.sqrt(sum(x*x for x in delta))
        factor = min(1., accel*DT/norm) if norm else 1.
        self.velocity = tuple(old+factor*d for old,d in zip(self.velocity,delta))
        yaw_error = math.atan2(math.sin(target.yaw-pose.yaw), math.cos(target.yaw-pose.yaw))
        want = max(-yaw_max, min(yaw_max, 2*yaw_error))
        self.yaw_rate += max(-yaw_accel*DT, min(yaw_accel*DT, want-self.yaw_rate))
        dx, dy, dz = (v*DT for v in self.velocity)
        self.pose = type(pose)(pose.x+dx, pose.y+dy, pose.z+dz, pose.yaw+self.yaw_rate*DT)
        return self.pose, self.velocity


class RawReference:
    """Pass the model's position target unchanged; derive velocity without clipping."""
    def __init__(self, origin):
        self.pose = origin

    def step(self, target, previous_target):
        velocity = tuple((getattr(target,k)-getattr(previous_target,k))/DT for k in ('x','y','z'))
        self.pose = target
        return target, velocity


class UncappedSmoother:
    """Smooth the replay controller's position and velocity without saturation."""
    def __init__(self, origin):
        self.pose = origin
        self.velocity = (0., 0., 0.)
        self.yaw_rate = 0.
        self.alpha = 1. - math.exp(-DT / SMOOTH_VELOCITY_TAU_S)

    def step(self, target):
        pose = self.pose
        desired = tuple(2. * (getattr(target,k) - getattr(pose,k)) for k in ('x','y','z'))
        self.velocity = tuple(old + self.alpha * (want-old)
                              for old,want in zip(self.velocity,desired))
        yaw_error = math.atan2(math.sin(target.yaw-pose.yaw), math.cos(target.yaw-pose.yaw))
        self.yaw_rate += self.alpha * (2. * yaw_error - self.yaw_rate)
        self.pose = type(pose)(pose.x+self.velocity[0]*DT, pose.y+self.velocity[1]*DT,
                               pose.z+self.velocity[2]*DT, pose.yaw+self.yaw_rate*DT)
        return self.pose, self.velocity


class Schedule:
    def __init__(self):
        self.plans = deque()
        self.end = None

    def add(self, plan, end, replace_after=None):
        if replace_after is not None:
            while self.plans and self.plans[-1]['switch'] > replace_after:
                self.plans.pop()
        if (end is not None and end <= plan['switch']) or (self.plans and plan['switch'] <= self.plans[-1]['switch']):
            raise ValueError('Invalid P450 plan order or episode end')
        if len(self.plans) >= 64:
            raise RuntimeError('P450 schedule queue is full')
        self.plans.append(plan)
        self.end = end

    def sample(self, now):
        while len(self.plans) > 1 and self.plans[1]['switch'] <= now:
            self.plans.popleft()
        if not self.plans or (self.end is not None and now >= self.end):
            return None
        plan = self.plans[0]
        if now < plan['switch']:
            return None
        index = math.floor((now-plan['origin'])/DT + 1e-9)
        if not 0 <= index < 45:
            return None
        return plan['id'], index, plan['targets'][index]


class Aircraft:
    def __init__(self, args):
        import rospy
        import rosgraph
        from prometheus_msgs.msg import UAVCommand, UAVState, UAVControlState
        from mavros_msgs.msg import ExtendedState, RCIn
        from geometry_msgs.msg import PoseStamped
        from core import Pose, SafetyError
        from official_fence import check_plan
        from official_trajectory_replay import valid, message
        from ros_replay import Telemetry, other_publishers
        from startup_checks import controller_snapshot, check_controller_snapshot

        self.rospy, self.message_type, self.pose_type = rospy, UAVCommand, Pose
        self.valid, self.message, self.check_plan = valid, message, check_plan
        self.safety_error = SafetyError
        self.other_publishers = other_publishers
        self.verify_controller = lambda: check_controller_snapshot(controller_snapshot(rospy), args.replay_root)
        config = json.loads(args.site_config.read_text(encoding='utf-8-sig'))
        self.review_config=None if config.get('review_pause') is None else review_settings(config['review_pause'])
        self.review_epoch=0
        self.review_window=None
        self.review_resume_pending=False
        self.review_deadline=None
        self.review_anchor_image_stamp=None
        self.review_entry_image_source_stamp=None
        self.review_anchor_sources=None
        self.site_confirmed = args.site_confirmed
        self.ground_start = getattr(args, 'ground_start', False)
        self.no_speed_limit = getattr(args, 'no_speed_limit', False)
        if self.no_speed_limit and config.get('allow_unbounded_reference_speed') is not True:
            raise SafetyError('The site configuration does not allow uncapped reference speeds')
        self.max_duration = None
        if self.ground_start:
            self.max_duration, = finite((config.get('max_duration_s', 0),))
            if self.max_duration <= 0:
                raise SafetyError('Ground start requires an approved positive site duration')
        self.ground_guard = None
        self.px4_lock = threading.Lock()
        self.px4_pose = None
        self.px4_arrival = None
        self.px4_source_stamp = None
        self.config_verified = config.get('site_verified') is True
        h,v,y,a,w,b = finite(config[k] for k in (
            'horizontal_reference_speed_m_s','vertical_reference_speed_m_s','yaw_reference_rate_deg_s',
            'reference_acceleration_m_s2','yaw_acceleration_deg_s2','min_battery'))
        if not (0 < h <= .83 and 0 < v <= .83 and 0 < y <= 80 and 0 < a <= .7
                and 0 < w <= 100 and 0 <= b <= 1):
            raise SafetyError('P450 site configuration exceeds replay limits')
        self.caps, self.min_battery = (h,v,math.radians(y),a,math.radians(w)), b
        if (UAVCommand.TRAJECTORY, UAVCommand.Move, UAVCommand.Current_Pos_Hover, UAVState.MID360) != (6,4,2,10):
            raise SafetyError('Installed Prometheus message schema does not match replay')
        rospy.init_node('p450_h45_online', anonymous=True, disable_signals=True)
        self.master = rosgraph.Master(rospy.get_name())
        self.telemetry = Telemetry()
        self.subscribers = [
            rospy.Subscriber('/uav1'+suffix, msg, lambda m, k=key: self.telemetry.callback(k,m), queue_size=1)
            for suffix,msg,key in (
                ('/prometheus/state',UAVState,'state'),('/prometheus/control_state',UAVControlState,'control'),
                ('/mavros/extended_state',ExtendedState,'extended'),('/mavros/rc/in',RCIn,'rc'))
        ]
        if self.ground_start:
            self.subscribers.append(rospy.Subscriber('/uav1/mavros/local_position/pose', PoseStamped,
                                                     self.px4_pose_callback, queue_size=1))
        self.live_observations = getattr(args, 'live_observations', False)
        self.live_frame = None
        self.live_lock = threading.Lock()
        if self.live_observations:
            from sensor_msgs.msg import Image
            self.subscribers.append(rospy.Subscriber(args.camera_topic, Image,
                                    self.live_image_callback, queue_size=1, buff_size=4*1024*1024))
        self.publisher = None
        self.baseline = None
        self.anchor = None
        self.anchor_id = 0
        self.fence = None
        self.schedule = Schedule()
        self.prepared = None
        self.sequence = 0
        self.positions = 0
        self.position_attempted = False
        self.closed = False
        self.hold_published = None
        self.lease_until = None
        self.smoother = None
        self.last_topic_check = 0.
        self.topic_error = None
        self.track_max = args.tracking_error_m

    def live_image_callback(self, msg):
        # Associate measured state with each received image, never a later RPC reply.
        with self.telemetry.lock:
            state = self.telemetry.messages.get('state')
            state_at = self.telemetry.arrivals.get('state')
        if state is None:
            return
        with self.live_lock:
            self.live_frame = (msg, state, state_at, time.monotonic())

    def live_observe(self):
        if not self.live_observations:
            raise RuntimeError('Start the bridge with --live-observations')
        self.snapshot(airborne=False if self.publisher is None else True)
        with self.live_lock:
            frame = self.live_frame
        if frame is None:
            return dict(waiting_for_image=True)
        msg, state, state_at, image_at = frame
        sources=None
        if self.review_config is not None:
            sources=frame_evidence(self.review_config,
                image_source=msg.header.stamp.to_sec(),state_source=state.header.stamp.to_sec(),
                image_at=image_at,state_at=state_at,now=time.monotonic(),
                ros_now=self.rospy.Time.now().to_sec())
        from cv_bridge import CvBridge
        import cv2
        bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, encoded = cv2.imencode('.png', bgr)
        if not ok:
            raise RuntimeError('P450 image encoding failed')
        yaw = float(state.attitude[2])
        vx, vy, vz = map(float, state.velocity)
        self.anchor = self.pose_type(*map(float, state.position), yaw)
        self.anchor_id += 1
        self.anchor_stamp = state_at
        self.review_anchor_image_stamp=image_at
        self.review_anchor_sources=sources
        return dict(anchor_id=self.anchor_id, stamp=state_at, pose=vars(self.anchor),
                    review_epoch=self.review_epoch,source_freshness=sources,
                    state4=[math.cos(yaw)*vx+math.sin(yaw)*vy,
                            -math.sin(yaw)*vx+math.cos(yaw)*vy, vz, float(state.attitude_rate[2])],
                    images={'uav':base64.b64encode(encoded).decode('ascii')},
                    image_stamps={'uav':image_at}, image_ros_stamp=msg.header.stamp.to_sec(),
                    state_ros_stamp=state.header.stamp.to_sec(), image_state_offset_s=image_at-state_at)

    def px4_pose_callback(self, msg):
        try:
            stamp = float(msg.header.stamp.to_sec())
            position = finite((msg.pose.position.x, msg.pose.position.y, msg.pose.position.z))
            orientation = finite((msg.pose.orientation.x, msg.pose.orientation.y,
                                  msg.pose.orientation.z, msg.pose.orientation.w))
            if not math.isfinite(stamp) or stamp <= 0 or not .9 <= sum(x*x for x in orientation) <= 1.1:
                return
        except (AttributeError, TypeError, ValueError):
            return
        with self.px4_lock:
            if self.px4_source_stamp is None or stamp > self.px4_source_stamp:
                self.px4_pose = position
                self.px4_arrival = time.monotonic()
                self.px4_source_stamp = stamp

    def check_px4_pose(self, state):
        with self.px4_lock:
            pose, arrival = self.px4_pose, self.px4_arrival
        if pose is None or arrival is None or not 0 <= time.monotonic()-arrival <= .5:
            raise self.safety_error('ground start: PX4 local position is unavailable or stale')
        difference = math.sqrt(sum((a-getattr(state['pose'],k))**2 for a,k in zip(pose,('x','y','z'))))
        if difference > .2:
            raise self.safety_error('ground start: PX4 and aircraft position disagree')

    def snapshot(self, airborne=True, *, check_takeoff_timeout=True):
        state = self.telemetry.snapshot()
        if self.baseline is None:
            self.valid(state, require_airborne=airborne, min_battery=self.min_battery)
        elif self.ground_guard is not None:
            self.valid(state, self.baseline, require_airborne=False, min_battery=self.min_battery)
            self.check_px4_pose(state)
            self.ground_guard.check(state, time.monotonic(), check_timeout=check_takeoff_timeout)
        else:
            self.valid(state, self.baseline, min_battery=self.min_battery)
        return state

    def ground_wait_status(self):
        if not self.ground_start or not (self.site_confirmed and self.config_verified):
            raise RuntimeError('P450 ground start requires a confirmed site and verified configuration')
        state = self.snapshot(airborne=False)
        self.check_px4_pose(state)
        if state['landed'] not in (1, 2, 3):
            raise RuntimeError('P450 ground start requires ground, taking off, or in-air status')
        self.verify_controller()
        if self.other_publishers(self.rospy, TOPIC):
            raise RuntimeError('P450 command topic is already in use')
        fence = self.rospy.get_param(CONTROLLER+'/geo_fence')
        self.check_plan(state['pose'], [state['pose']], fence)
        return dict(ready=bool(state['armed'] and state['mode'] == 'OFFBOARD'
                               and state['control_state'] == 2),
                    armed=state['armed'], mode=state['mode'],
                    control_state=state['control_state'], landed=state['landed'])

    def check_ground_start(self):
        status = self.ground_wait_status()
        if not status['ready']:
            raise RuntimeError('Ground start requires armed OFFBOARD COMMAND_CONTROL')
        state = self.snapshot(airborne=False)
        self.check_px4_pose(state)
        GroundStart(state['pose']).check(state, time.monotonic())
        return state

    def ground_ready(self):
        self.check_ground_start()
        return dict(ground_start_ready=True)

    def preview_takeoff(self, req):
        # This is a read-only plan check before asking the pilot to switch modes.
        self.ground_wait_status()
        state = self.snapshot(airborne=False)
        duration, = finite((req['duration_s'],))
        if not 0 < duration <= self.max_duration:
            raise RuntimeError('P450 episode duration exceeds the approved site configuration')
        actions = req['actions']
        if len(actions) != 45:
            raise ValueError('P450 initial takeoff preview requires all 45 actions')
        targets = [world_target(self.pose_type, state['pose'], row) for row in actions]
        GroundStart(state['pose']).check_plan(targets)
        fence = self.rospy.get_param(CONTROLLER+'/geo_fence')
        bounds = self.check_plan(state['pose'], targets, fence)
        return dict(preview_valid=True, bounds=bounds)

    def status(self):
        deadline = time.monotonic()+5
        while True:
            try:
                state = self.snapshot(airborne=False)
                break
            except self.safety_error:
                if time.monotonic() >= deadline or self.publisher is not None:
                    raise
                time.sleep(.02)
        self.verify_controller()
        if self.publisher is None and self.other_publishers(self.rospy,TOPIC):
            raise RuntimeError('P450 command topic is already in use')
        return dict(version=VERSION, pose=vars(state['pose']), mode=state['mode'],
                    control_state=state['control_state'],
                    armed=state['armed'], landed=state['landed'], battery=state['battery'],
                    positions_published=self.positions, reference_speed_limited=not self.no_speed_limit,
                    reference_smoothed=True, reference_acceleration_limited=not self.no_speed_limit,
                    live_observations=self.live_observations, live_continuous_supported=True,
                    max_duration_s=self.max_duration)

    def begin(self):
        if self.publisher is not None or not (self.site_confirmed and self.config_verified):
            raise RuntimeError('P450 start requires a fresh site confirmation and verified site config')
        state = self.check_ground_start() if self.ground_start else self.snapshot()
        self.verify_controller()
        if self.other_publishers(self.rospy, TOPIC):
            raise RuntimeError('P450 command topic is already in use')
        self.fence = self.rospy.get_param(CONTROLLER+'/geo_fence')
        self.check_plan(state['pose'], [state['pose']], self.fence)
        self.publisher = self.rospy.Publisher(TOPIC,self.message_type,queue_size=1,latch=False)
        self.baseline = list(state['channels'][:4])
        if self.ground_start:
            self.ground_guard = GroundStart(state['pose'])
            self.ground_guard.check(state, time.monotonic())
        deadline = time.monotonic()+2
        while not self.publisher.get_num_connections():
            if self.rospy.is_shutdown() or time.monotonic() >= deadline:
                raise RuntimeError('Dedicated P450 controller subscriber is unavailable')
            time.sleep(.01)
        self.check_topic()
        threading.Thread(target=self.watch_topic,daemon=True).start()
        self.heartbeat()
        return {}

    def check_topic(self):
        pubs, subs, _ = self.master.getSystemState()
        if set(dict(pubs).get(TOPIC, [])) != {self.rospy.get_name()} or CONTROLLER not in dict(subs).get(TOPIC, []):
            raise RuntimeError('P450 command topic ownership or controller connection changed')
        self.last_topic_check = time.monotonic()

    def watch_topic(self):
        while not self.closed:
            time.sleep(.25)
            if self.closed:
                return
            try:
                self.check_topic()
            except Exception as exc:
                self.topic_error = str(exc)
                return

    def heartbeat(self):
        if self.publisher is None:
            raise RuntimeError('P450 has no execution session')
        self.snapshot()
        self.lease_until = time.monotonic()+LEASE
        return {}

    def observe(self):
        if self.publisher is None:
            raise RuntimeError('P450 has no execution session')
        self.anchor = self.snapshot()['pose']
        self.anchor_id += 1
        self.anchor_stamp = time.monotonic()
        return dict(anchor_id=self.anchor_id, stamp=self.anchor_stamp, pose=vars(self.anchor))

    def prepare(self, req):
        self.check_review_command(req)
        if self.review_resume_pending and req.get('fresh_start') is not True:
            raise RuntimeError('Review resume requires a fresh start at row zero')
        now = time.monotonic()
        origin, = finite((req['origin'],))
        switch = req['switch']
        arrival = switch is None
        fresh = req.get('fresh_start', False)
        continuous = req.get('live_continuous', False)
        if type(continuous) is not bool or (continuous and not getattr(self, 'live_observations', False)):
            raise ValueError('Continuous execution requires live observations')
        if hasattr(self, 'continuous_mode') and continuous != self.continuous_mode:
            raise ValueError('Live/recorded session mode cannot change')
        if type(fresh) is not bool or (fresh and not arrival):
            raise ValueError('Invalid fresh-start option')
        if not arrival:
            switch, = finite((switch,))
        if (self.publisher is None or self.anchor is None or req['anchor_id'] != self.anchor_id
                or not 0 <= now-self.anchor_stamp or len(req['actions']) != 45
                or (arrival and not fresh and not origin <= now < origin+45*DT)
                or (not arrival and (not origin <= switch < origin+45*DT or switch-now < .08))):
            raise ValueError('P450 plan anchor, horizon or switch deadline is invalid')
        self.snapshot()
        targets = [world_target(self.pose_type,self.anchor,row) for row in req['actions']]
        first = 0 if fresh else max(0, math.floor(((now if arrival else switch)-origin)/DT + 1e-9))
        self.check_plan(self.anchor, targets[first:], self.fence)
        if self.ground_guard is not None:
            self.ground_guard.check_plan(targets[first:])
        self.check_review_deadline(req)
        self.prepared = dict(id=req['plan_id'],origin=origin,switch=switch,targets=targets,
                             anchor=self.anchor, fresh_start=fresh, live_continuous=continuous)
        self.anchor = None
        return {}

    def commit(self, req):
        self.check_review_command(req)
        if self.review_resume_pending:
            self.review_window.update(self._review_sample(),self._review_now())
            if not self.review_window.ready or self.review_window.failed:
                raise RuntimeError('Review hold is not ready for commit')
            self.review_window.validate(self.review_window.previous,self._review_now())
            if self._review_now()-self.review_window.started>self.review_config['max_review_s']:
                raise RuntimeError('Review hold deadline expired')
            if self.prepared is None or self.prepared.get('fresh_start') is not True:
                raise RuntimeError('Review resume commit requires a fresh start')
        if self.prepared is None or self.prepared['id'] != req['plan_id']:
            raise RuntimeError('P450 commit plan identity failed')
        now = time.monotonic()
        arrival = self.prepared['switch'] is None
        fresh = self.prepared.get('fresh_start', False)
        immediate = req.get('arrival_on_receive', False)
        if type(immediate) is not bool:
            raise ValueError('Invalid arrival option')
        if immediate:
            if not arrival:
                raise ValueError('Immediate commit requires arrival preparation')
            switch = now
            if fresh:
                self.prepared['origin'] = now
            if not self.prepared['origin'] <= now < self.prepared['origin']+45*DT:
                raise RuntimeError('P450 prediction horizon exhausted')
        else:
            switch = req['switch'] if arrival else self.prepared['switch']
            switch, = finite((switch,))
            if not self.prepared['origin'] <= switch < self.prepared['origin']+45*DT:
                raise RuntimeError('P450 switch is outside the H45 horizon')
            if switch-now < (.010 if arrival else .04):
                raise RuntimeError('P450 commit reached the executor too late')
        remaining = req.get('remaining_execution_s')
        continuous = self.prepared.get('live_continuous', False)
        if req.get('live_continuous', False) != continuous:
            raise ValueError('Commit mode differs from prepared mode')
        if fresh and not continuous and (remaining is None or not math.isfinite(remaining) or not 0 < remaining <= self.max_duration):
            raise ValueError('Invalid remaining execution duration')
        end = None if continuous else (now+remaining if fresh else req['end'])
        if end is not None:
            end, = finite((end,))
        self.snapshot()
        self.check_review_deadline(req)
        if self.review_deadline is not None and self.review_resume_pending:
            self.review_deadline.check_start(switch)
        self.prepared['switch'] = switch
        self.schedule.add(self.prepared,end,replace_after=now if arrival else None)
        if self.review_resume_pending:
            if self.review_deadline is not None:self.review_deadline.release()
            self.review_window=None
            self.review_resume_pending=False
            self.smoother=None
        self.continuous_mode = continuous
        self.prepared = None
        return dict(scheduled_start=switch)

    def publish(self, pose, velocity, hold=False):
        self.sequence += 1
        self.publisher.publish(self.message(self.message_type,pose,velocity,self.sequence,
                                           self.rospy.Time.now(),hold=hold))

    def _review_now(self):
        return time.monotonic()

    def _review_sample(self):
        # The original snapshot retains all controller/RC/localization checks.
        state=self.snapshot()
        if state['landed']!=2:
            raise RuntimeError('Review pause requires airborne state')
        with self.telemetry.lock:
            message=self.telemetry.messages['state']
            stamp=self.telemetry.arrivals['state']
            source=float(message.header.stamp.to_sec())
            source_age(source,self.rospy.Time.now().to_sec(),self.review_config['max_feedback_age_s'],'state')
            return dict(stamp=stamp,source_stamp=source,
                        pose=[*map(float,message.position),float(message.attitude[2])],
                        velocity=[*map(float,message.velocity),float(message.attitude_rate[2])])

    def check_review_command(self,req,*,new_pause=False):
        if self.review_config is None:
            return
        if type(req.get('review_epoch')) is not int or req['review_epoch']!=self.review_epoch:
            raise ReviewRejected('Wrong/missing P450 review epoch')
        if not new_pause:self.check_review_deadline(req)
        if self.review_window is not None and not self.review_resume_pending:
            raise ReviewRejected('P450 is review-paused')
        if self.review_window is not None and self.review_window.failed:
            raise RuntimeError('Review failure is latched')

    def check_review_deadline(self,req):
        grant=getattr(self,'review_deadline',None)
        if grant is not None:
            try:grant.validate(req,now=self._review_now())
            except ValueError as exc:raise ReviewRejected(str(exc)) from exc
        elif (self.review_config is not None and self.review_config.get('require_deadline_binding',True)
                and self.review_window is not None):
            raise ReviewRejected('Missing review deadline grant')

    def review_status(self):
        result=dict(configured=self.review_config is not None,review_epoch=self.review_epoch,
                    phase='open',ready=False,physical_hold_proven=False,task_outcome='unknown')
        result['review_deadline_required']=bool(self.review_config and self.review_config.get('require_deadline_binding',True))
        if self.review_config is not None and self.publisher is not None and not self.closed:
            result['feedback']=self._review_sample()
        grant=getattr(self,'review_deadline',None)
        if grant is not None:result.update(grant.fields(),review_deadline_active=not grant.released)
        if self.review_window is not None:
            result.update(self.review_window.summary(),
                          phase='resume_pending' if self.review_resume_pending else 'holding')
        if grant is not None and not grant.released and self._review_now()>=grant.expires_s:result['ready']=False
        return result

    def pause_review(self,req):
        if self.review_config is None:
            raise RuntimeError('P450 review config is required')
        self.check_review_command(req,new_pause=True)
        if self.publisher is None or self.closed:
            raise RuntimeError('Review pause requires an active control channel')
        grant=DeadlineGrant.from_pause(req,now=self._review_now(),max_review_s=self.review_config['max_review_s'],
            previous=getattr(self,'review_deadline',None),required=self.review_config.get('require_deadline_binding',True))
        sample=self._review_sample()
        window=ReviewHold(self.review_config,sample,self._review_now())
        # serve() serializes requests and ticks. No old schedule item can race
        # after this clear, unlike a separate asynchronous publishing thread.
        self.prepared=None;self.schedule=Schedule();self.anchor=None;self.anchor_stamp=None
        self.anchor_id+=1;self.review_anchor_image_stamp=None;self.smoother=None
        frame=getattr(self,'live_frame',None)
        self.review_entry_image_source_stamp=(float(frame[0].header.stamp.to_sec()) if frame is not None else None)
        self.review_anchor_sources=None
        self.review_epoch+=1;self.review_window=window;self.review_resume_pending=False
        self.review_deadline=grant
        self.check_plan(self.pose_type(*sample['pose']),[self.pose_type(*window.target)],self.fence)
        self.publish(self.pose_type(*window.target),(0.,0.,0.),hold=False)
        if grant is not None:grant.check_time(self._review_now())
        return self.review_status()

    def review_tick(self):
        if self.review_deadline is not None:self.review_deadline.check_time(self._review_now())
        self.review_window.update(self._review_sample(),self._review_now())
        # Move/TRAJECTORY with a fixed measured-entry reference and zero
        # feed-forward. Current_Pos_Hover is not assumed to retain that target.
        if self.review_deadline is not None:self.review_deadline.check_time(self._review_now())
        self.publish(self.pose_type(*self.review_window.target),(0.,0.,0.),hold=False)

    def resume_review(self,req):
        if self.review_window is None or self.review_resume_pending:
            raise RuntimeError('No review hold to resume')
        if type(req.get('review_epoch')) is not int or req['review_epoch']!=self.review_epoch:
            raise RuntimeError('Wrong/missing P450 review epoch')
        self.check_review_deadline(req)
        now=self._review_now()
        self.review_window.update(self._review_sample(),now)
        if not self.review_window.ready or self.review_window.failed:
            raise RuntimeError('P450 measured hold is not ready')
        self.review_window.validate(self.review_window.previous,now)
        if now-self.review_window.started>self.review_config['max_review_s']:
            raise RuntimeError('Review hold deadline expired')
        if (self.anchor is None or req.get('anchor_id')!=self.anchor_id
                or self.anchor_stamp is None or self.review_anchor_image_stamp is None
                or min(self.anchor_stamp,self.review_anchor_image_stamp)<=self.review_window.started
                or not 0<=now-self.anchor_stamp<=self.review_config['max_feedback_age_s']
                or not 0<=now-self.review_anchor_image_stamp<=self.review_config['max_image_age_s']):
            raise RuntimeError('Resume requires a fresh pause-era image and observation')
        sources=self.review_anchor_sources
        if (not isinstance(sources,dict) or sources.get('source_freshness_verified') is not True
                or sources['state_source_stamp']<=self.review_window.entry['source_stamp']
                or (self.review_entry_image_source_stamp is not None
                    and sources['image_source_stamp']<=self.review_entry_image_source_stamp)):
            raise RuntimeError('Resume requires advancing verified image/state source timestamps')
        # Keep the fixed hold and its monitor until a NEW chunk is committed.
        self.check_review_deadline(req)
        self.review_resume_pending=True
        return self.review_status()

    def tick(self, now):
        if self.publisher is None:
            return
        if now >= self.lease_until:
            raise RuntimeError('P450 coordinator heartbeat expired')
        if self.topic_error or now-self.last_topic_check >= LEASE:
            raise RuntimeError(self.topic_error or 'P450 controller topic check expired')
        state = self.snapshot()
        if self.review_window is not None:
            self.review_tick()
            return
        point = self.schedule.sample(now)
        if point is None:
            if self.smoother is not None:
                self.publish(state['pose'], (0.,0.,0.),hold=True)
                self.smoother = None
            return
        plan, index, target = point
        actual = state['pose']
        if self.smoother is None:
            self.smoother = UncappedSmoother(actual) if self.no_speed_limit else Smoother(actual,self.caps)
        previous = self.smoother.pose
        if math.sqrt(sum((getattr(actual,k)-getattr(previous,k))**2 for k in ('x','y','z'))) > self.track_max:
            raise RuntimeError('P450 exceeded approved reference tracking error')
        pose, velocity = self.smoother.step(target)
        if math.sqrt(sum((getattr(actual,k)-getattr(pose,k))**2 for k in ('x','y','z'))) > self.track_max:
            raise RuntimeError('P450 smoothed reference exceeds approved tracking error')
        self.check_plan(actual,[pose],self.fence)
        if self.ground_guard is not None:
            self.ground_guard.note_command(now)
        self.position_attempted = True
        self.publish(pose,velocity)
        self.positions += 1
        print(json.dumps(dict(event='position',plan_id=plan,action_index=index,stamp=now,
             target=vars(target),reference=vars(pose),actual=vars(actual),velocity=velocity)),file=sys.stderr,flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.publisher is not None:
            try:
                if self.positions or self.position_attempted:
                    state = self.snapshot(check_takeoff_timeout=False)
                    self.publish(state['pose'],(0.,0.,0.),hold=True)
                    time.sleep(.1)
                    self.hold_published = True
            except Exception as exc:
                self.hold_published = False
                print('P450 hold unavailable: '+str(exc),file=sys.stderr,flush=True)
            finally:
                self.publisher.unregister()
                self.publisher = None
        for subscriber in self.subscribers:
            subscriber.unregister()


def read_commands(queue):
    try:
        while True:
            line = sys.stdin.readline(65536)
            if not line:
                break
            if not line.endswith('\n'):
                raise ValueError('Oversized P450 bridge request')
            queue.put(json.loads(line),timeout=.25)
    finally:
        queue.put(None,timeout=.25)


def send(request_id, data):
    print('H45_RPC '+json.dumps(dict(id=request_id,monotonic=time.monotonic(),**data),allow_nan=False),flush=True)


def serve(aircraft):
    commands = Queue(maxsize=8)
    threading.Thread(target=read_commands,args=(commands,),daemon=True).start()
    handlers = {'status':lambda _: aircraft.status(), 'clock':lambda _: {},
                'review_pause':aircraft.pause_review,
                'review_status':lambda _: aircraft.review_status(),
                'review_resume':aircraft.resume_review,
                'check_ground_start':lambda _: aircraft.ground_ready(),
                'ground_wait_status':lambda _: aircraft.ground_wait_status(),
                'preview_takeoff':aircraft.preview_takeoff,
                'begin':lambda _: aircraft.begin(), 'heartbeat':lambda _: aircraft.heartbeat(),
                'observe':lambda _: aircraft.observe(), 'observe_live':lambda _: aircraft.live_observe(), 'prepare':aircraft.prepare, 'commit':aircraft.commit}
    next_tick = time.monotonic()
    try:
        while not aircraft.rospy.is_shutdown():
            now = time.monotonic()
            if now >= next_tick:
                try:
                    aircraft.tick(now)
                except Exception as exc:
                    print(json.dumps(dict(event='bridge_fatal',reason=str(exc),
                        monotonic=time.monotonic())),flush=True)
                    raise
                next_tick += (math.floor((now-next_tick)/DT) + 1) * DT
            try:
                req = commands.get(timeout=max(.001,next_tick-time.monotonic()))
            except Empty:
                continue
            if req is None:
                return
            op = req.get('op')
            if op == 'stop':
                aircraft.close()
                send(req['id'],dict(positions_published=aircraft.positions,
                                    hold_published=aircraft.hold_published))
                return
            try:
                data = handlers[op](req)
            except ReviewRejected as exc:
                send(req.get('id'),dict(accepted=False,rejected=True,reason=str(exc),
                                      review_epoch=aircraft.review_epoch))
                continue
            except Exception as exc:
                send(req.get('id'),dict(error=str(exc)))
                raise
            send(req['id'], data)
            if op == 'commit' and aircraft.smoother is None:
                next_tick = min(next_tick, aircraft.schedule.plans[-1]['switch'])
    finally:
        aircraft.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-root',type=Path,required=True)
    parser.add_argument('--site-config',type=Path,required=True)
    parser.add_argument('--site-confirmed',action='store_true')
    parser.add_argument('--ground-start',action='store_true',
                        help='Start from ON_GROUND using the initial model trajectory as the takeoff')
    parser.add_argument('--live-observations', action='store_true')
    parser.add_argument('--camera-topic', default='/uav1/camera/color/image_raw')
    parser.add_argument('--no-speed-limit',action='store_true',
                        help='Smooth position and velocity without applying speed or acceleration caps')
    parser.add_argument('--tracking-error-m',type=float,required=True)
    args = parser.parse_args()
    if not 0 < args.tracking_error_m <= .5:
        parser.error('A reviewed tracking limit in (0, 0.5] m is required')
    sys.path.insert(0,str(args.replay_root))
    def interrupt(*_):
        raise KeyboardInterrupt('P450 bridge terminated')
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    serve(Aircraft(args))


if __name__ == '__main__':
    main()
