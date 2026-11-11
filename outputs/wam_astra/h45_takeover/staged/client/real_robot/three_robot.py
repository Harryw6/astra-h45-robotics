"""Windows H45 coordinator for Go2, Piper and P450 on a common RTC schedule."""
import argparse
import concurrent.futures
from getpass import getpass
import json
import math
from pathlib import Path
from queue import Empty, Queue
import shlex
import signal
import threading
import time
import uuid

from .controller import Device
from .review_errors import ReviewSuperseded
from .core import DT
from .goal_distance_realtime import RecordedEpisode, RemotePolicy, require_unitree_execution_ready, run, save

CAPABILITY = 'h45_coordinated_start_v1'
ARRIVAL_CAPABILITY = 'h45_coordinated_arrival_v1'
P450_VERSION = 'p450_h45_position_stream_v6'
P450_RAW_VERSION = 'p450_h45_position_stream_v4'
P450_GROUND_VERSION = 'p450_h45_position_stream_v3'
P450_LEGACY_VERSION = 'p450_h45_position_stream_v2'
P450_ACTIONS = ['uav.target_dx_heading','uav.target_dy_heading','uav.target_dz','uav.target_dyaw']
FUTURE_LEAD = .35
ARRIVAL_MIN_LEAD = .085
ARRIVAL_SEND_LEAD = .100  # Leave margin under the gateway's 120 ms window for the Windows clock tick.
P450_ARRIVAL_MIN_LEAD = .010
REPLAY_ROOT = '/home/ROBOT_USER/p450_position_replay_20261102'
BRIDGE_PATH = '/home/ROBOT_USER/p450_online_h45_20261106/p450_online_h45.py'
_DEFAULT_RPC_TIMEOUT = object()
GROUND_SITE_CONFIG = '/home/ROBOT_USER/p450_online_h45_20261106/site.task12.ground.json'


class AircraftClient:
    def __init__(self, args, trace):
        import paramiko
        self.lock = threading.Lock()
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.trace_lock = threading.Lock()
        self.trace = trace
        self.seq = 0
        self.offset = 0.
        self.uncertainty = float('inf')
        self.clock_checked = 0.
        self.closed = False
        self.positions = 0
        self.remote_error_log='/home/ROBOT_USER/astra_h45_release_20261108_01/bridge_'+uuid.uuid4().hex+'.stderr.log'
        self.conn = paramiko.SSHClient()
        self.conn.load_system_host_keys(str(Path.home()/'.ssh'/'known_hosts'))
        self.conn.set_missing_host_key_policy(paramiko.RejectPolicy())
        if args.p450_known_host:
            known = paramiko.HostKeys(str(Path.home()/'.ssh'/'known_hosts')).lookup(args.p450_known_host)
            if not known:
                raise ValueError('No pinned P450 SSH key for '+args.p450_known_host)
            for algorithm,key in known.items():
                self.conn.get_host_keys().add(args.p450_host,algorithm,key)
        passwd = getpass('P450 SSH password: ') if args.ask_p450_password else None
        try:
            self.conn.connect(args.p450_host,username='amov',password=passwd,key_filename=args.p450_key,
                              look_for_keys=passwd is None,allow_agent=passwd is None,
                              timeout=5,banner_timeout=6,auth_timeout=6)
            self.conn.get_transport().set_keepalive(5)
            cmd = shlex.join(['python3','-u',BRIDGE_PATH,'--replay-root',REPLAY_ROOT,
                   '--site-config',args.site_config,'--tracking-error-m',str(args.tracking_error_m)]
                   + (['--site-confirmed'] if args.site_confirmed else [])
                   + (['--ground-start'] if getattr(args, 'ground_start', False) else [])
                   + (['--live-observations'] if getattr(args, 'live_observations', False) else [])
                   + (['--no-speed-limit'] if getattr(args, 'no_speed_limit', False) else []))
            setup = ('source /opt/ros/noetic/setup.bash && '
                     'source /home/ROBOT_USER/prometheus_mavros/devel/setup.bash && '
                     'source /home/ROBOT_USER/p450_experiment/devel/setup.bash')
            remote = 'bash -lc '+shlex.quote('{ '+setup+'; } >&2 && exec '+cmd+' 2>'+shlex.quote(self.remote_error_log))
            self.log(dict(event='bridge_start',remote_error_log=self.remote_error_log))
            self.stdin, out, err = self.conn.exec_command(remote)
            self.channel = self.stdin.channel
            self.channel.settimeout(.35)
            for stream, protocol in ((out,True),(err,False)):
                threading.Thread(target=self.read,args=(stream,protocol),daemon=True).start()
            # Starting Python and ROS can take longer than a steady-state
            # command. Keep startup time out of the clock-offset samples.
            self.call('clock',timeout=7.)
        except BaseException:
            self.conn.close()
            raise

    def log(self, value):
        with self.trace_lock:
            with self.trace.open('a',encoding='utf-8') as handle:
                handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

    def read(self, stream, protocol):
        import socket
        partial = b''
        try:
            while True:
                try:
                    chunk = stream.channel.recv(65536) if protocol else stream.channel.recv_stderr(65536)
                except socket.timeout:
                    if self.closed:
                        break
                    continue
                if not chunk:
                    break
                partial += chunk
                while b'\n' in partial:
                    line, partial = partial.split(b'\n', 1)
                    self._handle_line(line.decode('utf-8',errors='replace'), protocol)
        except Exception as exc:
            self.log(dict(event='bridge_reader_error',error=str(exc)))
        finally:
            if protocol:
                self.close()

    def _handle_line(self, line, protocol):
                if protocol and line.startswith('H45_RPC '):
                    answer = json.loads(line[8:])
                    with self.pending_lock:
                        reply = self.pending.get(answer.get('id'))
                    if reply is None:
                        raise RuntimeError('Unexpected P450 response ID')
                    reply.put(answer)
                else:
                    try:
                        data = json.loads(line)
                    except ValueError:
                        data = dict(event='remote_log',line=line.strip())
                    if isinstance(data,dict) and data.get('event') == 'position':
                        self.positions += 1
                    self.log(data)

    def call(self, op, timeout=_DEFAULT_RPC_TIMEOUT, **body):
        if timeout is _DEFAULT_RPC_TIMEOUT:
            timeout = None if op in ('prepare', 'commit', 'observe', 'observe_live') else .32
        reply = Queue()
        # Serialize writes only. Waiting for a chunk ACK must not prevent a
        # heartbeat (or stop) from travelling over the same SSH connection.
        with self.lock:
            with self.pending_lock:
                if self.closed:
                    raise RuntimeError('P450 SSH channel is closed')
                self.seq += 1
                request_id = self.seq
                self.pending[request_id] = reply
            try:
                self.stdin.write(json.dumps(dict(id=request_id,op=op,**body),allow_nan=False)+'\n')
                self.stdin.flush()
            except BaseException:
                self.close()
                raise
        try:
            try:
                answer = reply.get(timeout=timeout)
            except Empty:
                self.close()
                raise RuntimeError(f'P450 {op} response timed out') from None
            if answer is None or answer.get('id') != request_id or answer.get('error'):
                detail = answer.get('error') if answer else 'SSH process closed'
                self.log(dict(event='rpc_failure',op=op,request_id=request_id,detail=detail,
                              remote_error_log=self.remote_error_log))
                self.close()
                raise RuntimeError(f'P450 {op} failed: {detail or "response ID mismatch"}')
            return answer
        finally:
            with self.pending_lock:
                self.pending.pop(request_id, None)

    def calibrate(self):
        samples = []
        for _ in range(5):
            before = time.monotonic()
            answer = self.call('clock')
            after = time.monotonic()
            samples.append(((after-before)/2,answer['monotonic']-(before+after)/2))
        self.uncertainty,self.offset = min(samples)
        self.clock_checked = time.monotonic()
        if not all(map(math.isfinite,(self.uncertainty,self.offset))):
            raise RuntimeError('P450 monotonic clock sample is invalid')

    def close(self):
        with self.pending_lock:
            self.closed = True
            for reply in self.pending.values():
                reply.put(None)
        self.conn.close()


class ThreeRobotDevice(Device):
    """Extend only the existing H45 prepare/commit and watchdog boundary."""
    def __init__(self, url, token, aircraft, ground_start=False, no_speed_limit=False):
        super().__init__(url,token,timeout=1.2)
        self.response_timeouts.update({endpoint: None for endpoint in
                                      ('/stream/prepare', '/stream/commit', '/observation')})
        self.aircraft = aircraft
        self.staged = None
        self.snapshot = None
        self.plane_started = False
        self.plane_stop = None
        self.paired = 0
        self.failed = threading.Event()
        self.finishing = threading.Event()
        self.stop_lock = threading.Lock()
        self.commit_lock = threading.Lock()
        self.last_commit_s = 0.
        self.effective_end = None
        self.phase = None
        self.ground_start = ground_start
        self.no_speed_limit = no_speed_limit

    def calibrate(self):
        status = super().calibrate()
        if not {CAPABILITY,ARRIVAL_CAPABILITY} <= set(status.get('capabilities',[])):
            raise RuntimeError('Dog gateway does not advertise coordinated H45 arrival')
        if self.no_speed_limit:
            if status.get('velocity_limits_enforced') is not False:
                raise RuntimeError('Go2 gateway still enforces an application velocity limit')
            if (status.get('piper_target_policy') != 'raw_targets_sdk_firmware_limits_unchanged'
                    or status.get('actuator_setpoints_rate_limited') is not False):
                raise RuntimeError('Piper gateway does not report raw, unrate-limited targets')
        if not self.plane_started:
            plane = self.aircraft.call('status',timeout=7)
            versions = ({P450_VERSION} if self.no_speed_limit else
                        {P450_VERSION, P450_RAW_VERSION, P450_GROUND_VERSION} if self.ground_start else
                        {P450_VERSION, P450_RAW_VERSION, P450_GROUND_VERSION, P450_LEGACY_VERSION})
            if plane.get('version') not in versions:
                raise RuntimeError('P450 H45 bridge version mismatch')
            if self.no_speed_limit and (plane.get('reference_speed_limited') is not False
                                        or plane.get('reference_acceleration_limited') is not False
                                        or plane.get('reference_smoothed') is not True):
                raise RuntimeError('P450 bridge does not report uncapped, smoothed references')
        self.aircraft.calibrate()
        if self.uncertainty+self.aircraft.uncertainty > .025:
            raise RuntimeError('Combined dog/P450 clock uncertainty exceeds 25 ms')
        return status

    def plane_time(self, dog_time):
        return dog_time-self.offset+self.aircraft.offset

    def stop_plane(self):
        with self.stop_lock:
            if self.plane_started and not self.aircraft.closed:
                try:
                    self.plane_stop = self.aircraft.call('stop',timeout=.35)
                except Exception as exc:
                    self.plane_stop = dict(error=str(exc))
                finally:
                    self.aircraft.close()

    def abort(self):
        self.failed.set()
        # Neither device can wait for model inference to return to stop.
        stop_plane = threading.Thread(target=self.stop_plane,daemon=True)
        stop_plane.start()
        try:
            return super().call('/stop',{})
        finally:
            stop_plane.join(timeout=.8)

    def wait_for_ground_start(self):
        announced = False
        while True:
            status = self.aircraft.call('ground_wait_status',timeout=5)
            if status.get('ready') is True:
                self.aircraft.call('check_ground_start',timeout=5)
                return
            if not announced:
                print(json.dumps(dict(event='p450_waiting_for_operator',
                      message=('实时闭环已待命。请手动起飞、稳定悬停并切入策略，然后让狗站稳。'
                               if getattr(self, 'live_input', False) else
                               '预测已预热。请解锁并切入策略控制，程序会自动接入动作流。')),
                      ensure_ascii=False),flush=True)
                announced = True
            time.sleep(.05)

    def wait_for_operator_ready(self):
        if not self.ground_start:
            return super().call('/status')
        self.wait_for_ground_start()
        print(json.dumps(dict(event='waiting_for_go2_standing',
                              message='飞机策略通道已就绪；等待狗站稳。'),ensure_ascii=False),flush=True)
        while True:
            plane = self.aircraft.call('ground_wait_status', timeout=5)
            if not plane.get('ready'):
                self.wait_for_ground_start()
            status = super().call('/status')
            if not status.get('execution_enabled') or any(status.get(k) for k in ('fault','hardware_fault','sensor_error')):
                require_unitree_execution_ready(status)
            if status.get('go2_standing') is not False:
                require_unitree_execution_ready(status)
                return status
            time.sleep(.1)

    def validate_initial_plan(self, actions, duration_s):
        if self.ground_start:
            self.aircraft.call('preview_takeoff', timeout=5,
                               duration_s=float(duration_s),
                               actions=[[float(value) for value in row[10:14]] for row in actions])

    def call(self, endpoint, body=None):
        if endpoint == '/stop':
            return self.abort()
        if endpoint == '/stream/finish':
            self.finishing.set()
            self.stop_plane()
            if (not self.plane_stop or self.plane_stop.get('error')
                    or self.plane_stop.get('hold_published') is not True):
                raise RuntimeError('P450 stop/hold acknowledgement missing')
            return super().call(endpoint,body)
        if endpoint == '/reset':
            if self.ground_start:
                # Check the aircraft before the dog/arm gateway enters a new session.
                self.wait_for_ground_start()
                require_unitree_execution_ready(super().call('/status'))
            dog = super().call(endpoint,body)
            try:
                self.aircraft.call('begin',timeout=3)
                self.plane_started = True
                return dog
            except BaseException:
                try:
                    self.abort()
                except Exception:
                    pass
                raise
        if endpoint == '/stream/heartbeat':
            try:
                if self.failed.is_set():
                    raise RuntimeError('Three-robot execution already stopped')
                dog = super().call(endpoint,body)
                if not self.finishing.is_set():
                    self.aircraft.call('heartbeat')
                return dog
            except Exception:
                if self.finishing.is_set() and not self.failed.is_set():
                    return super().call(endpoint,body)
                try:
                    self.abort()
                except Exception:
                    pass
                raise
        if self.failed.is_set() and endpoint in ('/observation','/stream/prepare','/stream/commit'):
            raise RuntimeError('Three-robot execution already stopped')
        if endpoint == '/observation':
            observation = super().call(endpoint,body)
            aircraft = self.aircraft.call('observe')
            self.snapshot = aircraft
            return {**observation, 'p450_anchor': aircraft['pose'],
                    'p450_anchor_offset_s': aircraft['stamp']-self.plane_time(observation['stamp'])}
        if endpoint == '/stream/prepare':
            # A concurrent producer must wait for BOTH commit replies before
            # sending even the prepare request for the following chunk.
            with self.commit_lock:
                if self.failed.is_set():
                    raise RuntimeError('Three-robot execution already stopped')
                if self.staged is not None:
                    raise RuntimeError('Previous chunk is still awaiting commit')
                return self.prepare(body)
        if endpoint == '/stream/commit':
            with self.commit_lock:
                if self.failed.is_set():
                    raise RuntimeError('Three-robot execution already stopped')
                return self.commit(body)
        return super().call(endpoint,body)

    def prepare(self, body):
        self.staged = None
        plan = uuid.uuid4().hex
        fresh = bool(body.get('bootstrap') or body.get('resume_from_wait'))
        origin = body['origin_at']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            dog_future = pool.submit(super().call, '/stream/prepare', dict(body, plan_id=plan))
            plane_future = pool.submit(self.aircraft.call, 'prepare', plan_id=plan,
                                       anchor_id=self.snapshot['anchor_id'],
                                       origin=self.plane_time(origin), switch=None,
                                       fresh_start=fresh,
                                       live_continuous=body.get('live_continuous', False),
                                       **self.review_fields(plane=True),
                                       actions=[row[10:14] for row in body['actions']])
            dog = dog_future.result()
            plane = plane_future.result()
        if plane.get('accepted') is False:
            if plane.get('rejected') is True:
                raise ReviewSuperseded('P450 rejected preparation: '+str(plane.get('reason')))
            raise RuntimeError('P450 rejected preparation: '+str(plane.get('reason')))
        if not dog.get('accepted'):
            return dog
        if dog['plan_id'] != plan:
            raise RuntimeError('Gateway must support client-assigned chunk IDs')
        remaining = dog.get('remaining_execution_s') if fresh else None
        continuous = body.get('live_continuous', False)
        if continuous and dog.get('live_continuous') is not True:
            raise RuntimeError('Gateway did not acknowledge continuous live mode')
        if fresh and not continuous and (remaining is None or not math.isfinite(remaining) or remaining <= 0):
            raise RuntimeError('Gateway must support per-endpoint arrival starts')
        self.remaining_execution_s = remaining
        self.live_continuous = continuous
        self.staged = (plan,fresh,origin,None)
        return dog

    def commit(self, body):
        plan,fresh,origin,_ = self.staged
        if body['plan_id'] != plan:
            raise RuntimeError('Prepared dog/P450 plan IDs do not match')
        try:
            commit_started = time.perf_counter()
            continuous = getattr(self, 'live_continuous', False)
            if not fresh and not continuous and self.effective_end is None:
                raise RuntimeError('Previous episode end is unavailable')
            # No shared execution deadline: each endpoint switches on receipt.
            # Both jobs are launched before waiting for either acknowledgement.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                dog_future = pool.submit(super().call, '/stream/commit', dict(plan_id=plan,**self.review_fields()))
                plane_future = pool.submit(self.aircraft.call, 'commit', plan_id=plan,
                                           arrival_on_receive=True,
                                           **self.review_fields(plane=True),
                                           live_continuous=continuous,
                                           remaining_execution_s=getattr(self, "remaining_execution_s", None),
                                           end=None if fresh or continuous else self.plane_time(self.effective_end))
                dog = dog_future.result()
                plane = plane_future.result()
            if plane.get('accepted') is False:
                if plane.get('rejected') is True:
                    raise ReviewSuperseded('P450 rejected commit: '+str(plane.get('reason')))
                raise RuntimeError('P450 rejected commit: '+str(plane.get('reason')))
            if not dog.get('accepted'):
                raise RuntimeError('Dog rejected a concurrently committed P450 prediction: '
                                   + str(dog.get('reason')))
            if not fresh and dog['origin_at'] != origin:
                raise RuntimeError('Dog committed a different observation origin')
            self.effective_end = dog['effective_end_at']
            self.last_commit_s = time.perf_counter()-commit_started
            self.paired += 1
            return {**dog,'paired_ack':dict(plan_id=plan, dog=True, p450=True),
                    'p450_commit':dict(plan_id=plan,confirmed_at=plane['monotonic'],
                                      aircraft_switch_at=plane['scheduled_start'],
                                      round_trip_s=self.last_commit_s)}
        except Exception:
            if getattr(self,'review_phase',None) not in ('pausing','holding'):
                try:
                    self.abort()
                except Exception:
                    pass
            raise
        finally:
            self.staged = None

    def review_fields(self,plane=False):
        """Optional review transport, unchanged legacy wire format otherwise."""
        epoch=getattr(self,'p450_review_epoch' if plane else 'dog_review_epoch',None)
        if epoch is None:
            return {}
        result=dict(review_epoch=epoch)
        if not plane:
            result['session']=self.session
        if hasattr(self,'review_deadline_fields'):result.update(self.review_deadline_fields(plane=plane))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','execute'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-step',type=int,default=25000)
    parser.add_argument('--policy-url',help='Override the configured model tunnel')
    parser.add_argument('--p450-host',default='P450_HOST_REQUIRED')
    parser.add_argument('--p450-known-host',help='Verify a changed IP against the existing trusted host name or IP')
    parser.add_argument('--p450-key')
    parser.add_argument('--ask-p450-password',action='store_true')
    parser.add_argument('--site-config',help='Absolute path to an approved onboard site configuration')
    parser.add_argument('--site-confirmed',action='store_true')
    parser.add_argument('--ground-start',action='store_true',
                        help='Use the initial policy trajectory for P450 takeoff from the ground')
    parser.add_argument('--no-speed-limit',action='store_true',
                        help='Smooth P450 references without speed or acceleration limits')
    parser.add_argument('--tracking-error-m',type=float)
    args = parser.parse_args()
    if args.site_config is None:
        args.site_config = (GROUND_SITE_CONFIG if args.ground_start or args.no_speed_limit else
                            REPLAY_ROOT+'/site.h45.task04.smooth.live.json')
    if args.mode == 'execute' and not (args.site_confirmed and args.tracking_error_m is not None):
        parser.error('Execution requires --site-confirmed and --tracking-error-m from the flight operator')
    if args.tracking_error_m is None:
        args.tracking_error_m = .5  # Only passed to the read-only check.
    if not 0 < args.tracking_error_m <= .5:
        parser.error('--tracking-error-m must be in (0, 0.5]')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    def interrupt(*_):
        raise KeyboardInterrupt('Operator stopped three-robot execution')
    signal.signal(signal.SIGINT,interrupt)
    signal.signal(signal.SIGTERM,interrupt)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace = args.output.with_suffix('.p450.jsonl')
    with trace.open('x',encoding='utf-8'):
        pass
    aircraft = AircraftClient(args,trace)
    device = ThreeRobotDevice(config['unitree'],args.token_file.read_text().strip(),aircraft,
                              ground_start=args.ground_start,no_speed_limit=args.no_speed_limit)
    source = policy = None
    try:
        status = device.calibrate()
        policy = RemotePolicy(args.policy_url or config['policy'],checkpoint_step=args.checkpoint_step)
        if (policy.metadata.get('action_names',[])[10:] != P450_ACTIONS
                or policy.metadata.get('action_units',[])[10:] != ['m','m','m','rad']
                or policy.metadata.get('fps') != 15):
            raise ValueError('H45 server P450 action channels or units do not match the position executor')
        if any(policy.metadata.get(key) != value for key,value in {
                'uav_encoding':'target_pose_relative_to_measured_chunk_start',
                'uav_xy_frame':'chunk_start_yaw_forward_left',
                'uav_yaw_encoding':'wrapped_radians',
                'anchor':'one_causal_measured_pose_at_observation_time_for_all_45_steps'}.items()):
            raise ValueError('H45 server P450 coordinate frame or anchor contract does not match')
        source = RecordedEpisode(args.episode)
        if args.mode == 'check':
            print(json.dumps(dict(mode='check',checkpoint=policy.metadata['checkpoint'],
                                  dog=status,p450=aircraft.call('status',timeout=7),
                                  source_episode=source.metadata['source_episode']),indent=2))
        else:
            summary = run(source,policy,device,args.output,True,'unitree',rtc_delay_s=2.0,rtc_handoff='arrival')
            published = (device.plane_stop or {}).get('positions_published')
            if published is None and aircraft.positions:
                published = aircraft.positions
            summary.update(execution_targets=['go2','piper','p450'],paired_commits=device.paired,
                           uav_commands=published>0 if published is not None else None,
                           p450_stop=device.plane_stop,p450_trace=str(trace))
            save(args.output/'summary.json',summary)
            if not summary['complete']:
                raise SystemExit(1)
    finally:
        device.stop_plane()
        aircraft.close()
        if source is not None:
            source.close()
        if policy is not None:
            policy.close()


if __name__ == '__main__':
    main()
