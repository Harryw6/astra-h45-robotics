"""Go2-only streaming execution for explicit recorded-video experiments; no Piper I/O."""
import fcntl
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np

from .core import command_limits, finite, heading
from .gateway import Gateway, handler
from .go2_velocity_sequence import validate_feedback, parse_response

ROOT = Path(__file__).resolve().parent.parent


class Go2Feedback:
    kind = 'unitree'
    input_errors = {}
    def __init__(self, hardware):self.hardware=hardware;self.origin=None
    @property
    def error(self):
        try:self.hardware.check_execution();return None
        except Exception as exc:return str(exc)
    def reset(self):self.origin=self.hardware.read_state()
    def snapshot(self, diagnostic=False):
        state=self.hardware.read_state();origin=self.origin or state
        xy=heading(np.array(state['position'][:2])-origin['position'][:2],origin['rpy'][2])
        yaw=state['rpy'][2]-origin['rpy'][2]
        values=[*xy,state['body_height'],*state['rpy'][:2],math.sin(yaw),math.cos(yaw),
                *state['velocity'][:2],state['yaw_speed']]
        return dict(kind='unitree',stamp=time.monotonic(),state=values,images={},image_stamps={},
                    state_stamps=[state['monotonic_ns']/1e9],diagnostic_only=diagnostic,
                    state_schema='go2_execution_feedback_only_not_policy_input',readiness_issues=[])


class Go2Hardware:
    action_kind = 'go2'
    def __init__(self, config):
        self.config=config;self.fault=None;self.execute=True
        self.lock_file=open('/tmp/heterovla-v3-unitree.lock','a')
        fcntl.flock(self.lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.run=ROOT/'outputs'/('go2_validation_live_'+str(time.monotonic_ns()))
        self.run.mkdir(parents=True);self.dds=self.run/'dds';self.dds.mkdir()
        self.motion_socket=Path('/tmp')/('go2-live-'+str(time.monotonic_ns())+'.sock')
        self.processes=[];self.logs=[];self.sensors=Go2Feedback(self)
        try:
            self.spawn('v3_go2_state',[config.get('interface','eth0'),str(self.dds)])
            print(json.dumps(dict(waiting_for_standing=True,run=str(self.run),piper_commands=False)),flush=True)
            deadline=time.monotonic()+180;stable=0
            while time.monotonic()<deadline:
                try:self.read_state();stable+=1
                except (OSError,ValueError,RuntimeError):stable=0
                if stable>=3:break
                time.sleep(.05)
            else:raise RuntimeError('Standing readiness timeout')
            limits=command_limits('go2',config.get('velocity_limits'))
            self.spawn('v3_go2_motion',[config.get('interface','eth0'),str(self.motion_socket),
                                       *map(str,limits),'200','3'])
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                self.read_state()
                try:
                    if self.motion_socket.exists() and self.command('STATE')['state_valid']:break
                except (OSError,RuntimeError):pass
                time.sleep(.02)
            else:raise RuntimeError('Go2 SDK bridge not ready')
        except BaseException:
            self.close();raise
    def spawn(self,name,args):
        log=(self.run/(name+'.log')).open('ab',buffering=0);self.logs.append(log)
        self.processes.append(subprocess.Popen([str(ROOT/'real_robot/build'/name),*args],
                                               stdout=log,stderr=log))
    def read_state(self):
        if any(p.poll() is not None for p in self.processes):raise RuntimeError('Go2 bridge exited')
        packet=json.loads((self.dds/'go2_snapshot.json').read_text())
        return validate_feedback(packet,time.monotonic(),tuple(self.config.get('go2_normal_state_codes',[0])),.075)
    def command(self,text):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(.08);client.connect(str(self.motion_socket));client.sendall((text+'\n').encode())
            result=client.makefile('r').readline(512)
        return parse_response(result)
    def check_execution(self):
        if self.fault:raise RuntimeError(self.fault)
        self.read_state()
    def send(self,row):
        self.check_execution();row=finite(row,(3,));self.command('MOVE %.17g %.17g %.17g'%tuple(row))
    def hold(self):self.command('STOP')
    def halt(self):
        if self.motion_socket.exists():self.command('STOP')
    def close(self):
        if self.motion_socket.exists():
            try:self.command('STOP');self.command('QUIT')
            except Exception:pass
        for p in reversed(self.processes):
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in self.logs:log.close()
        self.lock_file.close()


def main():
    config=json.loads((ROOT/'real_robot/config/unitree.json').read_text())
    token=(Path.home()/'.config/heterovla-live/token').read_text().strip()
    def interrupted(*_):raise KeyboardInterrupt('Operator stopped Go2 gateway')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    hardware=Go2Hardware(config)
    service=Gateway(hardware,execute=True,velocity_scale=1.,
                    velocity_limits=config.get('velocity_limits'))
    try:server=ThreadingHTTPServer(('127.0.0.1',config['port']),handler(service,token));server.timeout=.1
    except BaseException:service.close();raise
    stopped=threading.Event()
    def stop(*_):stopped.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps(dict(ready=True,port=config['port'],action_kind='go2',run=str(hardware.run))),flush=True)
    ended=None
    try:
        while not stopped.is_set():
            server.handle_request()
            if not service.active and (service.fault or service.stream.finished):
                ended=time.monotonic() if ended is None else ended
                if time.monotonic()-ended>2:break
    finally:
        (hardware.run/'final_status.json').write_text(json.dumps(service.status(),indent=2)+'\n')
        server.server_close();service.close()
        print(json.dumps(dict(stopped=True,run=str(hardware.run))),flush=True)


if __name__=='__main__':main()
