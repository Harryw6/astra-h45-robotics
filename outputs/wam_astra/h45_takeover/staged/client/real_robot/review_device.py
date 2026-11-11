"""Paired H45 review transport; device feedback, not this client, gates hold."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import math
import threading
import time
import uuid

from .controller import Device
from .live_h45 import LiveDevice
from .review_errors import ReviewSuperseded


class ReviewDevice(LiveDevice):
    def __init__(self,*args,review_rpc_timeout_s,deadline_clock_max_age_s=None,
                 deadline_clock_margin_s=None,**kwargs):
        if (type(review_rpc_timeout_s) not in (int,float)
                or not math.isfinite(review_rpc_timeout_s) or review_rpc_timeout_s<=0):
            raise ValueError('Positive finite review RPC timeout required')
        super().__init__(*args,**kwargs)
        self.review_rpc_timeout_s=review_rpc_timeout_s
        self.response_timeouts.update({path:review_rpc_timeout_s for path in
            ('/review/pause','/review/status','/review/resume')})
        self.review_requested=threading.Event()
        self.review_lock=threading.Lock()
        self.review_phase='open'
        self.review_generation=0
        self.dog_review_epoch=None
        self.p450_review_epoch=None
        self.session=None
        self.require_bootstrap=False
        self.last_review_receipt=None
        if deadline_clock_max_age_s is not None and (type(deadline_clock_max_age_s) not in (int,float)
                or not math.isfinite(deadline_clock_max_age_s) or deadline_clock_max_age_s<=0):
            raise ValueError('Positive finite deadline clock age required')
        self.deadline_clock_max_age_s=deadline_clock_max_age_s
        self.deadline_clock_margin_s=deadline_clock_margin_s
        self.remote_deadlines=None
        self.review_transaction_id=None
        self.review_deadline_host_s=None
        self.deadline_support_verified=False
        self.deadline_clock_maps=None

    def _paired(self,dog_endpoint,dog_body,plane_op,plane_body):
        # Start both requests before waiting. Every result is observed, including
        # partial failure; callers abort both rather than proceed with one side.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(Device.call,self,dog_endpoint,dog_body),
                     pool.submit(self.aircraft.call,plane_op,
                                 timeout=self.review_rpc_timeout_s,**plane_body)]
            values=[];errors=[]
            for future in futures:
                try: values.append(future.result())
                except BaseException as exc: values.append(None);errors.append(exc)
        if errors: raise errors[0]
        return values

    @contextmanager
    def _commit_barrier(self):
        if not self.commit_lock.acquire(timeout=self.review_rpc_timeout_s):
            raise RuntimeError('Review commit barrier timed out; endpoint abort required')
        try:
            yield
        finally:
            self.commit_lock.release()

    def calibrate(self):
        if self.deadline_clock_max_age_s is None:
            raise RuntimeError('Configure deadline_clock_max_age_s before review execution')
        self._check_deadline_clock_margin()
        status=super().calibrate()
        plane=self.aircraft.call('review_status',timeout=self.review_rpc_timeout_s)
        if ('h45_review_pause_v1' not in status.get('capabilities',[])
                or 'h45_review_deadline_v1' not in status.get('capabilities',[])
                or status.get('review',{}).get('configured') is not True
                or status.get('review',{}).get('review_deadline_required') is not True
                or plane.get('configured') is not True or plane.get('review_deadline_required') is not True):
            raise RuntimeError('Both endpoints need configured H45 review support')
        self.dog_review_epoch=status['review_epoch']
        self.p450_review_epoch=plane['review_epoch']
        self.deadline_support_verified=True
        self.deadline_clock_maps=self.review_clock_snapshot()
        return status

    def call(self,endpoint,body=None):
        if endpoint=='/reset' and not self.deadline_support_verified:
            raise RuntimeError('Verify remote review deadline support before starting execution')
        result=super().call(endpoint,body)
        if endpoint=='/reset':
            self.session=result['session'];self.dog_review_epoch=result['review_epoch']
        return result

    def pause_for_review(self,*,reason,deadline_host_s=None):
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Review reason required')
        with self.review_lock:
            if self.review_phase=='holding':
                return self.last_review_receipt
            if self.session is None or self.dog_review_epoch is None or self.p450_review_epoch is None:
                raise RuntimeError('Review endpoints/session have not been initialized')
            self.review_requested.set()  # Inference cannot race another commit.
            self.review_generation+=1
            self.review_phase='pausing'
            started=time.perf_counter()
            try:
                proposals=None
                if deadline_host_s is not None:
                    if self.review_deadline_host_s is not None:
                        deadline_host_s=min(deadline_host_s,self.review_deadline_host_s)
                    proposals=self.mapped_review_deadlines(deadline_host_s)
                elif self.deadline_support_verified:
                    raise RuntimeError('Runtime review requires a host deadline')
                with self._commit_barrier():
                    self.staged=None
                    transaction=uuid.uuid4().hex if proposals is not None else None
                    dog_body=dict(session=self.session,review_epoch=self.dog_review_epoch)
                    plane_body=dict(review_epoch=self.p450_review_epoch)
                    if proposals is not None:
                        dog_body.update(review_transaction_id=transaction,review_deadline_s=proposals['unitree'])
                        plane_body.update(review_transaction_id=transaction,review_deadline_s=proposals['p450'])
                    dog,plane=self._paired('/review/pause',
                        dog_body,'review_pause',plane_body)
                    if (dog.get('session')!=self.session
                            or type(dog.get('review_epoch')) is not int
                            or dog['review_epoch']!=self.dog_review_epoch+1
                            or dog.get('review',{}).get('phase')!='holding'
                            or type(plane.get('review_epoch')) is not int
                            or plane['review_epoch']!=self.p450_review_epoch+1
                            or plane.get('phase')!='holding'):
                        raise RuntimeError('Paired review pause acknowledgement mismatch')
                    if proposals is not None:
                        actual={}
                        for key,meta,answer in (('unitree',dog['review'],dog),('p450',plane,plane)):
                            bound=meta.get('review_deadline_s');stamp=answer.get('monotonic')
                            if (meta.get('review_deadline_required') is not True
                                    or meta.get('review_deadline_active') is not True
                                    or meta.get('review_transaction_id')!=transaction
                                    or type(bound) not in (int,float) or not math.isfinite(bound)
                                    or type(stamp) not in (int,float) or not math.isfinite(stamp)
                                    or bound>proposals[key] or bound<=stamp):
                                raise RuntimeError('Invalid/expired remote review deadline acknowledgement')
                            actual[key]=bound
                        self.remote_deadlines=actual;self.review_transaction_id=transaction
                        self.review_deadline_host_s=deadline_host_s
                    self.dog_review_epoch=dog['review_epoch'];self.p450_review_epoch=plane['review_epoch']
                self.review_phase='holding'
                self.last_review_receipt=dict(phase='holding',reason=reason,
                    generation=self.review_generation,session=self.session,dog=dog,p450=plane,
                    pause_request_to_ack_s=time.perf_counter()-started,
                    ready=False,physical_hold_proven=False,task_outcome='unknown')
                return self.last_review_receipt
            except BaseException:
                self.review_phase='fault'
                self.abort()
                raise

    def hold_status(self):
        dog,plane=self._paired('/review/status',None,'review_status',{})
        if (dog.get('session')!=self.session or dog.get('review_epoch')!=self.dog_review_epoch
                or plane.get('review_epoch')!=self.p450_review_epoch
                or dog.get('phase')!='holding' or plane.get('phase')!='holding'):
            raise RuntimeError('Review hold state/epoch changed')
        if self.remote_deadlines is not None:
            for key,meta in (('unitree',dog),('p450',plane)):
                if (meta.get('review_transaction_id')!=self.review_transaction_id
                        or meta.get('review_deadline_s')!=self.remote_deadlines[key]
                        or meta.get('review_deadline_active') is not True):
                    raise RuntimeError('Review deadline grant changed while holding')
        return dict(phase='holding',ready=dog.get('ready') is True and plane.get('ready') is True,
                    dog=dog,p450=plane,physical_hold_proven=False,task_outcome='unknown')

    def monitor_status(self):
        # Neither operation creates a model observation nonce/aircraft anchor.
        before=time.monotonic()
        dog,plane=self._paired('/status',None,'review_status',{})
        after=time.monotonic()
        if (0<=after-before<=.025 and all(type(v.get('monotonic')) in (int,float)
                and math.isfinite(v['monotonic']) for v in (dog,plane))):
            midpoint=(before+after)/2;uncertainty=(after-before)/2
            # Dedicated deadline estimates do not retime an executing policy.
            self.deadline_clock_maps={key:dict(offset_s=value['monotonic']-midpoint,
                uncertainty_s=uncertainty,checked_host_s=after) for key,value in (('unitree',dog),('p450',plane))}
        return dict(unitree=dog,p450=plane)

    def review_clock_snapshot(self):
        return dict(unitree=dict(offset_s=self.offset,uncertainty_s=self.uncertainty,
                                  checked_host_s=self.clock_checked),
                    p450=dict(offset_s=self.aircraft.offset,uncertainty_s=self.aircraft.uncertainty,
                              checked_host_s=self.aircraft.clock_checked))

    def mapped_review_deadlines(self,deadline_host_s,*,now_s=None):
        now=time.monotonic() if now_s is None else now_s
        if (type(deadline_host_s) not in (int,float) or not math.isfinite(deadline_host_s)
                or deadline_host_s<=now):
            raise RuntimeError('Review host deadline is invalid or expired')
        if self.deadline_clock_max_age_s is None:
            raise RuntimeError('Deadline clock validity must be explicitly configured')
        margin=self._check_deadline_clock_margin()
        maps=self.deadline_clock_maps or self.review_clock_snapshot()
        for value in maps.values():
            if (any(type(value[k]) not in (int,float) or not math.isfinite(value[k])
                    for k in ('offset_s','uncertainty_s','checked_host_s'))
                    or value['uncertainty_s']<0
                    or not 0<=now-value['checked_host_s']<=self.deadline_clock_max_age_s):
                raise RuntimeError('Deadline clock calibration is stale/invalid')
        if sum(value['uncertainty_s'] for value in maps.values())>.025:
            raise RuntimeError('Deadline clock uncertainty exceeds H45 contract')
        # server = host + offset, with true offset >= estimate-uncertainty.
        return {key:deadline_host_s+value['offset_s']-value['uncertainty_s']-margin for key,value in maps.items()}

    def _check_deadline_clock_margin(self):
        # Site configuration must bound relative clock drift over calibration
        # age + review horizon, in addition to reported host timer resolution.
        # RTT bounds alone do not establish that drift bound.
        margin=self.deadline_clock_margin_s
        if (type(margin) not in (int,float) or not math.isfinite(margin)
                or margin<time.get_clock_info('monotonic').resolution):
            raise RuntimeError('Explicit deadline clock margin must cover host timer resolution and verified drift')
        return margin

    def refresh_review_clocks(self):
        with self.review_lock:
            if self.review_phase!='holding' or not self.review_requested.is_set():
                raise RuntimeError('Review clock refresh requires paired hold')
            status=Device.calibrate(self)
            self.aircraft.calibrate()
            if (status.get('fault') or status.get('hardware_fault')
                    or status.get('review_epoch')!=self.dog_review_epoch
                    or self.uncertainty+self.aircraft.uncertainty>.025):
                raise RuntimeError('Review clock/feedback verification failed')
            self.deadline_clock_maps=self.review_clock_snapshot()
            return self.review_clock_snapshot()

    def capture_review(self,deadline_s):
        if self.review_phase!='holding' or not self.review_requested.is_set():
            raise RuntimeError('Review observation requires the paired hold')
        while time.monotonic()<deadline_s:
            remaining=deadline_s-time.monotonic()
            timeout=min(self.review_rpc_timeout_s,remaining)
            # A separate bounded HTTP handle avoids mutating the foreground
            # stream's timeout settings while it may still be receiving.
            dog_client=Device(self.url,self.token,timeout=timeout)
            with ThreadPoolExecutor(max_workers=2) as pool:
                dog_future=pool.submit(dog_client.call,'/observation')
                plane_future=pool.submit(self.aircraft.call,'observe_live',timeout=timeout)
                dog=dog_future.result();plane=plane_future.result()
            if plane.get('waiting_for_image'):
                time.sleep(min(.01,max(0.,deadline_s-time.monotonic())));continue
            self.snapshot=plane
            return {**dog,'p450_anchor':plane['pose'],'p450_packet':plane}
        raise RuntimeError('Fresh review capture deadline expired')

    def resume_after_review(self,packet):
        with self.review_lock,self._commit_barrier():
            if self.review_phase!='holding' or not self.review_requested.is_set():
                raise RuntimeError('No paused review to resume')
            if self.hold_status().get('ready') is not True:
                raise RuntimeError('Both measured holds must be ready')
            plane_packet=packet.get('p450_packet',{})
            if (packet.get('session')!=self.session or packet.get('review_epoch')!=self.dog_review_epoch
                    or plane_packet.get('review_epoch')!=self.p450_review_epoch):
                raise RuntimeError('Fresh pause-era observation epoch mismatch')
            try:
                dog,plane=self._paired('/review/resume',
                    dict(session=self.session,review_epoch=self.dog_review_epoch,
                         observation_id=packet['observation_id'],**self.review_deadline_fields()),
                    'review_resume',dict(review_epoch=self.p450_review_epoch,anchor_id=plane_packet['anchor_id'],
                                         **self.review_deadline_fields(plane=True)))
                if (dog.get('session')!=self.session or dog.get('review_epoch')!=self.dog_review_epoch
                        or dog.get('review',{}).get('phase')!='open'
                        or dog.get('review_resume_pending') is not True
                        or plane.get('review_epoch')!=self.p450_review_epoch
                        or plane.get('phase')!='resume_pending'):
                    raise RuntimeError('Paired review resume acknowledgement mismatch')
                if self.remote_deadlines is not None:
                    for key,meta in (('unitree',dog['review']),('p450',plane)):
                        if (meta.get('review_transaction_id')!=self.review_transaction_id
                                or meta.get('review_deadline_s')!=self.remote_deadlines[key]
                                or meta.get('review_deadline_active') is not True):
                            raise RuntimeError('Remote review deadline changed on resume')
                self.require_bootstrap=True
                self.effective_end=None
                self.review_phase='resume_pending'
                self.review_requested.clear()
                return dict(phase='resume_pending',generation=self.review_generation,
                            physical_hold_proven=False,dog=dog,p450=plane)
            except BaseException:
                self.review_phase='fault';self.abort();raise

    def _prediction_authority(self,body):
        if self.review_requested.is_set() or self.review_phase not in ('open','resume_pending'):
            raise ReviewSuperseded('Prediction blocked by review')
        if type(body.get('review_generation')) is not int or body['review_generation']!=self.review_generation:
            raise ReviewSuperseded('Obsolete prediction generation')

    def prepare(self,body):
        self._prediction_authority(body)
        if self.require_bootstrap and body.get('bootstrap') is not True:
            raise RuntimeError('Review recovery requires fresh bootstrap and empty RTC')
        result=super().prepare(dict(body,session=self.session,review_epoch=self.dog_review_epoch,
                                    **self.review_deadline_fields()))
        try:
            self._prediction_authority(body)
        except RuntimeError:
            self.staged=None
            raise
        return result

    def commit(self,body):
        self._prediction_authority(body)
        result=super().commit(body)
        self._prediction_authority(body)
        if result.get('accepted'):
            if self.remote_deadlines is not None:
                paired=result.get('paired_ack',{})
                if paired.get('dog') is not True or paired.get('p450') is not True:
                    self.abort()
                    raise RuntimeError('Review deadline requires paired acknowledgement before release')
            self.require_bootstrap=False;self.review_phase='open'
            self.remote_deadlines=None;self.review_transaction_id=None;self.review_deadline_host_s=None
        return result

    def review_deadline_fields(self,plane=False):
        if self.remote_deadlines is None:return {}
        return dict(review_transaction_id=self.review_transaction_id,
                    review_deadline_s=self.remote_deadlines['p450' if plane else 'unitree'])
