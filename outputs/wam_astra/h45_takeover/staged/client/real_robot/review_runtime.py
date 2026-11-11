"""Independent event/hold/review coordination. No model socket ownership here."""
from collections import deque
from copy import deepcopy
import math
import threading
import time
import uuid


class ReviewRuntime:
    def __init__(self,device,review_job,*,monitor_source,poll_interval_s,review_timeout_s,
                 prediction_gate=None,audit=None,clock=time.monotonic):
        for name,value in (('poll_interval_s',poll_interval_s),('review_timeout_s',review_timeout_s)):
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Positive finite '+name+' required')
        if poll_interval_s>=review_timeout_s or not callable(review_job) or not callable(monitor_source):
            raise ValueError('Bounded polling and callable review job/monitor source required')
        self.device=device;self.review_job=review_job;self.monitor_source=monitor_source
        self.period=poll_interval_s;self.timeout=review_timeout_s;self.clock=clock
        self.prediction_gate=prediction_gate;self.audit=audit
        self._lock=threading.RLock();self._pending=threading.Event();self._closed=threading.Event()
        self._wake=threading.Event();self._monitor=None;self._worker=None
        self._fault=None;self._deadline=None;self._result=None;self._analysis=None;self._control=None
        self._seen=set();self._history=deque(maxlen=3);self._candidate=None
        self.phase='open'

    @property
    def pending(self):return self._pending.is_set()

    def wait_pending(self,timeout):return self._pending.wait(timeout)

    def _emit(self,event,**fields):
        if self.audit:
            self.audit(dict(event=event,host_monotonic=self.clock(),phase=self.phase,
                generation=self.device.review_generation,physical_hold_proven=False,**fields))

    def start(self):
        if self._monitor is not None:raise RuntimeError('Review monitor already started')
        self._monitor=threading.Thread(target=self._watch,name='h45-review-monitor',daemon=True)
        self._monitor.start()

    def _fail(self,error):
        with self._lock:
            if self._fault is not None:return
            self._fault=str(error);self.phase='fault';self._wake.set()
        try:self.device.abort()
        except Exception as exc:
            with self._lock:self._fault+='; abort: '+str(exc)
        self._emit('review_failed',reason=self._fault)

    def check(self):
        if callable(getattr(self.prediction_gate,'check',None)):
            try:self.prediction_gate.check()
            except Exception as exc:self._fail(exc)
        if self._deadline is not None and self.clock()>=self._deadline:
            self._fail('Review deadline expired; late model results have no authority')
        if self.device.failed.is_set() and self._fault is None:
            self._fail('Device execution stopped during review runtime')
        if self._fault is not None:raise RuntimeError(self._fault)

    def _watch(self):
        while not self._closed.wait(self.period):
            try:
                self.check()
                sample=self.monitor_source()
                if not isinstance(sample,dict):raise RuntimeError('Monitor source must return an object')
                dog=sample.get('unitree',{})
                if dog.get('fault') or dog.get('hardware_fault'):
                    raise RuntimeError('Device feedback fault: '+str(dog.get('fault') or dog.get('hardware_fault')))
                if not self.pending and dog:
                    if dog.get('review_epoch')!=self.device.dog_review_epoch:
                        raise RuntimeError('Device epoch changed outside the active review protocol')
                event=sample.get('event')
                if event is not None:self.request_review(event)
            except Exception as exc:
                self._fail(exc)
                return

    def request_review(self,event):
        if (not isinstance(event,dict) or any(not isinstance(event.get(k),str) or not event[k].strip()
                                           for k in ('event_id','reason'))):
            raise ValueError('A review event requires stable event_id and reason')
        with self._lock:
            if self._closed.is_set():raise RuntimeError('Review runtime closed')
            if event['event_id'] in self._seen:return False
            if self.pending and self.phase!='held':return False
            self._seen.add(event['event_id'])
            # Rechecks during the SAME hold never extend its original budget.
            if self._deadline is None:self._deadline=self.clock()+self.timeout
            self._pending.set();self.device.review_requested.set();self.phase='pausing'
            self._result=None;self._analysis=None
            self._control=dict(schema_version=1,request_id=uuid.uuid4().hex,
                               session_id=self.device.session,generation=None)
            self._worker=threading.Thread(target=self._review,args=(deepcopy(event),),
                                          name='h45-astra-review',daemon=True)
            self._emit('review_requested',event_id=event['event_id'],
                       checkpoint_kinds=event.get('checkpoint_kinds',[]))
            self._worker.start()
            return True

    def _review(self,event):
        try:
            self.device.pause_for_review(reason=event['reason'],deadline_host_s=self._deadline)
            with self._lock:
                self._control['generation']=self.device.review_generation
                control=deepcopy(self._control);self.phase='holding'
                deadline=self._deadline
            self._emit('paired_pause_ack',request_id=control['request_id'])
            while not self._closed.is_set():
                self.check()
                if self.device.hold_status().get('ready') is True:break
                self._closed.wait(self.period)
            if self._closed.is_set():return
            self._emit('hold_ready',request_id=control['request_id'])
            if hasattr(self.device,'refresh_review_clocks'):self.device.refresh_review_clocks()
            self.check()
            packet=self.device.capture_review(deadline)
            with self._lock:
                self._analysis=deepcopy(packet);history=deepcopy(list(self._history))
                candidate=deepcopy(self._candidate);self.phase='astra_pending'
            self._emit('astra_started',request_id=control['request_id'])
            result=self.review_job(packet=packet,control=control,event=event,deadline_s=deadline,
                                   history=history,candidate=candidate)
            self.check()
            value=result.get('runtime_policy_directive') if isinstance(result,dict) else None
            if not isinstance(value,dict):raise RuntimeError('No validated runtime policy directive')
            if any(value.get(k)!=control[k] for k in ('request_id','session_id','generation')):
                raise RuntimeError('Foreign or obsolete Astra review identity')
            if type(value.get('generation')) is not int or self.device.review_generation!=control['generation']:
                raise RuntimeError('Astra generation changed while model was running')
            if value.get('directive') not in ('keep_holding','replan_policy'):
                raise RuntimeError('Unsupported Astra runtime directive')
            instruction=value.get('policy_instruction')
            if not isinstance(instruction,str) or (value['directive']=='replan_policy')!=bool(instruction.strip()):
                raise RuntimeError('Invalid Astra policy instruction')
            if callable(getattr(self.prediction_gate,'reviewed',None)):
                self.prediction_gate.reviewed(result,candidate=candidate,event=event,generation=control['generation'])
            with self._lock:
                if self._closed.is_set() or self._fault:return
                self._result=deepcopy(result)
                self.phase='replan_ready' if value['directive']=='replan_policy' else 'held'
                self._wake.set()
            self._emit('astra_completed',request_id=control['request_id'],directive=value['directive'])
        except Exception as exc:self._fail(exc)

    def superseded(self,generation):
        return self.pending or self.device.review_requested.is_set() or generation!=self.device.review_generation

    def remember(self,packet):
        with self._lock:
            if not self.pending:self._history.append(deepcopy(packet))

    def on_prediction(self,packet,actions,generation):
        if self.superseded(generation):return True
        with self._lock:
            self._candidate=dict(observation_id=packet['observation_id'],generation=generation,
                origin_at=packet['stamp'],model_actions=actions.tolist(),
                action_encoding='H45 raw model schema; not absolute joint angles',committed=False)
            if callable(getattr(self.prediction_gate,'committed',None)):
                self._candidate['checkpoint_anchor']=dict(state=deepcopy(packet['state']),
                    p450_packet={k:deepcopy(v) for k,v in packet['p450_packet'].items() if k!='images'})
        if self.prediction_gate:
            event=self.prediction_gate(packet,actions,generation)
            if event is not None:self.request_review(event)
        return self.superseded(generation)

    def recover(self,check_wait):
        """Foreground only: the previous policy reply must already be drained."""
        while self.pending:
            check_wait();self.check()
            with self._lock:
                ready=self.phase=='replan_ready';analysis=deepcopy(self._analysis)
                result=deepcopy(self._result);control=deepcopy(self._control)
            if not ready:
                self._wake.wait(self.period);self._wake.clear();continue
            if self.device.hold_status().get('ready') is not True:
                self._closed.wait(self.period);continue
            if hasattr(self.device,'refresh_review_clocks'):self.device.refresh_review_clocks()
            self.check()
            packet=self.device.capture_review(self._deadline)
            # Compare each stamp only in its own device domain. The recovery
            # observation is never the packet used before the long Astra call.
            if (packet['observation_id']==analysis['observation_id'] or packet['stamp']<=analysis['stamp']
                    or packet['p450_packet']['stamp']<=analysis['p450_packet']['stamp']):
                self._closed.wait(self.period);continue
            self.check()
            recovery_actions=None
            if callable(getattr(self.prediction_gate,'recovery_actions',None)):
                recovery_actions=self.prediction_gate.recovery_actions(packet,control['generation'])
            self.device.resume_after_review(packet)
            self.check()
            with self._lock:
                self.phase='resume_pending';self._pending.clear()
            self._emit('fresh_recovery_observation',observation_id=packet['observation_id'])
            return dict(packet=packet,generation=control['generation'],
                        policy_instruction=result['runtime_policy_directive']['policy_instruction'],review=result,
                        recovery_actions=recovery_actions)
        return None

    def note_commit(self,generation):
        error=None
        candidate=None
        with self._lock:
            if self._fault is not None:error=self._fault
            elif self._deadline is not None and self.clock()>=self._deadline:
                error='Review deadline expired before paired commit acknowledgement'
            elif self.superseded(generation):return False
            else:
                if self._candidate and self._candidate['generation']==generation:self._candidate['committed']=True
                candidate=deepcopy(self._candidate)
                if self.phase=='resume_pending':
                    self.phase='open';self._deadline=None
        if error is not None:
            self._fail(error)
            return False
        if candidate is not None and callable(getattr(self.prediction_gate,'committed',None)):
            try:self.prediction_gate.committed(candidate)
            except Exception as exc:
                self._fail(exc)
                return False
        self._emit('paired_commit',generation_at_prediction=generation)
        return True

    def close(self):
        self._closed.set();self._wake.set()
        if self.pending or self.phase=='resume_pending':
            try:self.device.abort()
            except Exception:pass
        if self._monitor and self._monitor is not threading.current_thread():self._monitor.join(timeout=1.)
