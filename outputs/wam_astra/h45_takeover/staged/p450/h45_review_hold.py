"""Explicitly configured feedback checks; no ROS, model or actuator access."""
import math


def number(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise RuntimeError('Invalid finite '+name)
    return float(value)


def vector(value, name):
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise RuntimeError('Expected four values: '+name)
    return [number(v,name) for v in value]


def settings(value):
    if not isinstance(value,dict) or not (value.get('calibration_verified') is True
            or value.get('configuration_source') == 'h45_runtime_v1'):
        raise RuntimeError('Review hold requires explicit runtime settings')
    result=dict(value)
    required=value.get('require_deadline_binding',True)
    if type(required) is not bool:raise RuntimeError('require_deadline_binding must be boolean')
    result['require_deadline_binding']=required
    for name in ('max_feedback_age_s','max_image_age_s','max_review_s'):
        result[name]=number(value.get(name),name)
        if result[name]<=0: raise RuntimeError('Positive '+name+' required')
    result['min_hold_duration_s']=number(value.get('min_hold_duration_s'),'min_hold_duration_s')
    if result['min_hold_duration_s']<0: raise RuntimeError('Negative stable duration')
    if type(value.get('min_hold_samples')) is not int or value['min_hold_samples']<2:
        raise RuntimeError('At least two distinct feedback samples required')
    result.pop('max_pose_drift',None)  # Retired adapter-only threshold; tolerate old configs.
    for name in ('max_velocity',):
        result[name]=vector(value.get(name),name)
        if any(v<=0 for v in result[name]): raise RuntimeError('Positive '+name+' required')
    return result


def frame_evidence(config,*,image_source,state_source,image_at,state_at,now,ros_now):
    """Check each clock only against the same clock domain, never subtract them."""
    for name,stamp,limit in (('image',image_at,config['max_image_age_s']),
                             ('state',state_at,config['max_feedback_age_s'])):
        if not 0<=number(now,'monotonic')-number(stamp,name+' arrival')<=limit:
            raise RuntimeError('Stale/future '+name+' arrival')
    for name,stamp,limit in (('image',image_source,config['max_image_age_s']),
                             ('state',state_source,config['max_feedback_age_s'])):
        source_age(stamp,ros_now,limit,name)
    return dict(image_source_stamp=float(image_source),state_source_stamp=float(state_source),
                source_freshness_verified=True,source_clock_domain='ROS',
                arrival_clock_domain='aircraft_monotonic')


def source_age(stamp,ros_now,limit,name):
    source=number(stamp,name+' source')
    if source<=0 or not 0<=number(ros_now,'ROS now')-source<=limit:
        raise RuntimeError('Stale/future/invalid '+name+' source timestamp')


class ReviewHold:
    def __init__(self,config,sample,now):
        self.config=settings(config)
        self.started=number(now,'now')
        self.entry=self.validate(sample,now)
        self.target=list(self.entry['pose'])
        self.previous=self.entry
        self.source_advanced_at=self.entry['stamp']
        self.since=None;self.samples=0;self.ready=False;self.failed=False

    def validate(self,sample,now):
        now=number(now,'now')
        stamp=number(sample.get('stamp'),'feedback stamp')
        source=number(sample.get('source_stamp'),'source stamp')
        if source<=0 or not 0<=now-stamp<=self.config['max_feedback_age_s']:
            raise RuntimeError('Stale/future/invalid review feedback')
        return dict(stamp=stamp,source_stamp=source,pose=vector(sample.get('pose'),'pose'),
                    velocity=vector(sample.get('velocity'),'velocity'))

    def update(self,sample,now):
        if self.failed: raise RuntimeError('Review feedback failure already latched')
        try:
            if not 0<=now-self.started<=self.config['max_review_s']:
                raise RuntimeError('Review deadline expired')
            current=self.validate(sample,now)
            delta=[a-b for a,b in zip(current['pose'],self.target)]
            delta[3]=math.atan2(math.sin(delta[3]),math.cos(delta[3]))
            self.pose_error=delta  # Diagnostic only; native flight limits remain in Aircraft.
            if (current['stamp']<self.previous['stamp']
                    or current['source_stamp']<self.previous['source_stamp']):
                raise RuntimeError('Review feedback clock moved backwards')
            advanced=(current['stamp']>self.previous['stamp']
                      and current['source_stamp']>self.previous['source_stamp'])
            if current['source_stamp']>self.previous['source_stamp']:
                self.source_advanced_at=current['stamp']
            if now-self.source_advanced_at>self.config['max_feedback_age_s']:
                raise RuntimeError('Review source timestamp stopped advancing')
            republished=(current['stamp']>self.previous['stamp']
                         and current['source_stamp']==self.previous['source_stamp'])
            stationary=all(abs(v)<=limit for v,limit in zip(current['velocity'],self.config['max_velocity']))
            self.previous=current
            if not stationary or republished:
                self.since=None;self.samples=0;self.ready=False
                return
            if not advanced:
                return  # Never count a duplicate; age is checked above.
            if self.since is None: self.since=current['stamp']
            self.samples+=1
            self.ready=(self.samples>=self.config['min_hold_samples']
                        and current['stamp']-self.since>=self.config['min_hold_duration_s'])
        except Exception:
            self.failed=True;self.ready=False
            raise

    def summary(self):
        return dict(ready=self.ready and not self.failed,samples=self.samples,
                    stable_since=self.since,target_pose=list(self.target),failed=self.failed,
                    physical_hold_proven=False,task_outcome='unknown')
