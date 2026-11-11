"""One device-local review-to-first-commit grant. No clock synchronization/I/O."""
import math


def finite_number(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Invalid finite '+name)
    return float(value)


class DeadlineGrant:
    def __init__(self,transaction_id,expires_s):
        self.transaction_id=transaction_id
        self.expires_s=expires_s
        self.released=False

    @classmethod
    def from_pause(cls,request,*,now,max_review_s,previous=None,required=True):
        keys={'review_transaction_id','review_deadline_s'}
        if not keys.intersection(request) and not required:
            if previous is not None and not previous.released:
                raise ValueError('An active review deadline cannot be erased by optional binding')
            return None
        transaction=request.get('review_transaction_id')
        if not isinstance(transaction,str) or not transaction.strip() or len(transaction)>128:
            raise ValueError('Missing/invalid review transaction identity')
        deadline=finite_number(request.get('review_deadline_s'),'review deadline')
        local=finite_number(now,'device monotonic')+finite_number(max_review_s,'local review budget')
        if deadline<=0 or max_review_s<=0:raise ValueError('Positive review deadline/budget required')
        if previous is not None and transaction==previous.transaction_id:
            raise ValueError('Review transaction identity cannot be reused for a new pause epoch')
        cutoff=min(deadline,local)
        if previous is not None and not previous.released:
            cutoff=min(cutoff,previous.expires_s)
        return cls(transaction,cutoff)

    def fields(self):
        return dict(review_transaction_id=self.transaction_id,review_deadline_s=self.expires_s)

    def check_time(self,now):
        if not self.released and finite_number(now,'device monotonic')>=self.expires_s:
            raise RuntimeError('Review deadline expired on device')

    def check_start(self,start):
        if not self.released and finite_number(start,'scheduled recovery start')>=self.expires_s:
            raise RuntimeError('Recovery start reaches or exceeds review deadline')

    def validate(self,request,*,now):
        if self.released:
            if {'review_transaction_id','review_deadline_s'}.intersection(request):
                raise ValueError('Review grant already consumed/released')
            return
        if (request.get('review_transaction_id')!=self.transaction_id
                or finite_number(request.get('review_deadline_s'),'review deadline')!=self.expires_s):
            raise ValueError('Mismatched review transaction/deadline')
        self.check_time(now)

    def release(self):
        self.released=True
