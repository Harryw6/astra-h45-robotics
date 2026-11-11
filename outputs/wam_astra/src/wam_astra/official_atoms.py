"""Small SDK adapter, not a new navigation controller.

Go2: unitree_sdk2(py) ObstaclesAvoidClient.MoveToIncrementPosition.
P450: Prometheus UAVCommand.Move / XYZ_POS_BODY.
The caller owns controller selection and completion feedback. A successful
no-reply SDK call only proves submission, never physical arrival.
"""
import json
import math
import threading


class OfficialAtoms:
    def __init__(self, go2, publish_body_position, *, last_command_id):
        # Prometheus accepts a new BODY target only when its uint32 Command_ID
        # exceeds the previous controller command. The previous writer must
        # supply its sequence at ownership handoff; process restart is not zero.
        if type(last_command_id) is not int or not 0<=last_command_id<2**32:
            raise ValueError('last_command_id must be the previous controller uint32 sequence')
        self.go2=go2
        self.publish_body_position=publish_body_position
        self.receipts={}
        self.lock=threading.Lock()
        self.command_id=last_command_id

    def dispatch(self, action_id, action):
        if not isinstance(action_id,str) or not action_id:
            raise ValueError('A unique action ID is required')
        decision=action.get('decision')
        if decision not in ('move','observe','hold','handoff_pi'):
            raise ValueError('Unknown decision')
        vector=None
        if decision=='move':
            distance=action.get('distance_m')
            if isinstance(distance,bool) or not isinstance(distance,(int,float)) or not math.isfinite(distance) or distance<=0:
                raise ValueError('Expected positive finite metres')
            axes={'forward':(1,0,0),'backward':(-1,0,0),'left':(0,1,0),
                  'right':(0,-1,0),'up':(0,0,1),'down':(0,0,-1)}
            if action.get('robot') not in ('go2','uav') or action.get('direction') not in axes:
                raise ValueError('Unknown robot or direction')
            vector=tuple(float(distance*v) for v in axes[action['direction']])
            if action['robot']=='go2' and vector[2]:
                raise ValueError('Go2 cannot translate vertically')
        signature=json.dumps(action,sort_keys=True,allow_nan=False)
        with self.lock:
            if action_id in self.receipts:
                prior_signature,receipt=self.receipts[action_id]
                if prior_signature!=signature:raise ValueError('Action ID reused with different contents')
                return dict(receipt)
            if vector is not None and action['robot']=='uav' and self.command_id==2**32-1:
                raise ValueError('Prometheus Command_ID sequence exhausted; explicit controller handoff required')
            receipt=dict(action_id=action_id,status='no_command',arrival_confirmed=False)
            self.receipts[action_id]=(signature,receipt)
            if vector is None:return dict(receipt)
            # Mark before the call: an interrupted no-reply request must not be
            # automatically resent as a second relative movement.
            receipt['status']='unknown'
            if action['robot']=='go2':
                code=self.go2.MoveToIncrementPosition(vector[0],vector[1],0.)
                receipt['sdk_rc']=code
                receipt['status']='submitted' if code==0 else 'rejected'
            else:
                self.command_id+=1
                receipt['command_id']=self.command_id
                self.publish_body_position(command_id=self.command_id,body_xyz=vector)
                receipt.update(status='submitted',command_id=self.command_id)
            return dict(receipt)


class PrometheusBodyPublisher:
    """ROS1 official message writer; no implicit mode switch, takeoff or land.

    Instantiate on the aircraft after choosing the controller's actual command
    topic. Do not run beside another writer. Keep one instance per ROS session.
    """
    def __init__(self, topic='/uav1/prometheus/command'):
        import rospy
        from prometheus_msgs.msg import UAVCommand
        self.rospy=rospy
        self.message_type=UAVCommand
        self.publisher=rospy.Publisher(topic,UAVCommand,queue_size=1)

    def __call__(self, *, command_id, body_xyz):
        if not self.publisher.get_num_connections():
            raise RuntimeError('No controller subscribes to the selected command topic')
        msg=self.message_type()
        msg.header.stamp=self.rospy.Time.now()
        msg.header.frame_id='BODY'
        msg.Agent_CMD=msg.Move
        msg.Move_mode=msg.XYZ_POS_BODY
        msg.position_ref=list(body_xyz)
        msg.yaw_ref=0.0
        msg.Yaw_Rate_Mode=False
        msg.Command_ID=command_id
        self.publisher.publish(msg)
