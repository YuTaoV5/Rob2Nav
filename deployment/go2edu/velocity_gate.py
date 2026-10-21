"""Noetic reference high-level velocity gate. Default: dry-run, no SDK import.

Not a certified emergency stop or an independent process-failure watchdog.
Only --execute opens robot DDS, and every start still requires manual arming.
"""
import argparse,json,math,threading,time

class Gate:
    limits=(.20,.10,.30)
    def __init__(self,robot_required=False):
        self.robot_required=robot_required;self.received={};self.command=(0.,0.,0.);self.output=[0.,0.,0.]
        self.armed=False;self.reason='disarmed';self.goal=None;self.pose=None;self.publishers=0;self.last_step=None;self.started=0.;self.plan_at=None
    def stop(self,reason):
        self.armed=False;self.goal=None;self.plan_at=None;self.output=[0.,0.,0.];self.reason=reason
    def feed(self,key,now,stamp_age=0.):
        if not math.isfinite(stamp_age) or not -.02<=stamp_age<(.5 if key=='cloud' else .25):
            self.received.pop(key,None)
            if self.armed:self.stop('invalid_'+key+'_stamp')
            return False
        self.received[key]=now;return True
    def health(self,now):
        for key,ttl in [('cmd',.30),('odom',.25),('cloud',.50),('deadman',.40)]+([('robot',.30)] if self.robot_required else []):
            age=now-self.received.get(key,-1e9)
            if not 0<=age<ttl:return 'stale_'+key
        if self.publishers!=1:return 'command_publisher_count'
        return ''
    def arm(self,now):
        why=self.health(now)
        if why:return False,why
        self.armed=True;self.started=now;self.goal=None;self.plan_at=None;self.output=[0.,0.,0.];self.reason='waiting_new_goal';return True,self.reason
    def set_command(self,command,now):
        if len(command)!=3 or not all(math.isfinite(x) for x in command):self.stop('nonfinite_command');self.received.pop('cmd',None);return
        self.command=tuple(command);self.feed('cmd',now)
    def set_pose(self,position,now,stamp_age):
        if not all(math.isfinite(x) for x in position):self.stop('nonfinite_pose');self.received.pop('odom',None);return
        if self.pose is not None and self.armed:
            dt=now-self.received.get('odom',now)
            if dt<.25 and math.dist(position,self.pose)>.40:self.stop('localization_jump');return
        if self.feed('odom',now,stamp_age):self.pose=tuple(position)
    def set_goal(self,position,now):
        if not all(math.isfinite(x) for x in position):self.stop('nonfinite_goal');return
        if self.armed and self.pose is not None:
            self.goal=tuple(position);self.plan_at=None;self.output=[0.,0.,0.];self.reason='waiting_new_plan'
    def set_plan(self,now):
        if self.armed and self.goal is not None and self.plan_at is None:self.plan_at=now
    def step(self,now):
        dt=min(.1,max(0.,now-(self.last_step if self.last_step is not None else now)));self.last_step=now
        if not self.armed:return (0.,0.,0.)
        why=self.health(now)
        if why:self.stop(why);return (0.,0.,0.)
        if now-self.started>120:self.stop('task_timeout');return (0.,0.,0.)
        if self.goal is None:return (0.,0.,0.)
        if self.pose is not None and math.dist(self.pose[:2],self.goal[:2])<.35:self.stop('arrived');return (0.,0.,0.)
        if self.plan_at is None or self.received.get('cmd',-1e9)<self.plan_at:return (0.,0.,0.)
        self.reason='active'
        desired=[max(-limit,min(limit,x)) for x,limit in zip(self.command,self.limits)]
        # Limit combined translation, not only each axis independently.
        norm=math.hypot(desired[0],desired[1])
        if norm>.20:desired[0]*=.20/norm;desired[1]*=.20/norm
        for i,rate in enumerate((.20,.20,.40)):
            delta=rate*dt;self.output[i]+=max(-delta,min(delta,desired[i]-self.output[i]))
        return tuple(self.output)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--interface');a,_=p.parse_known_args()
    if a.execute and a.replay:p.error('--replay cannot be combined with --execute')
    if a.execute and not a.interface:p.error('--execute requires --interface, e.g. enp2s0')
    import rospy
    from geometry_msgs.msg import Twist,PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool,String
    from std_srvs.srv import SetBool,SetBoolResponse
    from scan_planner.msg import Bspline
    rospy.init_node('rob2nav_velocity_gate',disable_signals=False)
    if rospy.get_param('/use_sim_time',False) and not a.replay:raise RuntimeError('Clock replay requires --replay; hardware execution refuses use_sim_time')
    gate=Gate(a.execute);lock=threading.RLock();sdk=None;ever_armed=False;goal_stamp=[None]
    if a.execute:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(0,a.interface);sdk=SportClient();sdk.SetTimeout(.1);sdk.Init()
        def robot(m):
            with lock:
                rpy=m.imu_state.rpy
                if not all(math.isfinite(x) for x in rpy) or max(abs(rpy[0]),abs(rpy[1]))>.6:gate.stop('robot_posture');gate.received.pop('robot',None);return
                gate.feed('robot',time.monotonic())
        state_sub=ChannelSubscriber('rt/sportmodestate',SportModeState_);state_sub.Init(robot,10)
    status=rospy.Publisher('/rob2nav/gate_status',String,queue_size=2)
    def command(m):
        with lock:gate.set_command((m.linear.x,m.linear.y,m.angular.z),time.monotonic())
    cmd_sub=rospy.Subscriber('/rob2nav/cmd_vel_raw',Twist,command,queue_size=1,tcp_nodelay=True)
    def pose(m):
        with lock:
            if m.header.frame_id!='nav_odom':gate.stop('wrong_odom_frame');gate.received.pop('odom',None);return
            p=m.pose.pose.position;q=m.pose.pose.orientation
            values=(q.x,q.y,q.z,q.w)
            if not all(math.isfinite(x) for x in values) or abs(sum(x*x for x in values)-1)>.02:gate.stop('invalid_orientation');gate.received.pop('odom',None);return
            gate.set_pose((p.x,p.y,p.z),time.monotonic(),rospy.Time.now().to_sec()-m.header.stamp.to_sec())
    def cloud(m):
        with lock:
            if m.header.frame_id!='nav_odom' or m.width*m.height<20:
                gate.received.pop('cloud',None)
                if gate.armed:gate.stop('invalid_cloud')
                return
            gate.feed('cloud',time.monotonic(),rospy.Time.now().to_sec()-m.header.stamp.to_sec())
    def deadman(m):
        with lock:
            if m.data:gate.feed('deadman',time.monotonic())
            else:gate.stop('operator_release');gate.received.pop('deadman',None)
    def goal(m):
        with lock:
            age=rospy.Time.now().to_sec()-m.header.stamp.to_sec()
            if m.header.frame_id!='nav_odom' or not -.02<=age<.5:
                if gate.armed:gate.stop('invalid_goal_frame_or_stamp')
                return
            p=m.pose.position;gate.set_goal((p.x,p.y,p.z),time.monotonic());goal_stamp[0]=m.header.stamp.to_sec()
    def plan(m):
        with lock:
            stamp=m.start_time.to_sec()
            if goal_stamp[0] is not None and stamp>=goal_stamp[0] and -.1<=rospy.Time.now().to_sec()-stamp<.5:
                gate.set_plan(time.monotonic())
    def arm(m):
        nonlocal ever_armed
        with lock:
            gate.publishers=cmd_sub.get_num_connections()
            if not m.data:gate.stop('operator_disarm');return SetBoolResponse(True,'disarmed')
            ok,why=gate.arm(time.monotonic());ever_armed|=ok;return SetBoolResponse(ok,why)
    rospy.Subscriber('/rob2nav/body_pose',Odometry,pose,queue_size=2)
    rospy.Subscriber('/rob2nav/cloud_world',PointCloud2,cloud,queue_size=1)
    rospy.Subscriber('/rob2nav/operator_heartbeat',Bool,deadman,queue_size=1)
    rospy.Subscriber('/move_base_simple/goal',PoseStamped,goal,queue_size=1)
    rospy.Subscriber('/planning/bspline',Bspline,plan,queue_size=1)
    rospy.Service('/rob2nav/arm',SetBool,arm)
    try:
        while not rospy.is_shutdown():
            now=time.monotonic()
            with lock:
                now=time.monotonic()
                gate.publishers=cmd_sub.get_num_connections();v=gate.step(now)
                data=dict(execute=a.execute,armed=gate.armed,reason=gate.reason,velocity=list(v),input_ages={k:now-t for k,t in gate.received.items()})
            if sdk is not None and ever_armed:
                try:
                    rc=sdk.Move(*v) if data['armed'] and gate.goal is not None and any(abs(x)>1e-5 for x in v) else sdk.StopMove()
                    if rc!=0:
                        with lock:gate.stop('sdk_error_'+str(rc))
                except Exception as e:
                    with lock:gate.stop('sdk_exception')
                    rospy.logerr_throttle(1,str(e))
            status.publish(String(data=json.dumps(data)));time.sleep(.05)
    finally:
        if sdk is not None and ever_armed:sdk.StopMove()

if __name__=='__main__':main()
