"""ROS1 sidecar. Only this process imports rospy; no DDS writer here."""
import argparse
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from .router import RobotRouter


def main():
    import rospy
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import PointCloud2
    from geometry_msgs.msg import PoseStamped, Twist
    from scan_planner.msg import Bspline
    p = argparse.ArgumentParser()
    p.add_argument('--runtime-command', type=Path, required=True)
    p.add_argument('--map-id', required=True); p.add_argument('--frame', default='nav_odom')
    p.add_argument('--map-aligned', action='store_true')
    p.add_argument('--scan-mode', type=int, choices=(1,3), default=3)
    args = p.parse_args(rospy.myargv()[1:])
    rospy.init_node('voice_task_bridge')
    if rospy.get_param('/use_sim_time', False): raise RuntimeError('实机桥拒绝 ROS 回放时钟')
    configured_mode = rospy.get_param('/scan_planner_node/fsm/navi_mode', None)
    if configured_mode != args.scan_mode: raise RuntimeError('SCAN navi_mode 与桥配置不一致')
    height = float(rospy.get_param('/scan_planner_node/grid_map/body_height'))
    if not .15 < height < .7: raise RuntimeError('需要真实机身参考高度')
    argv = json.loads(args.runtime_command.read_text(encoding='utf-8'))
    if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv): raise ValueError('运行时命令必须为 argv 数组')
    process = subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,text=True,encoding='utf-8',bufsize=1,shell=False)
    inputs = queue.Queue(maxsize=128); runtime_events = queue.Queue(maxsize=128); output = queue.Queue(maxsize=128)
    write_lock = threading.Lock(); parent_closed = threading.Event()
    def runtime_send(command):
        with write_lock:
            if process.poll() is not None: raise RuntimeError('MoE 运行时已退出')
            process.stdin.write(json.dumps(command)+'\n'); process.stdin.flush()
    def emit(event): output.put_nowait(event)
    def writer():
        while True:
            value = output.get()
            if value is None: return
            print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    def reader(stream, target, parent=False):
        try:
            for line in stream:
                try:
                    value=json.loads(line)
                    if isinstance(value,dict): target.put(value,timeout=.1)
                except (ValueError,queue.Full): continue
        finally:
            if parent: parent_closed.set()
    threading.Thread(target=writer,daemon=True).start()
    threading.Thread(target=reader,args=(sys.stdin,inputs,True),daemon=True).start()
    threading.Thread(target=reader,args=(process.stdout,runtime_events),daemon=True).start()
    goal_pub = rospy.Publisher('/move_base_simple/goal',PoseStamped,queue_size=1)
    path_pub = rospy.Publisher('/initial_path',RosPath,queue_size=1)
    def publish_goal(goal, position):
        stamp=rospy.Time.now()
        def pose(xyz):
            msg=PoseStamped();msg.header.frame_id=args.frame;msg.header.stamp=stamp
            msg.pose.position.x,msg.pose.position.y,msg.pose.position.z=xyz
            msg.pose.orientation.z=math.sin(goal['yaw']/2);msg.pose.orientation.w=math.cos(goal['yaw']/2)
            return msg
        if args.scan_mode == 1:
            if abs(goal['xyz'][2]-position[2])>.12: raise ValueError('SCAN 模式1不能执行跨高度路线；使用模式3')
            goal_pub.publish(pose(goal['xyz']))
        else:
            path=RosPath();path.header.frame_id=args.frame;path.header.stamp=stamp
            path.poses=[pose([*position[:2],position[2]-height]),pose([*goal['xyz'][:2],goal['xyz'][2]-height])]
            path_pub.publish(path)
        return stamp.to_sec()
    router=RobotRouter(runtime_send,publish_goal,emit,args.map_id,args.frame,args.map_aligned)
    sensor_queue=queue.Queue(maxsize=256)
    def enqueue(item):
        try:sensor_queue.put_nowait(item)
        except queue.Full:pass
    def odom(msg):
        q=msg.pose.pose.orientation;p=msg.pose.pose.position;v=msg.twist.twist.linear
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        enqueue(('odom',([p.x,p.y,p.z],yaw,[v.x,v.y,v.z],msg.header.frame_id,rospy.Time.now().to_sec()-msg.header.stamp.to_sec())))
    def cloud(msg):enqueue(('cloud',(msg.header.frame_id,rospy.Time.now().to_sec()-msg.header.stamp.to_sec())))
    def plan(msg):enqueue(('plan',(msg.start_time.to_sec(),)))
    def velocity(msg):enqueue(('velocity',([msg.linear.x,msg.linear.y,msg.angular.z],)))
    rospy.Subscriber('/rob2nav/body_pose',Odometry,odom,queue_size=5)
    rospy.Subscriber('/rob2nav/cloud_world',PointCloud2,cloud,queue_size=1)
    rospy.Subscriber('/planning/bspline',Bspline,plan,queue_size=2)
    cmd_sub=rospy.Subscriber('/rob2nav/cmd_vel_raw',Twist,velocity,queue_size=1,tcp_nodelay=True)
    try:
        while not rospy.is_shutdown() and not parent_closed.is_set():
            while not runtime_events.empty(): router.on_runtime(runtime_events.get_nowait())
            while not sensor_queue.empty():
                kind,values=sensor_queue.get_nowait();getattr(router,'on_'+kind)(*values)
            while not inputs.empty():
                command=inputs.get_nowait()
                try:router.command(command)
                except (ValueError,RuntimeError) as exc:emit(dict(kind='failed',task_id=command.get('task_id'),reason=str(exc)))
            router.tick(publisher_count=cmd_sub.get_num_connections())
            time.sleep(.02)
    finally:
        try:runtime_send({'stop':True})
        finally:
            process.stdin.close()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.terminate()


if __name__=='__main__':main()
