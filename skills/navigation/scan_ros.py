"""Pinned SCAN ROS2 planner <-> loopback MuJoCo sensor/velocity bridge.

No native kinematic simulator, ground-truth obstacle list or robot SDK is used.
Run with the sourced ROS Jazzy / SCAN workspace system Python.
"""
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped,Twist
from sensor_msgs.msg import PointCloud2,PointField
from scan_planner_msgs.msg import Bspline

SOURCE=Path('/root/gpufree-data/scan-reproduction/upstream')
PIN='d0b921c9b05a6d291d144d60882b2e0e88d2c0e0'


class Bridge(Node):
    def __init__(self):
        super().__init__('rob2nav_scan_mujoco_bridge')
        if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!=PIN:raise RuntimeError('SCAN source pin differs')
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.sock.bind(('127.0.0.1',8771));self.sock.setblocking(False)
        self.odom=self.create_publisher(Odometry,'/quad_0/body_pose',10)
        self.sensor=self.create_publisher(Odometry,'/quad_0/lidar_pose',10)
        self.cloud=self.create_publisher(PointCloud2,'/quad_0/cloud',10)
        self.goal=self.create_publisher(PoseStamped,'/move_base_simple/goal',10)
        self.create_subscription(Twist,'/quad_0/cmd_vel',self.command,10)
        self.create_subscription(Bspline,'/planning/bspline',self.plan,10)
        self.cmd=[0.,0.,0.];self.cmd_at=0.;self.plans=0;self.command_count=0
        self.session=None;self.children=[];self.log=None;self.started=0.;self.goal_key=None;self.last_packet=0.
        self.goal_at=0.;self.goal_plan_baseline=0;self.goal_attempts=0
        self.create_timer(.01,self.receive)
    def command(self,m):
        self.cmd=[m.linear.x,m.linear.y,m.angular.z];self.cmd_at=time.monotonic();self.command_count+=1
    def plan(self,m):self.plans+=1
    def stop_children(self):
        for p in self.children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        self.children=[]
        if self.log:self.log.close();self.log=None
    def restart(self,session):
        self.stop_children();self.session=session;self.cmd=[0.,0.,0.];self.cmd_at=0.;self.plans=0;self.command_count=0;self.goal_key=None
        self.goal_at=0.;self.goal_plan_baseline=0;self.goal_attempts=0
        folder=Path('/root/gpufree-data/Rob2Nav/outputs/scan_web');folder.mkdir(parents=True,exist_ok=True)
        self.log=(folder/'planner.log').open('ab',buffering=0)
        config=SOURCE/'install/scan_planner/share/scan_planner/config'
        binary=SOURCE/'install/scan_planner/lib/scan_planner'
        commands=[
            [str(binary/'scan_planner_node'),'--ros-args','--params-file',str(config/'planner.yaml'),'-r','body_pose:=/quad_0/body_pose','-r','sensor_pose:=/quad_0/lidar_pose','-r','cloud:=/quad_0/cloud'],
            [str(binary/'closed_loop_controller'),'--ros-args','--params-file',str(config/'controllers.yaml'),'-r','body_pose:=/quad_0/body_pose','-r','cmd_vel:=/quad_0/cmd_vel']]
        self.children=[subprocess.Popen(c,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True) for c in commands]
        self.started=time.monotonic()
    def receive(self):
        packet=None
        while True:
            try:raw,peer=self.sock.recvfrom(65507);packet=(json.loads(raw),peer)
            except BlockingIOError:break
            except (ValueError,OSError):return
        if packet is None:
            if self.children and time.monotonic()-self.last_packet>10:self.stop_children();self.session=None
            return
        d,peer=packet;self.last_packet=time.monotonic()
        if d['session']!=self.session:self.restart(d['session'])
        stamp=self.get_clock().now().to_msg()
        odom=Odometry();odom.header.stamp=stamp;odom.header.frame_id='world';odom.child_frame_id='base'
        p,q=d['position'],d['quaternion']
        odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z=p
        odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z=q
        odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z=d['velocity']
        self.odom.publish(odom)
        sensor=Odometry();sensor.header=odom.header;sensor.pose.pose.orientation=odom.pose.pose.orientation
        sensor.pose.pose.position.x,sensor.pose.pose.position.y,sensor.pose.pose.position.z=d['sensor_position'];self.sensor.publish(sensor)
        points=d['points'];cloud=PointCloud2();cloud.header=odom.header;cloud.height=1;cloud.width=len(points)
        cloud.fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z'])]
        cloud.point_step=12;cloud.row_step=12*len(points);cloud.is_dense=True
        cloud.data=b''.join(struct.pack('<fff',*p) for p in points);self.cloud.publish(cloud)
        key=tuple(d['goal'])
        changed=key!=self.goal_key
        retry=self.plans<=self.goal_plan_baseline and time.monotonic()-self.goal_at>2
        if d['active'] and (changed or retry) and time.monotonic()-self.started>2 and self.goal.get_subscription_count()>0:
            goal=PoseStamped();goal.header=odom.header
            goal.pose.position.x,goal.pose.position.y=d['goal'];goal.pose.position.z=.3;goal.pose.orientation.w=1.
            if changed:self.goal_plan_baseline=self.plans;self.goal_attempts=0
            self.goal.publish(goal);self.goal_key=key;self.goal_at=time.monotonic();self.goal_attempts+=1
        age=time.monotonic()-self.cmd_at
        reply=dict(session=self.session,command=self.cmd if age<.3 and d['active'] else [0.,0.,0.],command_age=age if self.cmd_at else None,
            plans=self.plans,commands=self.command_count,source_commit=PIN,ready=all(p.poll() is None for p in self.children),
            sensor_points=len(points),goal_published=self.goal_key is not None,goal_attempts=self.goal_attempts,
            goal_subscribers=self.goal.get_subscription_count(),goal_acknowledged=self.plans>self.goal_plan_baseline)
        self.sock.sendto(json.dumps(reply).encode(),peer)


def main():
    def terminate(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    rclpy.init();node=Bridge()
    try:rclpy.spin(node)
    finally:node.stop_children();node.sock.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
