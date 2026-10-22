"""CPU simulator side of the isolated local ROS bridge; stale replies stop motion."""
import json
import socket
import time
import uuid
import numpy as np


class ScanClient:
    def __init__(self):
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.sock.bind(('127.0.0.1',0));self.sock.setblocking(False)
        self.session=uuid.uuid4().hex;self.telemetry={};self.received=0.
    def reset(self):self.session=uuid.uuid4().hex;self.telemetry={};self.received=0.
    def act(self,env,goal,active):
        from .tasks import TASK_SCENES
        task_scene=env.scene in TASK_SCENES
        points=env.points
        if task_scene and len(env.lidar_free_dirs):
            # True no-return beams, not dropped measurements. SCAN truncates
            # beyond max_ray_length as free space instead of occupied endpoints.
            free=np.asarray(env.cfg['lidar_position'])+env.lidar_free_dirs*(env.cfg['max_range']+.2)
            points=np.concatenate([points,free])
        if len(points)>450:points=points[np.linspace(0,len(points)-1,450,dtype=int)]
        rot=env.rotation
        world=env.data.xpos[env.base_id]+points@rot.T
        sensor=env.data.xpos[env.base_id]+rot@np.asarray(env.cfg['lidar_position'])
        payload=dict(session=self.session,dynamic_task=bool(getattr(env,'people',[])),low_step_navigation=bool(getattr(env,'step_navigation',False)),position=env.data.qpos[:3].tolist(),quaternion=env.data.qpos[3:7].tolist(),
            velocity=env.data.qvel[:3].tolist(),sensor_position=sensor.tolist(),points=world.tolist(),goal=goal.tolist(),goal_height=(.35+env.ground(float(goal[0]))) if task_scene else .3,active=bool(active),goal_epoch=int(env.data.time//4) if task_scene else 0)
        self.sock.sendto(json.dumps(payload).encode(),('127.0.0.1',8771))
        while True:
            try:
                raw,peer=self.sock.recvfrom(65507)
                if peer!=('127.0.0.1',8771):continue
                packet=json.loads(raw)
                if packet.get('session')!=self.session:continue
                self.telemetry=packet;self.received=time.monotonic()
            except BlockingIOError:break
        if time.monotonic()-self.received>.3 or not active or not self.telemetry.get('ready'):return np.zeros(3)
        limits=np.array([env.cfg['max_vx'],env.cfg['max_vy'],env.cfg['max_wz']])
        return np.clip(np.asarray(self.telemetry['command'])/limits,-1,1)
    def close(self):self.sock.close()
