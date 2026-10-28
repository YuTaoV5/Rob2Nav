"""Testable live SCAN/Rob2Nav routing; no ROS imports or motor publishers."""
import math
import time


class RobotRouter:
    def __init__(self, runtime_send, publish_goal, emit, map_id, frame='nav_odom', aligned=False, clock=time.monotonic):
        self.runtime_send, self.publish_goal, self.emit = runtime_send, publish_goal, emit
        self.map_id, self.frame, self.aligned, self.clock = map_id, frame, aligned, clock
        self.runtime = {}; self.runtime_at = -float('inf'); self.odom = None; self.cloud_at = -float('inf')
        self.parent_at = clock(); self.op = None; self.requested = None; self.last_runtime_heartbeat = -float('inf')
        self.plan_at = -float('inf'); self.goal_ros_stamp = float('inf'); self.cmd_at = -float('inf'); self.cmd = [0,0,0]
        self.arrival_since = None; self.last_emit = -float('inf'); self.parent_failed = False

    def on_runtime(self, status):
        self.runtime = status; self.runtime_at = self.clock()

    def on_odom(self, xyz, yaw, velocity, frame, age):
        if frame != self.frame or not 0 <= age < .5 or not all(math.isfinite(v) for v in xyz+[yaw]+velocity): return
        self.odom = dict(position=xyz, yaw=yaw, velocity=velocity, at=self.clock())

    def on_cloud(self, frame, age):
        if frame == self.frame and 0 <= age < .5: self.cloud_at = self.clock()

    def on_plan(self, start_stamp):
        if self.op and self.op['kind']=='navigate' and math.isfinite(start_stamp) and start_stamp >= self.goal_ros_stamp-.001:
            self.plan_at = self.clock()

    def on_velocity(self, values):
        if len(values)==3 and all(math.isfinite(v) for v in values): self.cmd, self.cmd_at = values, self.clock()

    def health(self):
        now = self.clock(); r = self.runtime
        live = now-self.runtime_at < .5
        localized = bool(self.odom and now-self.odom['at'] < .5 and now-self.cloud_at < .5)
        return dict(runtime_ready=live and r.get('phase') not in (None, 'OFF', 'FAULT'),
            stationary=live and bool(r.get('stationary')), grounded=live and bool(r.get('grounded')),
            profile=r.get('active'), phase=r.get('phase'), fault=r.get('fault') or r.get('stage_error'),
            localized=localized, map_aligned=self.aligned, map_id=self.map_id,
            frame_id=self.frame, position=self.odom['position'] if localized else None)

    def _event(self, kind, **kwargs):
        self.emit(dict(kind=kind, task_id=self.op.get('task_id') if self.op else None, **kwargs))

    def command(self, command):
        kind = command.get('kind'); now = self.clock()
        if kind == 'heartbeat': self.parent_at = now; return
        if kind == 'emergency':
            self.runtime_send({'stop': True}); self.op = None; return
        if kind not in ('prepare_walk', 'navigate', 'skill', 'stop'): raise ValueError('未知桥接指令')
        if not isinstance(command.get('task_id'), str): raise ValueError('任务 ID 缺失')
        previous = self.op
        self.op = dict(command, started=False, stand_queued=False, recover_emitted=False)
        self.arrival_since = None
        if kind == 'navigate':
            goal = command['goal']
            if goal['map_id'] != self.map_id or goal['frame_id'] != self.frame or not self.aligned:
                raise ValueError('地图/导航 frame 不匹配或尚未配准')
            if not self.health()['localized']: raise ValueError('定位数据缺失或过期')
            self.goal_ros_stamp = self.publish_goal(goal, self.odom['position'])
            self.plan_at = self.cmd_at = -float('inf')
        elif kind == 'stop':
            self.runtime_send({'velocity': [0,0,0]})
            if previous and previous['kind']=='skill' and not previous.get('started'):
                # Upstream cannot cancel an untriggered staged atomic transaction.
                # Never trigger it merely to obtain a normal exit.
                self.runtime_send({'stop': True})
                self._event('failed', reason='动作准备已取消，底层锁存停止；检查后重新启动运行时')
            elif previous and previous.get('stand_queued'):
                self.op['stand_queued'] = True

    def _select(self, profile):
        r = self.runtime
        if r.get('active') == profile and r.get('phase') in ('ACTIVE', 'ARMED'):
            self.requested = None; return True
        if self.requested == profile or r.get('pending') or r.get('loading'): return False
        if r.get('phase') not in ('READY', 'ACTIVE'): return False
        self.runtime_send({'switch': profile}); self.requested = profile; return False

    def tick(self, publisher_count=1):
        now = self.clock(); h = self.health()
        if now-self.parent_at > 1:
            if not self.parent_failed: self.runtime_send({'stop': True}); self.parent_failed = True
            return
        if now-self.last_runtime_heartbeat >= .1:
            self.runtime_send({'heartbeat': True}); self.last_runtime_heartbeat = now
        if now-self.last_emit >= .05:
            self.emit(dict(kind='health', health=h)); self.last_emit = now
        if not self.op: return
        if not h['runtime_ready'] or h['fault']:
            self._event('failed', reason='运行时不可用或技能资格/切换失败：'+str(h['fault']))
            self.runtime_send({'stop': True}); self.op = None; return
        op = self.op; kind = op['kind']; r = self.runtime
        if kind == 'prepare_walk':
            if self._select('walk_moe') and r.get('phase') == 'ACTIVE':
                self._event('walk_ready'); self.op = None
        elif kind == 'skill':
            if not op['started']:
                if self._select(op['profile']):
                    if r['phase'] == 'ARMED' and not op.get('triggered'):
                        self.runtime_send({'trigger': True}); op['triggered'] = True
                    elif r['phase'] == 'ACTIVE' and op.get('triggered'):
                        op['started'] = True; self._event('skill_started')
            else:
                if not op['stand_queued']:
                    if self._select('stand'): op['stand_queued'] = True
                    elif self.requested == 'stand' or r.get('pending') == 'stand': op['stand_queued'] = True
                if r.get('phase') not in ('ACTIVE', 'ARMED') and not op['recover_emitted']:
                    self._event('skill_recovering'); op['recover_emitted'] = True
                if r.get('active') == 'stand' and r.get('phase') == 'ACTIVE' and h['stationary'] and h['grounded']:
                    self._event('skill_done'); self.op = None
        elif kind == 'stop':
            self.runtime_send({'velocity': [0,0,0]})
            self._select('stand')
            if r.get('active') == 'stand' and r.get('phase') == 'ACTIVE' and h['stationary'] and h['grounded']:
                self._event('stand_ready'); self.op = None
        elif kind == 'navigate':
            if not h['localized'] or publisher_count != 1:
                self.runtime_send({'velocity': [0,0,0]}); self._event('failed', reason='定位失效或速度发布者不唯一')
                self.op = None; return
            if r.get('active') != 'walk_moe' or r.get('phase') != 'ACTIVE':
                self.runtime_send({'velocity': [0,0,0]}); return
            goal = op['goal']; position = self.odom['position']
            horizontal = math.hypot(position[0]-goal['xyz'][0], position[1]-goal['xyz'][1])
            vertical = abs(position[2]-goal['xyz'][2])
            yaw_error = math.atan2(math.sin(goal['yaw']-self.odom['yaw']), math.cos(goal['yaw']-self.odom['yaw']))
            if horizontal < .18 and vertical < .12:
                velocity = [0,0,max(-.3,min(.3,yaw_error))] if abs(yaw_error) > .15 else [0,0,0]
                settled = abs(yaw_error) < .15 and math.hypot(*self.odom['velocity'][:2]) < .08 and h['grounded']
                if settled:
                    if self.arrival_since is None: self.arrival_since = now
                    if now-self.arrival_since >= .5:
                        self.runtime_send({'velocity': [0,0,0]}); self._event('nav_arrived', leg=op['leg']); self.op = None; return
                else: self.arrival_since = None
            elif now-self.cmd_at < .25 and self.cmd_at >= self.plan_at and self.plan_at > -float('inf'):
                vx, vy, wz = self.cmd
                vx, vy = max(-.2,min(.2,vx)), max(-.1,min(.1,vy))
                scale = max(1, math.hypot(vx,vy)/.2)
                velocity = [vx/scale,vy/scale,max(-.3,min(.3,wz))]
            else: velocity = [0,0,0]
            self.runtime_send({'velocity': velocity})
