"""Mock orchestration and a correlated NDJSON subprocess transport."""
import collections
import json
import math
import queue
import subprocess
import threading
import time


class MockBackend:
    mode = 'demo'

    def __init__(self, map_id, position=(0., 0., .4)):
        self.map_id = map_id; self.position = list(position)
        self.operation = None; self.out = []; self.commands = []; self.now = 0
        self.phase = 'ACTIVE'; self.profile = 'stand'; self.fault = None

    def send(self, command):
        self.commands.append(command.copy()); kind = command['kind']; task = command.get('task_id')
        if kind == 'emergency': self.operation = None; self.fault = 'demo emergency'; return
        if kind == 'prepare_walk':
            self.operation = dict(kind=kind, task_id=task, at=self.now+.15)
        elif kind == 'skill':
            self.operation = dict(kind=kind, task_id=task, at=self.now+.15, stage=0, profile=command['profile'])
        elif kind == 'navigate':
            self.operation = dict(kind=kind, task_id=task, goal=command['goal'], leg=command['leg'])
        elif kind == 'stop':
            remaining = 1.0 if self.operation and self.operation['kind']=='skill' else .15
            self.operation = dict(kind=kind, task_id=task, at=self.now+remaining)

    def poll(self, now):
        dt = max(0, min(now-self.now, .2)); self.now = now
        result = self.out; self.out = []; op = self.operation
        if op:
            kind = op['kind']; task = op['task_id']
            if kind == 'navigate':
                xyz = op['goal']['xyz']; distance = math.dist(xyz, self.position)
                if distance <= max(.03, dt*.8):
                    self.position = list(xyz); self.operation = None
                    result.append(dict(kind='nav_arrived', task_id=task, leg=op['leg']))
                else:
                    self.position = [a+(b-a)*dt*.8/distance for a, b in zip(self.position, xyz)]
            elif now >= op['at']:
                if kind == 'prepare_walk':
                    self.profile = 'walk_moe'; self.operation = None
                    result.append(dict(kind='walk_ready', task_id=task))
                elif kind == 'stop':
                    self.profile = 'stand'; self.operation = None
                    result.append(dict(kind='stand_ready', task_id=task))
                elif kind == 'skill':
                    op['stage'] += 1
                    if op['stage'] == 1:
                        self.profile = op['profile']; op['at'] = now+1
                        result.append(dict(kind='skill_started', task_id=task))
                    elif op['stage'] == 2:
                        op['at'] = now+.4; result.append(dict(kind='skill_recovering', task_id=task))
                    else:
                        self.profile = 'stand'; self.operation = None
                        result.append(dict(kind='skill_done', task_id=task))
        moving = bool(self.operation and self.operation['kind'] in ('navigate', 'skill'))
        result.insert(0, dict(kind='health', health=dict(runtime_ready=True, stationary=not moving,
            grounded=True, localized=True, map_aligned=True, map_id=self.map_id, position=self.position.copy(),
            profile=self.profile, phase=self.phase, fault=self.fault)))
        return result

    def close(self): pass


class ProcessBackend:
    mode = 'live'

    def __init__(self, argv):
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
            raise ValueError('桥接配置必须是 argv 字符串数组，不接受 shell 命令')
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, encoding='utf-8', bufsize=1, shell=False)
        self.events = queue.Queue(maxsize=128); self.write_lock = threading.Lock(); self.last_heartbeat = -float('inf')
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                event = json.loads(line)
                if isinstance(event, dict): self.events.put(event, timeout=.2)
            except (ValueError, queue.Full): continue

    def send(self, command):
        if self.process.poll() is not None: raise RuntimeError('ROS 桥进程已退出')
        with self.write_lock:
            self.process.stdin.write(json.dumps(command, allow_nan=False)+'\n'); self.process.stdin.flush()

    def poll(self, now):
        if self.process.poll() is not None: raise RuntimeError('ROS 桥进程已退出')
        if now-self.last_heartbeat > .1:
            self.send({'kind': 'heartbeat'}); self.last_heartbeat = now
        out = []
        while True:
            try: out.append(self.events.get_nowait())
            except queue.Empty: return out

    def close(self):
        try: self.send({'kind': 'emergency'})
        finally:
            if self.process.stdin: self.process.stdin.close()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.terminate()
