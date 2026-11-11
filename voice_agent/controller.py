"""Correlated event FSM; motor ownership remains in the robot runtime."""
import collections
import copy
import math
import threading
import time
import uuid
from .intents import parse, SKILLS


class Controller:
    def __init__(self, places, backend, clock=time.monotonic, audit=None):
        self.places, self.backend, self.clock, self.audit = places, backend, clock, audit
        self.lock = threading.RLock()
        self.state = 'DISABLED'; self.generation = 0; self.task = None
        self.health = {}; self.health_at = -float('inf')
        self.events = collections.deque(maxlen=100); self.seen = collections.OrderedDict()
        self.reply = '控制未启用。'; self.last_result = None
        self._emit('boot', self.reply)

    def _emit(self, kind, message, **extra):
        self.reply = message
        event = dict(kind=kind, message=message, state=self.state, generation=self.generation,
                     task_id=self.task['id'] if self.task else None, at=self.clock(), **extra)
        self.events.append(event)
        if self.audit: self.audit(event)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(dict(state=self.state, generation=self.generation, task=self.task,
                health=self.health, health_fresh=self.clock()-self.health_at < 1.5,
                reply=self.reply, last_result=self.last_result, events=list(self.events),
                mode=self.backend.mode, skills=SKILLS))

    def _send(self, kind, **payload):
        self.backend.send(dict(kind=kind, task_id=self.task['id'] if self.task else None, **payload))

    def _healthy(self, navigation=False):
        if self.clock()-self.health_at > 1.5 or not self.health.get('runtime_ready'):
            raise ValueError('机器人运行时状态缺失或过期')
        if not self.health.get('grounded') or not self.health.get('stationary'):
            raise ValueError('等待机器人稳定站立后再开始任务')
        if navigation:
            if not self.health.get('localized') or not self.health.get('map_aligned'):
                raise ValueError('定位或全局地图配准尚未确认')
            if self.health.get('map_id') != self.places.data['map_id']:
                raise ValueError('当前定位地图与标注地图不一致')

    def submit(self, text, command_id=None, generation=None, final=True):
        with self.lock:
            intent = parse(text)
            urgent = intent['kind'] in ('stop', 'emergency', 'disable')
            command_id = command_id or uuid.uuid4().hex
            if not isinstance(command_id, str) or len(command_id) > 128 or type(final) is not bool:
                return {'accepted': False, 'message': '非法命令元数据'}
            if command_id in self.seen: return copy.deepcopy(self.seen[command_id])
            if not final: return {'accepted': False, 'message': '仅接受完整语句，未执行部分识别结果'}
            if not urgent and generation is not None and generation != self.generation:
                return {'accepted': False, 'message': '状态已改变，已丢弃过期语音结果'}
            accepted = False
            try:
                kind = intent['kind']
                if kind == 'emergency':
                    self._fault('收到急停请求'); accepted = True
                elif kind in ('stop', 'disable'):
                    stay_disabled = kind == 'disable' or self.state == 'DISABLED'
                    self.generation += 1
                    if self.state == 'FAULT': self._emit('stopped', '故障已锁存，请先检查并重启底层运行时。')
                    else:
                        self.state = 'STOPPING'
                        self.task = dict(id=uuid.uuid4().hex, kind='stop', disable=stay_disabled, deadline=self.clock()+25)
                        self._send('stop', disable=kind=='disable')
                        self._emit('stopping', '正在停止并等待稳定站立。')
                    accepted = True
                elif kind in ('status', 'places', 'skills'):
                    if kind == 'places': message = '已标注地点：' + '、'.join(p['name'] for p in self.places.data['places'])
                    elif kind == 'skills': message = '可调度技能：后空翻、侧空翻、倒立；实机执行以底层资格为准。'
                    else: message = '当前状态：' + self.state
                    self._emit('answer', message); accepted = True
                elif kind == 'enable':
                    if self.state not in ('DISABLED', 'IDLE', 'FAULT'): raise ValueError('请先停止当前任务')
                    self._healthy()
                    if self.health.get('fault'): raise ValueError('底层仍有故障，无法启用')
                    self.generation += 1; self.task = None; self.state = 'IDLE'
                    self._emit('enabled', '控制已启用，可以说地点或技能。'); accepted = True
                elif kind in ('navigate', 'skill'):
                    if self.state != 'IDLE': raise ValueError('当前未就绪，请先启用控制或等待当前任务结束')
                    self._healthy(navigation=kind=='navigate')
                    if kind == 'navigate':
                        matches = self.places.resolve(intent['destination'])
                        if not matches: raise ValueError('没有找到这个地点，请先在地图上标注名称或别名')
                        if len(matches) > 1: raise ValueError('地点有歧义，请指定：' + '、'.join(p['name'] for p in matches))
                        route = self.places.route(matches[0], self.health.get('position'), require_graph=self.backend.mode=='live')
                        self.task = dict(id=uuid.uuid4().hex, kind=kind, destination=matches[0]['name'],
                            route=route, leg=0, deadline=self.clock()+180)
                        self.state = 'NAV_PREPARING'; self.generation += 1
                        self._send('prepare_walk')
                        self._emit('navigation_preparing', '准备前往' + matches[0]['name'] + '。')
                    else:
                        skill = intent['skill']
                        if self.backend.mode == 'live' and not self.places.in_skill_zone(skill, self.health.get('position')):
                            raise ValueError('当前位置不在已标注的该技能执行区域')
                        self.task = dict(id=uuid.uuid4().hex, kind=kind, skill=skill,
                            profile=SKILLS[skill]['profile'], deadline=self.clock()+45)
                        self.state = 'SKILL_PREPARING'; self.generation += 1
                        self._send('skill', profile=self.task['profile'], skill=skill)
                        self._emit('skill_preparing', '准备执行' + SKILLS[skill]['name'] + '。')
                    accepted = True
                else: raise ValueError('未识别为控制指令。请一次说一个地点或技能，例如“去一楼会议室”。')
            except (ValueError, OSError, RuntimeError) as exc:
                self._emit('rejected', str(exc))
            result = dict(accepted=accepted, message=self.reply, state=self.state,
                          generation=self.generation, task_id=self.task['id'] if self.task else None)
            self.seen[command_id] = copy.deepcopy(result)
            while len(self.seen) > 256: self.seen.popitem(last=False)
            return result

    def _fault(self, reason):
        self.generation += 1
        try: self._send('emergency')
        except Exception: pass  # Latch locally even when the connection is already broken.
        finally:
            self.last_result = dict(ok=False, reason=reason, task_id=self.task['id'] if self.task else None)
            self.task = None; self.state = 'FAULT'; self._emit('fault', reason)

    def _complete(self, message):
        self.last_result = dict(ok=True, task_id=self.task['id'], kind=self.task['kind'], message=message)
        disabled = self.task.get('disable', False)
        self.task = None; self.state = 'DISABLED' if disabled else 'IDLE'
        self.generation += 1; self._emit('complete', message)

    def tick(self):
        with self.lock:
            try: events = self.backend.poll(self.clock())
            except Exception as exc:
                if self.state != 'FAULT': self._fault('控制连接异常：' + str(exc))
                return
            for event in events:
                if event.get('kind') == 'health':
                    self.health = copy.deepcopy(event['health']); self.health_at = self.clock()
                    if self.health.get('fault') and self.state not in ('FAULT', 'DISABLED'):
                        self._fault('底层运行时故障：' + str(self.health['fault']))
                    continue
                if not self.task or event.get('task_id') != self.task['id']: continue
                kind = event.get('kind')
                if kind == 'failed': self._fault(event.get('reason', '任务失败')); continue
                if kind == 'walk_ready' and self.state == 'NAV_PREPARING':
                    self.state = 'NAVIGATING'; self._send('navigate', goal=self.task['route'][0], leg=0)
                    self._emit('navigating', '正在前往' + self.task['destination'] + '。')
                elif kind == 'nav_arrived' and self.state == 'NAVIGATING' and event.get('leg') == self.task['leg']:
                    self.task['leg'] += 1
                    if self.task['leg'] < len(self.task['route']):
                        self.task['deadline'] = self.clock()+180
                        self._send('navigate', goal=self.task['route'][self.task['leg']], leg=self.task['leg'])
                    else:
                        self.state = 'STOPPING'; self.task['deadline'] = self.clock()+25
                        self._send('stop'); self._emit('arrived', '已到达，正在站稳。')
                elif kind == 'skill_started' and self.state == 'SKILL_PREPARING':
                    self.state = 'SKILL_EXECUTING'; self._emit('skill_started', '正在执行' + SKILLS[self.task['skill']]['name'] + '。')
                elif kind == 'skill_recovering' and self.state in ('SKILL_PREPARING', 'SKILL_EXECUTING'):
                    self.state = 'RECOVERING'; self._emit('recovering', '动作结束，正在恢复站立。')
                elif kind == 'skill_done' and self.state in ('SKILL_EXECUTING', 'RECOVERING'):
                    self._complete('技能已完成并恢复站立。')
                elif kind == 'stand_ready' and self.state == 'STOPPING':
                    self._complete('已到达' + self.task['destination'] + '并站稳。' if self.task['kind']=='navigate' else '已停止并稳定站立。')
            if self.task and self.clock() > self.task['deadline']: self._fault('任务超时，停止并锁存。')
            if self.state not in ('IDLE', 'DISABLED', 'FAULT') and self.clock()-self.health_at > 1.5:
                self._fault('机器人状态超时，停止并锁存。')
