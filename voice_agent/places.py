"""Persistent named places and explicit, operator-labelled route graph."""
import copy
import heapq
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid


def normalized(text):
    return re.sub(r'[\s，。！？、,.!?；;：:]', '', str(text)).lower()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('坐标必须是有限数值')
    return float(value)


class Places:
    def __init__(self, path, initial=None):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else copy.deepcopy(initial)
        if not self.data or self.data.get('schema') != 1:
            raise ValueError('需要 schema=1 的地图标注文件')
        for key in ('map_id', 'frame_id', 'navigation_frame'):
            if not isinstance(self.data.get(key), str) or not self.data[key]:
                raise ValueError('缺少地图标识/坐标系: ' + key)
        self.data.setdefault('places', [])
        self.data.setdefault('edges', [])
        seen = set()
        for p in self.data['places']:
            self._validate(p)
            if p['id'] in seen: raise ValueError('重复地点 ID')
            seen.add(p['id'])
        self._validate_edges(self.data['edges'])
        self._transform([0, 0, 0], 0)  # validate any supplied registration

    def _validate(self, place):
        if not isinstance(place.get('name'), str) or not 1 <= len(place['name']) <= 60:
            raise ValueError('地点名需为 1–60 字符')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', place.get('id', '')):
            raise ValueError('非法地点 ID')
        if not isinstance(place.get('floor'), str) or len(place['floor']) > 30:
            raise ValueError('需要楼层名称')
        if not isinstance(place.get('xyz'), list) or len(place['xyz']) != 3:
            raise ValueError('需要 xyz 三维位置')
        place['xyz'] = [number(v) for v in place['xyz']]
        place['yaw'] = number(place.get('yaw', 0))
        aliases = place.get('aliases', [])
        if not isinstance(aliases, list) or len(aliases) > 20 or any(not isinstance(v, str) or len(v) > 60 for v in aliases):
            raise ValueError('非法地点别名')
        place['aliases'] = aliases
        zones = place.get('skills', [])
        if not isinstance(zones, list) or any(x not in ('backflip', 'sideflip', 'handstand') for x in zones):
            raise ValueError('未知技能区域类型')
        place['skills'] = zones
        place['zone_radius'] = number(place.get('zone_radius', .5))
        if not 0 < place['zone_radius'] <= 5: raise ValueError('技能区域半径需在 0–5 米')

    def _validate_edges(self, edges):
        ids = {p['id'] for p in self.data['places']}
        if not isinstance(edges, list) or len(edges) > 2000: raise ValueError('非法路线边')
        for edge in edges:
            if set(edge) - {'from', 'to', 'bidirectional'}: raise ValueError('非法路线边字段')
            if edge.get('from') not in ids or edge.get('to') not in ids or edge['from'] == edge['to']:
                raise ValueError('路线端点不存在或重复')
            if type(edge.get('bidirectional', True)) is not bool: raise ValueError('路线方向需为布尔值')

    def snapshot(self):
        with self.lock: return copy.deepcopy(self.data)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='places-', suffix='.tmp', dir=str(self.path.parent))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as out:
                json.dump(self.data, out, ensure_ascii=False, indent=2, allow_nan=False)
                out.flush(); os.fsync(out.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp): os.unlink(temp)

    def put(self, value):
        with self.lock:
            place = {k: copy.deepcopy(v) for k, v in value.items() if k in
                     ('id', 'name', 'floor', 'xyz', 'yaw', 'aliases', 'skills', 'zone_radius')}
            place.setdefault('id', uuid.uuid4().hex)
            self._validate(place)
            for p in self.data['places']:
                if p['id'] != place['id'] and normalized(p['name']) == normalized(place['name']):
                    raise ValueError('地点名已存在，请使用楼层前缀区分')
            old = copy.deepcopy(self.data)
            self.data['places'] = [p for p in self.data['places'] if p['id'] != place['id']] + [place]
            try: self.save()
            except Exception: self.data = old; raise
            return copy.deepcopy(place)

    def remove(self, place_id):
        with self.lock:
            if not any(p['id'] == place_id for p in self.data['places']): raise ValueError('地点不存在')
            old = copy.deepcopy(self.data)
            self.data['places'] = [p for p in self.data['places'] if p['id'] != place_id]
            self.data['edges'] = [e for e in self.data['edges'] if place_id not in (e['from'], e['to'])]
            try: self.save()
            except Exception: self.data = old; raise

    def set_edges(self, edges):
        with self.lock:
            self._validate_edges(edges)
            old = self.data['edges']; self.data['edges'] = copy.deepcopy(edges)
            try: self.save()
            except Exception: self.data['edges'] = old; raise

    def resolve(self, name):
        key = normalized(name)
        with self.lock:
            return [copy.deepcopy(p) for p in self.data['places'] if
                    key in {normalized(v) for v in [p['name']] + p['aliases']}]

    def _transform(self, xyz, yaw):
        t = self.data.get('T_navigation_map')
        if t is None:
            if self.data['frame_id'] != self.data['navigation_frame']:
                raise ValueError('地图与导航坐标系不同，需配置实测 T_navigation_map')
            return list(xyz), yaw
        import numpy as np
        t = np.asarray(t, dtype=float)
        if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError('非法地图配准矩阵')
        r = t[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1) > 1e-5:
            raise ValueError('地图配准需为右手刚体变换')
        p = (t @ np.r_[xyz, 1])[:3]
        heading = r @ [math.cos(yaw), math.sin(yaw), 0]
        if math.hypot(heading[0], heading[1]) < .1: raise ValueError('地图朝向无法转为平面导航朝向')
        return p.tolist(), math.atan2(heading[1], heading[0])

    def goal(self, place):
        xyz, yaw = self._transform(place['xyz'], place['yaw'])
        return dict(xyz=xyz, yaw=yaw, frame_id=self.data['navigation_frame'],
                    map_id=self.data['map_id'], place_id=place['id'], name=place['name'], floor=place['floor'])

    def map_position(self, navigation_xyz):
        if navigation_xyz is None: return None
        t = self.data.get('T_navigation_map')
        if t is None: return list(navigation_xyz)
        import numpy as np
        return (np.linalg.inv(np.asarray(t, dtype=float)) @ np.r_[navigation_xyz, 1])[:3].tolist()

    def route(self, target, position, require_graph=False):
        """Dijkstra on operator-labelled edges; does not infer free space from PCD."""
        with self.lock:
            edges = self.data['edges']
            if not edges:
                if require_graph: raise ValueError('实机导航需要已标注路线边；点云本身不是全局路线')
                return [self.goal(target)]
            goals = {p['id']: self.goal(p) for p in self.data['places']}
            if not position: raise ValueError('没有机器人实际位置，无法选择路线起点')
            start = min(goals, key=lambda key: math.dist(goals[key]['xyz'], position))
            if math.dist(goals[start]['xyz'], position) > self.data.get('route_start_radius', 1.0):
                raise ValueError('机器人不在已标注路线起点附近，请先定位或补充起点')
            adjacency = {key: [] for key in goals}
            for e in edges:
                a, b = e['from'], e['to']; weight = math.dist(goals[a]['xyz'], goals[b]['xyz']) + .001
                adjacency[a].append((b, weight))
                if e.get('bidirectional', True): adjacency[b].append((a, weight))
            queue = [(0., start, [start])]; visited = set()
            while queue:
                cost, node, path = heapq.heappop(queue)
                if node in visited: continue
                visited.add(node)
                if node == target['id']:
                    ids = path[1:] or [node]
                    return [copy.deepcopy(goals[key]) for key in ids]
                for nxt, weight in adjacency[node]:
                    if nxt not in visited: heapq.heappush(queue, (cost+weight, nxt, path+[nxt]))
            raise ValueError('标注图中没有到该地点的可达路线')

    def in_skill_zone(self, skill, position):
        if not position: return False
        with self.lock:
            return any(skill in p['skills'] and math.dist(self.goal(p)['xyz'], position) <= p['zone_radius']
                       for p in self.data['places'])
