"""Single-owner physics thread; HTTP/UI never writes robot poses or motors."""
import argparse
import io
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Literal

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
import uvicorn


class Control(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    action: Literal['goal', 'start', 'stop', 'reset']
    goal: list[float] | None = Field(default=None, min_length=2, max_length=2)
    scene: Literal['empty', 'slalom', 'blocked'] | None = None


class Engine:
    def __init__(self, config=None, checkpoint=None):
        self.config, self.checkpoint = config, checkpoint
        self.commands = queue.Queue(maxsize=16)
        self.lock = threading.Lock()
        self.state = {'phase': 'INITIALIZING'}
        self.jpeg = b''
        self.error = None
        self.last_client = time.monotonic()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        from go2_nav.env import Go2MaplessEnv, load_config
        from go2_nav.planner import ReactivePlanner
        from .controller import action as navigation_action
        env = None
        try:
            cfg = load_config(self.config or Path(__file__).with_name('config.json'))
            config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            cfg.update(max_seconds=1e9)
            scene = 'empty'
            env = Go2MaplessEnv(cfg, scene=scene, sensor_mode='lidar')
            _, info = env.reset(seed=200)
            planner = ReactivePlanner(cfg)
            policy = None
            policy_sha = None
            if self.checkpoint:
                from stable_baselines3 import PPO
                metadata = Path(self.checkpoint).parent/'manifest.json'
                if not metadata.is_file():
                    metadata = Path(self.checkpoint).parent/'training_manifest.json'
                manifest = json.loads(metadata.read_text())
                if manifest.get('config_sha256') != config_sha or manifest.get('sensor_mode') != 'lidar':
                    raise ValueError('Checkpoint training configuration/sensor mode differs from this live worker')
                policy_sha = hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest()
                policy = PPO.load(self.checkpoint, device='cpu')
                if policy.observation_space.shape != (273,) or policy.action_space.shape != (3,):
                    raise ValueError('Navigation policy must match 273 observations / 3 normalized velocity actions')
            phase = 'READY'
            reason = ''
            goal = np.array([2., 0.])
            env.goal[:] = goal
            path = []
            events = []
            wall = time.monotonic()
            sim_start = float(env.data.time)
            started = 0.
            next_render = 0.
            boxes = []

            def geometry():
                return [dict(name=env.model.geom(i).name,
                    x=float(env.model.geom_pos[i, 0]), y=float(env.model.geom_pos[i, 1]),
                    sx=float(env.model.geom_size[i, 0]), sy=float(env.model.geom_size[i, 1]))
                    for i in sorted(env.obstacle_ids)]

            def valid(p):
                return (-1.4 <= p[0] <= 9.3 and -2.45 <= p[1] <= 2.45 and
                    all(not (abs(p[0]-b['x']) < b['sx']+.38 and
                        abs(p[1]-b['y']) < b['sy']+.38) for b in boxes))

            boxes = geometry()
            while not self.closed.is_set():
                while not self.commands.empty():
                    cmd = self.commands.get_nowait()
                    if cmd.action == 'reset':
                        scene = cmd.scene or scene
                        env.close()
                        env = Go2MaplessEnv(cfg, scene=scene, sensor_mode='lidar')
                        _, info = env.reset(seed=200)
                        boxes = geometry()
                        env.goal[:] = goal
                        phase, reason, path = 'READY', 'scene_reset', []
                        wall = time.monotonic()
                        sim_start = float(env.data.time)
                        next_render = 0.
                        planner = ReactivePlanner(cfg)
                    elif cmd.action == 'goal':
                        requested = np.asarray(cmd.goal, dtype=float)
                        if not valid(requested):
                            reason = 'goal_outside_free_workspace'
                        elif phase != 'FAULT':
                            goal = requested.copy()
                            env.goal[:] = goal
                            planner = ReactivePlanner(cfg)
                            reason = 'goal_selected'
                            if phase == 'ARRIVED':
                                phase = 'READY'
                    elif cmd.action == 'start' and phase in ('READY', 'STOPPED', 'ARRIVED'):
                        if valid(goal):
                            phase, reason, started = 'RUNNING', '', float(env.data.time)
                    elif cmd.action == 'stop':
                        if phase != 'FAULT':
                            phase, reason = 'STOPPED', 'user_stop'
                    events.append(dict(time=round(float(env.data.time), 3), action=cmd.action, reason=reason))
                    events = events[-12:]
                if phase == 'RUNNING' and time.monotonic()-self.last_client > 2.5:
                    phase, reason = 'STOPPED', 'client_timeout'
                if phase == 'RUNNING' and float(env.data.time)-started > 120:
                    phase, reason = 'STOPPED', 'navigation_timeout'
                if phase == 'RUNNING':
                    # Planner sees ray returns, odometry and target only. Display boxes never enter act().
                    action = navigation_action(env, planner, policy)
                else:
                    action = np.zeros(3)
                _, _, _, _, info = env.step(action)
                if info['collision'] or info['fallen']:
                    phase, reason = 'FAULT', 'collision' if info['collision'] else 'fallen'
                elif phase == 'RUNNING' and info['success']:
                    phase, reason = 'ARRIVED', 'within_goal_tolerance'
                if not path or np.linalg.norm(np.asarray(path[-1])-env.data.qpos[:2]) > .04:
                    path.append(env.data.qpos[:2].tolist())
                    path = path[-1600:]
                if env.data.time >= next_render:
                    env.model.site('goal').pos[:2] = goal
                    frame = env.render()
                    buf = io.BytesIO()
                    Image.fromarray(frame).save(buf, format='JPEG', quality=82)
                    elapsed = time.monotonic()-wall
                    snap = dict(phase=phase, reason=reason, scene=scene, goal=goal.tolist(),
                        goal_valid=bool(valid(goal)), position=env.data.qpos[:3].tolist(),
                        yaw=float(np.arctan2(env.rotation[1, 0], env.rotation[0, 0])),
                        seconds=float(env.data.time), distance=float(np.linalg.norm(goal-env.data.qpos[:2])),
                        command=env.command.tolist(), collision=info['collision'], fallen=info['fallen'],
                        path=path, boxes=boxes, events=events, hardware_enabled=False,
                        realtime_factor=(float(env.data.time)-sim_start)/max(elapsed, .01),
                        backend='MuJoCo / RoboGauge MoE / '+('PPO navigation' if policy else 'sensor-only reactive planner'),
                        policy_sha256=policy_sha,
                        planner_inputs='simulated lidar returns, odometry, selected goal; no obstacle metadata')
                    with self.lock:
                        self.state, self.jpeg = snap, buf.getvalue()
                    next_render = float(env.data.time)+.10
                delay = (float(env.data.time)-sim_start)-(time.monotonic()-wall)
                if delay > 0:
                    self.closed.wait(min(delay, .1))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = str(exc)
        finally:
            if env is not None:
                env.close()


def create_app(config=None, checkpoint=None):
    app = FastAPI(title='Live Go2 MuJoCo navigation')
    engine = Engine(config=config, checkpoint=checkpoint)

    @app.get('/')
    def page():
        return FileResponse(Path(__file__).with_name('index.html'))

    @app.get('/api/state')
    def state():
        engine.last_client = time.monotonic()
        if engine.error:
            raise HTTPException(503, engine.error)
        with engine.lock:
            return engine.state

    @app.get('/api/frame.jpg')
    def frame():
        with engine.lock:
            return Response(engine.jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.post('/api/control')
    def control(cmd: Control):
        if cmd.action == 'goal' and cmd.goal is None:
            raise HTTPException(422, 'goal requires two finite coordinates')
        if engine.error:
            raise HTTPException(503, engine.error)
        try:
            engine.commands.put_nowait(cmd)
        except queue.Full:
            raise HTTPException(429, 'control queue full')
        return {'accepted': True}

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Locally trained PPO velocity head; default is the classical planner')
    args = parser.parse_args()
    uvicorn.run(create_app(args.config, args.checkpoint), host='127.0.0.1', port=args.port, access_log=False)
