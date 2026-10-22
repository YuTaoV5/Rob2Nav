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
    action: Literal['goal', 'start', 'stop', 'reset', 'camera']
    goal: list[float] | None = Field(default=None, min_length=2, max_length=2)
    camera: dict[str,float] | None = None
    scene: Literal['empty', 'slalom', 'blocked','rooms_dynamic','follow_obstacles','follow_stairs','follow_sand','follow_mixed','stairs_door','stairs_route'] | None = None
    task: Literal['navigate','follow'] | None = None
    person_id: Literal['person_01','person_02','person_03'] = 'person_01'
    follow_distance: float = Field(default=1.2,ge=.9,le=2.5)
    difficulty: int = Field(default=2,ge=1,le=3)
    seed: int = Field(default=200,ge=0,le=999999)


class Engine:
    def __init__(self, config=None, checkpoint=None, planner_backend="ppo"):
        self.planner_backend=planner_backend
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
        from go2_nav.env import load_config
        from .task_env import TaskEnvironment as Go2MaplessEnv
        from .tasks import TASK_SCENES,STEP_TASK_SCENES,StaticRoute,follow_goal,pedestrian_guard
        from go2_nav.planner import ReactivePlanner
        from .controller import action as navigation_action
        env = None
        scan = None
        try:
            cfg = load_config(self.config or Path(__file__).with_name('config.json'))
            config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            cfg.update(max_seconds=1e9)
            scene = 'stairs_door' if self.planner_backend=='scan' else 'empty'
            task_mode='navigate';person_id='person_01';spacing=1.2
            tracking='not_requested';safety_reason='';last_seen=-1e9
            route=[];route_at=-1e9;route_goal=None;local_goal=None;local_at=-1e9
            min_person_distance=None;follow_errors=[];lost_samples=0;route_failures=0
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
            if self.planner_backend=='scan':
                from .scan_client import ScanClient
                scan=ScanClient()
            phase = 'READY'
            reason = ''
            goal = env.goal.copy() if scene in TASK_SCENES else np.array([2., 0.])
            env.goal[:] = goal
            path = []
            events = []
            wall = time.monotonic()
            sim_start = float(env.data.time)
            started = 0.
            next_render = 0.
            boxes = []
            camera_settings = dict(azimuth=135.,elevation=-45.,distance=5.,pan_x=0.,pan_y=0.)

            def geometry():
                return [dict(name=env.model.geom(i).name,
                    x=float(env.model.geom_pos[i, 0]), y=float(env.model.geom_pos[i, 1]),
                    sx=float(env.model.geom_size[i, 0]), sy=float(env.model.geom_size[i, 1]))
                    for i in sorted(env.obstacle_ids) if env.model.geom_bodyid[i]==0]

            def valid(p):
                return router.free(p)

            boxes = geometry();router=StaticRoute(boxes,clearance=.60 if scene=='stairs_route' else .37)
            while not self.closed.is_set():
                while not self.commands.empty():
                    cmd = self.commands.get_nowait()
                    if cmd.action == 'camera':
                        camera_settings.update(cmd.camera or {})
                        continue
                    if cmd.action == 'reset':
                        if scan:scan.reset()
                        scene = cmd.scene or scene
                        task_mode=cmd.task or ('follow' if scene.startswith('follow_') else 'navigate')
                        person_id=cmd.person_id;spacing=cmd.follow_distance
                        cfg['task_difficulty']=cmd.difficulty
                        tracking='not_requested';last_seen=-1e9;local_goal=None;route_goal=None;route=[];route_at=-1e9
                        min_person_distance=None;follow_errors=[];lost_samples=0;route_failures=0
                        env.close()
                        env = Go2MaplessEnv(cfg, scene=scene, sensor_mode='lidar')
                        _, info = env.reset(seed=cmd.seed)
                        boxes = geometry();router=StaticRoute(boxes,clearance=.60 if scene=='stairs_route' else .37)
                        if scene in TASK_SCENES:goal=env.goal.copy()
                        env.goal[:] = goal
                        phase, reason, path = 'READY', 'scene_reset', []
                        wall = time.monotonic()
                        sim_start = float(env.data.time)
                        next_render = 0.
                        planner = ReactivePlanner(cfg)
                    elif cmd.action == 'goal':
                        requested = np.asarray(cmd.goal, dtype=float)
                        if task_mode=='follow':
                            reason='follow_target_is_identity_locked'
                            continue
                        if not valid(requested):
                            reason = 'goal_outside_free_workspace'
                        elif phase != 'FAULT':
                            route_goal=None;local_goal=None
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
                tracks=env.observations();env.actors_active=phase=='RUNNING'
                env.follow_mode=task_mode=='follow';env.selected_person=person_id;env.guide_gap=spacing+1.3
                enabled=phase=='RUNNING';safety_reason='';person_distance=None
                if task_mode=='follow':
                    target=next((p for p in tracks if p['id']==person_id and p['visible']),None)
                    if target is not None:
                        last_seen=float(env.data.time);tracking='locked_simulated_id'
                        desired,person_distance=follow_goal(env.data.qpos[:2],target['position'],target['velocity'],max(.95,spacing-.20))
                        # A forward tracking carrot; the independent measured
                        # person-distance hold still enforces requested spacing.
                        if phase=='RUNNING':follow_errors.append(abs(person_distance-spacing));follow_errors=follow_errors[-2400:]
                        if router.free(desired):goal=desired;env.goal[:]=goal
                        else:enabled=False;safety_reason='follow_goal_not_in_free_map'
                        if person_distance<=spacing+.15:enabled=False;safety_reason='follow_distance_hold'
                    else:
                        enabled=False
                        if phase=='RUNNING':lost_samples+=1
                        tracking='occluded_stop' if env.data.time-last_seen<.8 else 'lost_stop'
                        safety_reason='selected_person_not_observed'
                if scene in TASK_SCENES and enabled:
                    now=float(env.data.time)
                    if route_goal is None or np.linalg.norm(goal-route_goal)>.4 or now-route_at>2.:
                        route=router.path(env.data.qpos[:2],goal);route_goal=goal.copy();route_at=now
                        if not route:route_failures+=1
                    while len(route)>1 and np.linalg.norm(np.asarray(route[1])-env.data.qpos[:2])<.5:route.pop(0)
                    if len(route)>1:
                        if local_goal is None or now-local_at>1.:
                            desired=router.carrot(env.data.qpos[:2],route)
                            if local_goal is None or np.linalg.norm(desired-local_goal)>.35:local_goal=desired
                            local_at=now
                    elif router.line(env.data.qpos[:2],goal):local_goal=goal.copy()
                    else:enabled=False;safety_reason='no_global_route'
                else:
                    if scene not in TASK_SCENES:local_goal=goal.copy()
                if scan is not None:
                    action=scan.act(env,local_goal if local_goal is not None else goal,enabled)
                elif enabled:
                    action=navigation_action(env,planner,policy)
                else:action=np.zeros(3)
                if scene in TASK_SCENES:
                    limits=np.array([cfg['max_vx'],cfg['max_vy'],cfg['max_wz']])
                    physical=np.clip(action*limits,[-.35,-.20,-.8],[.45,.20,.8])
                    if scene in STEP_TASK_SCENES:physical=np.clip(physical,[-.20,-.12,-.5],[.30,.12,.5])
                    physical,guard_reason=pedestrian_guard(physical,env.rotation,env.data.qpos[:2],tracks)
                    if guard_reason:safety_reason=guard_reason
                    action=physical/limits
                    for person in tracks:
                        distance=float(np.linalg.norm(np.asarray(person['position'])-env.data.qpos[:2]))
                        min_person_distance=distance if min_person_distance is None else min(min_person_distance,distance)
                _, _, _, _, info = env.step(action)
                if info['collision'] or info['fallen']:
                    phase, reason = 'FAULT', 'collision' if info['collision'] else 'fallen'
                elif phase == 'RUNNING' and task_mode=='navigate' and info['success']:
                    phase, reason = 'ARRIVED', 'within_goal_tolerance'
                if not path or np.linalg.norm(np.asarray(path[-1])-env.data.qpos[:2]) > .04:
                    path.append(env.data.qpos[:2].tolist())
                    path = path[-1600:]
                if env.data.time >= next_render:
                    env.model.site('goal').pos[:2] = goal
                    env.model.site('goal').pos[2] = .10+env.ground(float(goal[0]))
                    import mujoco
                    if env.renderer is None:env.renderer=mujoco.Renderer(env.model,height=480,width=640)
                    camera=mujoco.MjvCamera()
                    camera.azimuth=camera_settings['azimuth'];camera.elevation=camera_settings['elevation'];camera.distance=camera_settings['distance']
                    camera.lookat[:]=env.data.qpos[:3];camera.lookat[0]+=camera_settings['pan_x'];camera.lookat[1]+=camera_settings['pan_y']
                    env.renderer.update_scene(env.data,camera=camera);frame=env.renderer.render().copy()
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
                        backend='MuJoCo / RoboGauge MoE / '+('SCAN ROS2 native planner' if scan else 'PPO navigation' if policy else 'sensor-only reactive planner'),
                        scan=scan.telemetry if scan else None,
                        task=task_mode,people=tracks,selected_person=person_id,tracking=tracking,follow_distance=spacing,
                        person_distance=person_distance,safety_reason=safety_reason,route=route,local_goal=local_goal.tolist() if local_goal is not None else None,
                        task_metrics=dict(min_person_center_distance=min_person_distance,follow_mean_abs_error=float(np.mean(follow_errors)) if follow_errors else None,
                                          follow_samples=len(follow_errors),lost_samples=lost_samples,route_failures=route_failures),
                        terrain=getattr(env,'terrain_meta',{}).get('terrain','flat'),
                        terrain_surfaces=getattr(env,'terrain_meta',{}).get('surfaces',[]),sand_region=getattr(env,'terrain_meta',{}).get('sand_region'),
                        door=getattr(env,'terrain_meta',{}).get('door'),step_height_m=getattr(env,'terrain_meta',{}).get('step_height_m'),
                        course_metrics=getattr(env,'course_metrics',{}),
                        routing_clearance_m=router.clearance,
                        low_step_filter='support-relative 0.20m height rejection; known bounded low-step course' if scene in STEP_TASK_SCENES else None,
                        person_observation='simulator-labelled identity and pose with lidar-origin line-of-sight; not RGB/ReID',
                        camera=camera_settings.copy(),policy_sha256=policy_sha,
                        planner_inputs=('static building map A* subgoals + SCAN ray pointcloud + visible simulated person tracks' if scene in TASK_SCENES else 'simulated lidar returns, odometry, selected goal; no obstacle metadata'))
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
            if scan is not None:scan.close()
            if env is not None:
                env.close()


def create_app(config=None, checkpoint=None,planner_backend="ppo"):
    app = FastAPI(title='Live Go2 MuJoCo navigation')
    engine = Engine(config=config, checkpoint=checkpoint,planner_backend=planner_backend)

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
        engine.last_client=time.monotonic()
        if cmd.action == 'goal' and cmd.goal is None:
            raise HTTPException(422, 'goal requires two finite coordinates')
        if cmd.action=='camera':
            from studio.live_skills import Camera
            try:cmd.camera=Camera(**(cmd.camera or {})).model_dump()
            except ValueError as exc:raise HTTPException(422,str(exc))
        if (cmd.scene in ('rooms_dynamic','follow_obstacles','follow_stairs','follow_sand','follow_mixed','stairs_door','stairs_route') or cmd.task=='follow') and planner_backend!='scan':
            raise HTTPException(422,'Task scenarios require the SCAN worker')
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
    parser.add_argument('--planner',choices=['ppo','scan'],default='ppo')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Locally trained PPO velocity head; default is the classical planner')
    args = parser.parse_args()
    uvicorn.run(create_app(args.config, args.checkpoint,args.planner), host='127.0.0.1', port=args.port, access_log=False)
