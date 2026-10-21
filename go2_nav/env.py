from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "RoboGauge"
sys.path.insert(0, str(VENDOR))

# Reuse the actual upstream observation construction, policy and joint mapping.
from robogauge.tasks.robots.go2.go2 import Go2
from robogauge.tasks.robots.go2.go2_config import Go2Config
from robogauge.tasks.gauge.goal_data import GoalData, VelocityGoal
from robogauge.utils.logger import logger


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))


class Go2MaplessEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config=None, scene="slalom", render_mode=None, sensor_mode="both"):
        super().__init__()
        self.cfg = load_config() if config is None else dict(config)
        self.scene = scene
        self.render_mode = render_mode
        if sensor_mode not in ("both", "depth", "lidar"):
            raise ValueError(sensor_mode)
        self.sensor_mode = sensor_mode
        self.action_space = spaces.Box(-1, 1, (3,), np.float32)
        # 72 lidar sectors + 12x16 depth + goal direction/distance + body velocity + last cmd.
        self.observation_space = spaces.Box(-1, 1, (273,), np.float32)
        torch.set_num_threads(1)
        if logger.logger is None:
            logger.create("go2_mapless", "simulation", console_output=False)
        if self.cfg.get("locomotion", "mlp") == "moe":
            from robogauge.tasks.robots.go2.go2_moe import Go2MoE
            from robogauge.tasks.robots.go2.go2_moe_config import Go2MoEConfig
            self.robot = Go2MoE(Go2MoEConfig())
        else:
            self.robot = Go2(Go2Config())
        self.renderer = None
        self.model = None
        self.command = np.zeros(3)
        self._make_rays()
        for slow, fast in (("control_dt", "physics_dt"), ("navigation_dt", "control_dt")):
            ratio = self.cfg[slow] / self.cfg[fast]
            if not np.isclose(ratio, round(ratio)):
                raise ValueError(f"{slow} must be an integer multiple of {fast}")

    def _make_rays(self):
        c = self.cfg
        az = np.linspace(-np.pi, np.pi, c["lidar_azimuth_samples"], endpoint=False)
        el = np.deg2rad(c["lidar_elevation_degrees"])
        a, e = np.meshgrid(az, el)
        self.lidar_dirs = np.stack((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)), -1).reshape(-1, 3)
        w, h = c["depth_width"], c["depth_height"]
        u = (np.arange(w) + 0.5 - w/2) / (w/2) * np.tan(np.deg2rad(c["depth_horizontal_fov_degrees"])/2)
        v = (np.arange(h) + 0.5 - h/2) / (h/2) * np.tan(np.deg2rad(c["depth_vertical_fov_degrees"])/2)
        u, v = np.meshgrid(u, v)
        self.depth_dirs = np.stack((np.ones_like(u), -u, -v), -1).reshape(-1, 3)
        self.depth_dirs /= np.linalg.norm(self.depth_dirs, axis=1, keepdims=True)
        p = np.deg2rad(c["depth_pitch_degrees"])
        self.depth_rotation = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        self.ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def _world(self):
        root = ET.parse(VENDOR / "resources/robots/go2/go2.xml").getroot()
        root.find("compiler").set("meshdir", str((VENDOR / "resources/robots/go2/assets").resolve()))
        root.find("option").set("timestep", str(self.cfg["physics_dt"]))
        root.find("option").set("gravity", "0 0 -9.81")
        # Upstream top-level actuators do not inherit the body childclass.
        # Assign explicit motor defaults so torque clipping has valid limits.
        for motor in root.find("actuator"):
            motor.set("class", "knee" if "calf" in motor.get("name", "") else "go2")
        world = root.find("worldbody")
        base = world.find("body")
        base.set("pos", "0 0 0.445")
        if self.cfg["sensor_payload_kg"]:
            inertial = base.find("inertial")
            inertial.set("mass", str(float(inertial.get("mass")) + self.cfg["sensor_payload_kg"]))
        # Camera points along body +X; MuJoCo camera looks along its local -Z.
        ET.SubElement(base, "camera", name="d435i", pos=" ".join(map(str, self.cfg["depth_position"])),
                      xyaxes="0 -1 0 0 0 1", fovy=str(self.cfg["depth_vertical_fov_degrees"]))
        ET.SubElement(base, "geom", name="mid360_visual", type="cylinder", pos=" ".join(map(str, self.cfg["lidar_position"])),
                      size="0.035 0.03", rgba="0.15 0.15 0.18 1", contype="0", conaffinity="0", group="4", mass="0")
        ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", diffuse="0.9 0.9 0.9")
        ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1", rgba="0.72 0.75 0.79 1", friction="0.8 0.02 0.01", group="0")
        self.obstacle_names = []
        def box(name, x, y, sx, sy, sz=0.6):
            self.obstacle_names.append(name)
            ET.SubElement(world, "geom", name=name, type="box", pos=f"{x} {y} {sz}", size=f"{sx} {sy} {sz}",
                          rgba="0.75 0.3 0.13 1", friction="0.8 0.02 0.01", group="0")
        hall_end = 16. if self.scene in ("mixed", "course") else 10.
        box("wall_left", (hall_end-2)/2, 3.0, (hall_end+2)/2, 0.1)
        box("wall_right", (hall_end-2)/2, -3.0, (hall_end+2)/2, 0.1)
        box("wall_back", -2.0, 0, 0.1, 3)
        box("wall_end", hall_end, 0, 0.1, 3)
        self.goal = np.array([8., 0.])
        if self.scene == "slalom":
            for i, (x, y) in enumerate([(2.4, -0.4), (4.4, 0.65), (6.3, -0.45)]):
                box(f"obstacle_{i}", x, y, 0.3, 0.65)
        elif self.scene == "random":
            for i, x in enumerate(np.linspace(2, 6.8, 5)):
                box(f"obstacle_{i}", x, self.np_random.uniform(-1.4, 1.4), 0.25, self.np_random.uniform(.25, .55))
        elif self.scene == "blocked":
            box("blocking_wall", 3, 0, .15, 3)
        elif self.scene in ("water", "stairs", "platform", "gap", "mixed", "course"):
            from .terrains import populate
            self.terrain_meta = populate(world, self.scene, self.cfg)
            if self.scene == "course":
                self.obstacle_names.append("course_block")
            if "goal" in self.terrain_meta:
                self.goal = np.asarray(self.terrain_meta["goal"], dtype=float)
        elif self.scene in ('rooms_dynamic','follow_obstacles','follow_stairs','follow_sand','follow_mixed'):
            from skills.navigation.tasks import populate
            self.terrain_meta=populate(world,self.scene,self.cfg,self.obstacle_names)
            self.goal=np.asarray(self.terrain_meta['goal'],dtype=float)
        elif self.scene != "empty":
            raise ValueError(f"Unknown scene: {self.scene}")
        if self.cfg.get("goal") is not None:
            requested = np.asarray(self.cfg["goal"], dtype=float)
            if requested.shape != (2,) or not np.all(np.isfinite(requested)):
                raise ValueError("goal must be two finite odometry-frame coordinates")
            self.goal = requested.copy()
            # Evaluation only: support height at the requested point.
            for x0, x1, top in getattr(self, "terrain_meta", {}).get("surfaces", []):
                if x0 <= self.goal[0] < x1:
                    self.terrain_meta["goal_height"] = top
        goal_z = getattr(self, "terrain_meta", {}).get("goal_height", 0.)+.1
        ET.SubElement(world, "site", name="goal", type="sphere", pos=f"{self.goal[0]} {self.goal[1]} {goal_z}", size="0.15", rgba="0.1 0.8 0.2 1")
        ET.SubElement(root, "visual")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.obstacle_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.obstacle_names}
        # Assert upstream actuator/joint ordering instead of silently relying on qpos slices.
        self.joint_ids = self.model.actuator_trnid[:, 0].copy()
        self.qids = self.model.jnt_qposadr[self.joint_ids]
        self.vids = self.model.jnt_dofadr[self.joint_ids]
        assert len(self.joint_ids) == self.robot.num_action == 12

    @property
    def rotation(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def _proprio(self):
        return NS(proprio=NS(
            imu=NS(ang_vel=self.data.sensor("imu_gyro").data.copy(), quat=self.data.sensor("imu_quat").data.copy()),
            joint=NS(pos=self.data.qpos[self.qids].copy(), vel=self.data.qvel[self.vids].copy())))

    def _control(self, command):
        goal = GoalData("velocity", velocity_goal=VelocityGoal(lin_vel_x=float(command[0]), lin_vel_y=float(command[1]), ang_vel_yaw=float(command[2])))
        with torch.inference_mode():
            obs = self.robot.build_observation(self._proprio(), goal)
            target, kp, kd, kind = self.robot.get_action(obs)
        assert kind == "P"
        target, kp, kd = self.modify_targets(target, kp, kd, command)
        for _ in range(round(self.cfg["control_dt"] / self.cfg["physics_dt"])):
            torque = kp * (target - self.data.qpos[self.qids]) - kd * self.data.qvel[self.vids]
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)

    def modify_targets(self, target, kp, kd, command):
        """Extension point for physical joint skills/residuals; no root-state edits."""
        return target, kp, kd

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._world()
        self.data.qpos[self.qids] = self.robot.default_dof_pos
        mujoco.mj_forward(self.model, self.data)
        self.robot.reset()
        self.command = np.zeros(3)
        for _ in range(round(self.cfg["settle_seconds"] / self.cfg["control_dt"])):
            self._control(self.command)
        self.start_time = float(self.data.time)
        self.last_pos = self.data.qpos[:2].copy()
        self.path_length = 0.
        self.steps = 0
        self._sense()
        return self._observation(), self._info()

    def _rays(self, origin, dirs, noise):
        rot = self.rotation
        point = self.data.xpos[self.base_id] + rot @ np.asarray(origin)
        world_dirs = dirs @ rot.T
        ranges = np.empty(len(dirs))
        geom = np.zeros(1, dtype=np.int32)
        for i, direction in enumerate(world_dirs):
            distance = mujoco.mj_ray(self.model, self.data, point, direction, self.ray_groups, 1, -1, geom)
            ranges[i] = distance if distance >= 0 else self.cfg["max_range"]
        misses = ranges >= self.cfg["max_range"]
        valid = ~misses
        ranges += self.np_random.normal(0, noise, ranges.shape)
        ranges = np.clip(ranges, .05, self.cfg["max_range"])
        kept = self.np_random.random(len(ranges)) > self.cfg["sensor_dropout"]
        self._last_miss_mask = misses & kept
        valid &= kept
        ranges[~valid] = self.cfg["max_range"]
        points_body = np.asarray(origin) + dirs * ranges[:, None]
        points_world = self.data.xpos[self.base_id] + points_body @ rot.T
        # Ground is not a vertical collision obstacle for the navigation controller.
        valid &= (points_world[:, 2] > .12) & (points_world[:, 2] < 1.1)
        return ranges, points_body[valid]

    def _sense(self):
        self.lidar_ranges, lp = self._rays(self.cfg["lidar_position"], self.lidar_dirs, self.cfg["lidar_noise_std"])
        self.lidar_free_dirs = self.lidar_dirs[self._last_miss_mask].copy()
        dd = self.depth_dirs @ self.depth_rotation.T
        dr, dp = self._rays(self.cfg["depth_position"], dd, self.cfg["depth_noise_std"])
        # Optical-axis Z depth, not Euclidean ray length. Sensor frame +X is optical forward here.
        self.depth = (dr * self.depth_dirs[:, 0]).reshape(self.cfg["depth_height"], self.cfg["depth_width"])
        self.points = np.concatenate(([lp] if self.sensor_mode == "lidar" else [dp] if self.sensor_mode == "depth" else [lp, dp]), axis=0)
        self.scan = np.full(72, self.cfg["max_range"])
        if len(lp) and self.sensor_mode != "depth":
            angle = np.arctan2(lp[:, 1], lp[:, 0])
            bins = np.floor((angle + np.pi) / (2*np.pi) * 72).astype(int) % 72
            np.minimum.at(self.scan, bins, np.linalg.norm(lp[:, :2], axis=1))

    def _observation(self):
        rot = self.rotation
        goal_body = rot[:2, :2].T @ (self.goal - self.data.qpos[:2])
        distance = np.linalg.norm(goal_body)
        goal_direction = goal_body / max(distance, 1e-6)
        # Nearest-neighbour downsampling, fixed shape for PPO sensor ablations.
        yi = np.linspace(0, self.depth.shape[0]-1, 12).astype(int)
        xi = np.linspace(0, self.depth.shape[1]-1, 16).astype(int)
        depth = self.depth[np.ix_(yi, xi)].ravel() / self.cfg["max_range"]
        if self.sensor_mode == "lidar":
            depth = np.ones_like(depth)
        obs = np.concatenate((self.scan/self.cfg["max_range"], depth, goal_direction,
                              [min(distance/12, 1)], (rot.T @ self.data.qvel[:3])/2,
                              self.command / np.array([self.cfg["max_vx"], self.cfg["max_vy"], self.cfg["max_wz"]])))
        return np.clip(obs, -1, 1).astype(np.float32)

    def _collision(self):
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self.obstacle_ids and self.model.geom_bodyid[b] != 0) or (b in self.obstacle_ids and self.model.geom_bodyid[a] != 0):
                return True
        return False

    def _info(self):
        distance = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        return dict(distance=distance, success=distance < self.cfg["goal_tolerance"], collision=self._collision(),
                    fallen=bool(self.data.qpos[2] < .16 or self.rotation[2, 2] < .55),
                    seconds=float(self.data.time-self.start_time), path_length=self.path_length,
                    position=self.data.qpos[:3].tolist(), command=self.command.tolist())

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected three finite normalized velocity commands")
        previous = self.command.copy()
        before = np.linalg.norm(self.goal - self.data.qpos[:2])
        self.command = np.clip(action, -1, 1) * np.array([self.cfg["max_vx"], self.cfg["max_vy"], self.cfg["max_wz"]])
        for _ in range(round(self.cfg["navigation_dt"] / self.cfg["control_dt"])):
            self._control(self.command)
        self.path_length += float(np.linalg.norm(self.data.qpos[:2]-self.last_pos))
        self.last_pos = self.data.qpos[:2].copy()
        self.steps += 1
        self._sense()
        info = self._info()
        reward = 8*(before-info["distance"]) - .02 - .03*float(np.sum((self.command-previous)**2))
        reward += 30*info["success"] - 20*(info["collision"] or info["fallen"])
        terminated = info["success"] or info["collision"] or info["fallen"]
        truncated = info["seconds"] >= self.cfg["max_seconds"]
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.qpos[:3]
        camera.distance = 5
        camera.azimuth = 135
        camera.elevation = -45
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
