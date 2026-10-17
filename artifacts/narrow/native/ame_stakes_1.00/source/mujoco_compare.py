"""Compare frozen MoE, MoE+IK, and a learned actor in identical MuJoCo lanes.

Height-scan actor is a privileged reference. It is not a camera policy. The
fused actor instead receives current visible ray returns and an unknown mask.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from go2_nav.active_env import ActiveTerrainEnv
from go2_nav.terrain_env import terrain_config
from .geometry import lane_boxes
from .provenance import snapshot


def populate(world, scene, cfg):
    world.remove(world.find("geom[@name='floor']"))
    ame = cfg["kind"].startswith("ame_")
    meta = {"water": [], "gaps": [], "goal_height": 0., "surfaces": [], "goal": [3.55 if ame else 7.55, 0.]}
    for i, (x0, x1, y0, y1, top, bottom) in enumerate(lane_boxes(cfg["difficulty"], cfg["kind"])):
        ox,oy = (4.,4.) if ame else (1.,1.5)
        x0 -= ox; x1 -= ox; y0 -= oy; y1 -= oy
        ET.SubElement(world, "geom", name=f"lane_{i}", type="box",
            pos=f"{(x0+x1)/2} {(y0+y1)/2} {(top+bottom)/2}",
            size=f"{(x1-x0)/2} {(y1-y0)/2} {(top-bottom)/2}",
            rgba=".55 .60 .65 1", friction=".8 .02 .01", group="0")
        meta["surfaces"].append([x0, x1, top])
    return meta


class ComparisonEnv(ActiveTerrainEnv):
    def __init__(self, cfg, actor_path=None, io_path=None):
        self.actor = torch.jit.load(str(actor_path)).eval() if actor_path else None
        self.io = json.loads(Path(io_path).read_text()) if io_path else None
        if self.actor is not None and not np.all(np.asarray(self.io["stiffness"]) > 0):
            raise ValueError("Exported PD gains are invalid. Re-export from the explicit Isaac Lab actuator.")
        self.previous_action = np.zeros(12, dtype=np.float32)
        super().__init__(cfg, scene="platform")

    def _world(self):
        with patch("go2_nav.terrains.populate", populate):
            super()._world()
        # Fix the common test model before evaluating any final checkpoint.
        # RoboGauge's benchmark XML narrows physical thigh ranges and gives
        # feet a priority friction of 0.4; match the training test budget here.
        for leg in range(4):
            self.model.jnt_range[self.joint_ids[leg*3+1]] = [-1.5708, 3.4907] if leg < 2 else [-.5236, 4.5379]
        self.model.geom_friction[self.foot_ids, 0] = self.cfg.get("test_friction", .8)
        self.model.actuator_ctrlrange[:] = [-23.5, 23.5]
        if self.actor is not None:
            names = self.io["joint_names"]
            self.actor_order = np.array([list(self.joint_ids).index(mujoco.mj_name2id(self.model,
                mujoco.mjtObj.mjOBJ_JOINT, name)) for name in names])
            self.actor_default = np.asarray(self.io["default_joint_pos"])
            default = np.empty(12)
            default[self.actor_order] = self.actor_default
            self.robot.default_dof_pos = default
        self.previous_action[:] = 0.
        self.last_scan = None
        self.last_scan_time = -np.inf
        self.observed_world = np.empty((0, 3))
        self.observed_time = np.empty(0)

    def _height_scan(self):
        yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        # Matches Isaac Lab GridPattern ordering='xy'.
        x, y = np.meshgrid(np.linspace(-.55, 1.05, 17), np.linspace(-.5, .5, 11))
        p = self.data.qpos[:3]
        result = []
        geom = np.zeros(1, np.int32)
        for xx, yy in zip(x.ravel(), y.ravel()):
            origin = np.array([p[0]+c*xx-s*yy, p[1]+s*xx+c*yy, p[2]+20.])
            d = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]), self.ray_groups, 1, self.base_id, geom)
            z = origin[2]-d if d >= 0 else p[2]-1.5
            result.append(np.clip(p[2]-z-.5, -1, 1))
        return np.asarray(result)

    def _measured_scan(self):
        # Reuse the existing occlusion-aware current sensor returns, transform
        # to gravity-level local coordinates, and leave unobserved cells empty.
        now = float(self.data.time)
        yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        world, times = self.observed_world, self.observed_time
        if now-self.last_scan_time >= .099:
            self._sense()
            new = self.level_points @ rot.T+self.data.qpos[:3]
            world = np.concatenate((world, new))
            times = np.r_[times, np.full(len(new), now)]
            self.last_scan_time = now
        pts = (world-self.data.qpos[:3]) @ rot
        ix = np.rint((pts[:, 0]+.55)/.1).astype(int)
        iy = np.rint((pts[:, 1]+.5)/.1).astype(int)
        valid = (ix >= 0) & (ix < 17) & (iy >= 0) & (iy < 11) & (now-times < 1.)
        elevation = np.full(187, -np.inf)
        candidate = np.flatnonzero(valid)
        descending = candidate[np.argsort(-pts[candidate, 2], kind="stable")]
        bins = iy[descending]*17+ix[descending]
        _, first = np.unique(bins, return_index=True)
        selected = descending[first]
        elevation[iy[selected]*17+ix[selected]] = pts[selected, 2]
        self.observed_world, self.observed_time = world[selected], times[selected]
        mask = np.isfinite(elevation)
        heights = np.where(mask, np.clip(-elevation-.5, -1, 1), 0.)
        return np.r_[heights, mask.astype(float)]

    def _control(self, command):
        if self.actor is None:
            return super()._control(command)
        order = self.actor_order
        target = np.empty(12)
        if self.settling:
            action = np.zeros(12)
        else:
            proprio = np.r_[self.data.sensor("imu_gyro").data,
                self.rotation.T @ np.array([0., 0., -1.]), command,
                self.data.qpos[self.qids][order]-self.actor_default,
                self.data.qvel[self.vids][order], self.previous_action]
            obs_dim = self.io["observation_dim"]
            if obs_dim == 235:
                proprio = np.r_[self.rotation.T @ self.data.qvel[:3], proprio]
            if obs_dim == 45:
                obs = proprio
            else:
                if obs_dim in (232, 235):
                    scan = self._height_scan()
                elif obs_dim == 419:
                    scan = self._measured_scan()
                else:
                    raise ValueError(f"Unsupported actor dimension {obs_dim}")
                obs = np.r_[proprio, scan]
            with torch.inference_mode():
                action = self.actor(torch.tensor(obs, dtype=torch.float32)[None]).numpy()[0]
            action = np.clip(action, -10., 10.)
        self.previous_action = action.copy()
        target[order] = self.actor_default+float(self.io["action_scale"])*action
        kp, kd = np.empty(12), np.empty(12)
        kp[order], kd[order] = self.io["stiffness"], self.io["damping"]
        for _ in range(round(self.cfg["control_dt"]/self.cfg["physics_dt"])):
            torque = kp*(target-self.data.qpos[self.qids])-kd*self.data.qvel[self.vids]
            self.data.ctrl[:] = np.clip(torque, -23.5, 23.5)
            mujoco.mj_step(self.model, self.data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["moe", "ik", "learned"], required=True)
    p.add_argument("--actor")
    p.add_argument("--io")
    p.add_argument("--kind", choices=["flat", "stairs", "platform", "stones", "ame_stakes", "ame_alternate", "ame_stakes12"], default="stairs")
    p.add_argument("--difficulty", type=float, default=.5)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=801)
    p.add_argument("--friction", type=float, default=.8)
    p.add_argument("--video", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.backend == "learned" and not (args.actor and args.io):
        p.error("learned requires --actor and --io")
    cfg = terrain_config()
    goal_x = 3.5 if args.kind.startswith("ame_") else 7.5
    cfg.update(kind=args.kind, difficulty=args.difficulty, goal=[goal_x+.05, 0.],
        max_seconds=18., navigation_dt=.02, anticipatory_lift=args.backend == "ik",
        test_friction=args.friction,
        auto_jump=False, jump_backend="disabled", sensor_payload_kg=0., goal_tolerance=.15,
        depth_width=32, depth_height=20, depth_pitch_degrees=20., depth_position=[.25, 0., .06],
        lidar_position=[0., 0., .16], lidar_azimuth_samples=96,
        lidar_elevation_degrees=np.linspace(-7, 52, 12).tolist())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    sources = snapshot(out)
    env = ComparisonEnv(cfg, args.actor if args.backend == "learned" else None, args.io)
    records = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed+episode)
            # Held-out initial pose perturbations, after standing reset only.
            env.data.qpos[1] += env.np_random.uniform(-.05, .05)
            yaw0 = env.np_random.uniform(-.04, .04)
            env.data.qpos[3:7] = [np.cos(yaw0/2), 0, 0, np.sin(yaw0/2)]
            mujoco.mj_forward(env.model, env.data)
            trace, calf_impulse, fall, left_lane, reached = [], 0., False, False, False
            climbed = False
            writer = imageio.get_writer(out/f"episode_{episode}.mp4", fps=25) if args.video and episode == 0 else None
            try:
                for step in range(900):
                    yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
                    cmd = np.array([.7, 0., np.clip(-.5*yaw, -.5, .5)])
                    env.command = cmd
                    env._control(cmd)
                    if step % 5 == 0:
                        env._sense()
                    for ci, contact in enumerate(env.data.contact):
                        a, b = contact.geom1, contact.geom2
                        if (env.model.geom_bodyid[a] == 0) != (env.model.geom_bodyid[b] == 0):
                            robot_geom = b if env.model.geom_bodyid[a] == 0 else a
                            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(env.model.geom_bodyid[robot_geom])) or ""
                            if "calf" in name and robot_geom not in env.foot_ids:
                                force = np.zeros(6)
                                mujoco.mj_contactForce(env.model, env.data, ci, force)
                                calf_impulse += abs(float(force[0]))*.02
                    contacts = {int(c.geom1) for c in env.data.contact} | {int(c.geom2) for c in env.data.contact}
                    support = sum(f in contacts for f in env.foot_ids)
                    if args.kind == "platform":
                        height = .03+.37*args.difficulty
                        climbed |= bool(2. < env.data.qpos[0] < 4.8 and env.data.qpos[2] > height+.18 and support >= 2 and env.rotation[2, 2] > .7)
                    fall = bool(env.rotation[2, 2] < .55 or env.data.qpos[2] < .12)
                    base_hit = any((env.model.geom_bodyid[c.geom1] == env.base_id and env.model.geom_bodyid[c.geom2] == 0) or
                                   (env.model.geom_bodyid[c.geom2] == env.base_id and env.model.geom_bodyid[c.geom1] == 0) for c in env.data.contact)
                    fall |= base_hit
                    left_lane = bool(abs(env.data.qpos[1]) > .85 or env.data.qpos[0] < -.7)
                    reached = bool(env.data.qpos[0] > goal_x and support >= 2 and env.rotation[2, 2] > .7)
                    trace.append({"time": (step+1)*.02, "position": env.data.qpos[:3].tolist(), "support_feet": support})
                    if writer and step % 2 == 0:
                        writer.append_data(env.render())
                    if reached or fall or left_lane:
                        break
            finally:
                if writer: writer.close()
            records.append({"episode": episode, "success": reached and not fall and not left_lane,
                "fall": fall, "left_lane": left_lane, "position": env.data.qpos[:3].tolist(),
                "climbed_platform": climbed, "seconds": trace[-1]["time"], "calf_impulse_proxy": calf_impulse})
            (out/f"trajectory_{episode}.json").write_text(json.dumps(trace))
            print(records[-1], flush=True)
    finally:
        env.close()
    result = {"args": vars(args), "successes": sum(r["success"] for r in records),
        "episodes": records, "source_sha256": sources,
        "actor_sha256": hashlib.sha256(Path(args.actor).read_bytes()).hexdigest() if args.actor else None}
    (out/"summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
