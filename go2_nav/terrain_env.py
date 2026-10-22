"""Sensor-only traversability and evaluation on physical positive/negative terrain."""
import numpy as np
import mujoco
from .env import Go2MaplessEnv, load_config
from .skills import JumpSkill


def terrain_config(path=None):
    c = load_config()
    c.update(locomotion="moe", depth_pitch_degrees=25, depth_width=64, depth_height=48,
             max_climb_height=.24, jump_enabled=True, terrain_height=.08, max_vx=.85)
    if path:
        c.update(load_config(path))
    return c


class TerrainEnv(Go2MaplessEnv):
    def __init__(self, config=None, **kwargs):
        self.skill = JumpSkill((config or {}).get("jump_parameters"))
        self.residual = np.zeros(12)
        self.residual_policy = None
        super().__init__(terrain_config() if config is None else config, **kwargs)
        if self.cfg.get("residual_checkpoint"):
            from stable_baselines3 import PPO
            self.residual_policy = PPO.load(self.cfg["residual_checkpoint"], device="cpu")

    def reset(self, **kwargs):
        self.skill = JumpSkill(self.cfg.get("jump_parameters"))
        self.residual = np.zeros(12)
        self.water_entered = False
        self.airborne_time = 0.
        self.max_airborne_time = 0.
        self.jump_count = 0
        self.height_peak = 0.
        self.crossed_gap_airborne = False
        self.foot_ids = []
        self.support_z = 0.
        self.gap_edge_world = None
        self.settling = True
        self.visited_top=False
        self.terminal_stable_s=0.
        result = super().reset(**kwargs)
        self.settling = False
        return result

    def _world(self):
        super()._world()
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in ("FL", "FR", "RL", "RR")]
        self.foot_body_ids = set(self.model.geom_bodyid[self.foot_ids].tolist())

    def modify_targets(self, target, kp, kd, command):
        if self.residual_policy is not None and not self.settling:
            action, _ = self.residual_policy.predict(self.residual_features(command), deterministic=True)
            self.residual = np.clip(action, -1, 1)*self.cfg.get("residual_scale", .25)
        contacts = {int(c.geom1) for c in self.data.contact} | {int(c.geom2) for c in self.data.contact}
        stable_landing = sum(f in contacts for f in self.foot_ids) >= 2 and self.rotation[2, 2] > .8
        target, kp, kd = self.skill.targets(float(self.data.time), target, kp, kd, allow_release=stable_landing)
        target = target + self.residual
        limits = self.model.jnt_range[self.joint_ids]
        target = np.clip(target, limits[:, 0], limits[:, 1])
        self.robot.last_action = (target-self.robot.default_dof_pos)/self.robot.action_scale
        return target, kp, kd

    def residual_features(self, command):
        from robogauge.tasks.gauge.goal_data import GoalData, VelocityGoal
        goal = GoalData("velocity", velocity_goal=VelocityGoal(lin_vel_x=float(command[0]), lin_vel_y=float(command[1]), ang_vel_yaw=float(command[2])))
        proprio = self.robot.build_observation(self._proprio(), goal)[:33]
        profile = np.zeros(21)
        mask = np.zeros(21)
        points = getattr(self, "level_points", np.empty((0, 3)))
        reference = self.support_z-float(self.data.qpos[2])
        for i, x in enumerate(np.linspace(-.3, 1.7, 21)):
            p = points[(np.abs(points[:, 0]-x) < .09) & (np.abs(points[:, 1]) < .35)]
            if len(p):
                profile[i] = np.clip(np.median(p[:, 2])-reference, -.8, .5)
                mask[i] = 1
        return np.clip(np.r_[proprio, profile, mask, self.residual/.25], -5, 5).astype(np.float32)

    def _control(self, command):
        command = np.asarray(command).copy()
        if self.cfg.get("auto_jump", False) and not self.settling and self.gap_edge_world is not None:
            forward = self.rotation[:2, 0]
            distance = float(np.dot(self.gap_edge_world-self.data.qpos[:2], forward))
            if .12 < distance < self.cfg.get("jump_trigger_distance", .30):
                if self.skill.trigger(float(self.data.time)):
                    self.jump_count += 1
            if distance > 0 and distance < 1.2:
                command[0] = min(command[0], self.cfg.get("jump_approach_speed", .75))
        if self.skill.active:
            command = np.array([.75, 0., 0.])
        self.executed_command = command.copy()
        super()._control(command)
        contacts = {int(c.geom1) for c in self.data.contact} | {int(c.geom2) for c in self.data.contact}
        airborne = bool(self.foot_ids) and not any(f in contacts for f in self.foot_ids)
        stance = [f for f in self.foot_ids if f in contacts]
        if stance:
            self.support_z = float(np.median(self.data.geom_xpos[stance, 2]-.022))
            required=getattr(self,'terrain_meta',{}).get('required_top_height')
            if required is not None and self.support_z>=required-.04 and len(stance)>=2 and self.rotation[2,2]>.9:
                self.visited_top=True
        self.airborne_time = self.airborne_time+self.cfg["control_dt"] if airborne else 0.
        self.max_airborne_time = max(self.max_airborne_time, self.airborne_time)
        self.height_peak = max(self.height_peak, float(self.data.qpos[2]))
        for left, right, bottom in getattr(self, "terrain_meta", {}).get("gaps", []):
            if left < self.data.qpos[0] < right and self.airborne_time >= .06 and self.skill.active:
                self.crossed_gap_airborne = True
        for x0, x1, y0, y1 in getattr(self, "terrain_meta", {}).get("water", []):
            feet = self.data.geom_xpos[self.foot_ids]
            if np.any((feet[:, 0] > x0) & (feet[:, 0] < x1) & (feet[:, 1] > y0) & (feet[:, 1] < y1)):
                self.water_entered = True

    def _ray_returns(self, origin, dirs, noise):
        rot = self.rotation
        origin = np.asarray(origin)
        p = self.data.xpos[self.base_id] + rot @ origin
        ranges = np.empty(len(dirs))
        colors = np.zeros((len(dirs), 3))
        valid = np.zeros(len(dirs), bool)
        geom = np.zeros(1, np.int32)
        for i, direction in enumerate(dirs @ rot.T):
            d = mujoco.mj_ray(self.model, self.data, p, direction, self.ray_groups, 1, -1, geom)
            valid[i] = .05 < d < self.cfg["max_range"]
            ranges[i] = d if valid[i] else self.cfg["max_range"]
            if valid[i]:
                colors[i] = self.model.geom_rgba[int(geom[0]), :3]
        valid &= self.np_random.random(len(dirs)) >= self.cfg["sensor_dropout"]
        ranges = np.clip(ranges+self.np_random.normal(0, noise, len(dirs)), .05, self.cfg["max_range"])
        ranges[~valid] = self.cfg["max_range"]
        points = origin + dirs*ranges[:, None]
        return ranges, points, colors, valid

    def _sense(self):
        lr, lp, lc, lv = self._ray_returns(self.cfg["lidar_position"], self.lidar_dirs, self.cfg["lidar_noise_std"])
        dr, dp, dc, dv = self._ray_returns(self.cfg["depth_position"], self.depth_dirs @ self.depth_rotation.T, self.cfg["depth_noise_std"])
        self.lidar_ranges = lr
        self.depth = (dr*self.depth_dirs[:, 0]).reshape(self.cfg["depth_height"], self.cfg["depth_width"])
        self.rgb = dc.reshape(self.cfg["depth_height"], self.cfg["depth_width"], 3)
        pts = [lp[lv]] if self.sensor_mode == "lidar" else [dp[dv]] if self.sensor_mode == "depth" else [lp[lv], dp[dv]]
        self.surface_points = np.concatenate(pts, axis=0)
        # Sensor colour, not geom names/semantic ground truth, marks water risk.
        blue = dv & (dc[:, 2] > .6) & (dc[:, 2] > 2*dc[:, 0]) & (dc[:, 2] > 1.5*dc[:, 1])
        self.water_points = dp[blue] if self.sensor_mode != "lidar" else np.empty((0, 3))
        yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        yaw_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        level_rotation = yaw_rotation.T @ self.rotation
        self.level_points = self.surface_points @ level_rotation.T
        self.level_water = self.water_points @ level_rotation.T
        # Estimate horizontal support from neighbouring depth returns. Vertical
        # pit-wall pixels near ground height must not masquerade as a floor rim.
        grid = dp.reshape(self.cfg["depth_height"], self.cfg["depth_width"], 3)
        normals = np.cross(np.gradient(grid, axis=1), np.gradient(grid, axis=0)).reshape(-1, 3)
        normals_level = normals @ level_rotation.T
        horizontal = np.abs(normals_level[:, 2])/np.maximum(np.linalg.norm(normals_level, axis=1), 1e-9) > .7
        self.support_returns = (dp[dv & horizontal] @ level_rotation.T) if self.sensor_mode != "lidar" else np.empty((0, 3))
        reference = self.support_z-float(self.data.qpos[2])
        obstacle = self.level_points[:, 2] > reference+self.cfg["max_climb_height"]+.03
        self.points = self.level_points[obstacle]
        self.points = np.concatenate((self.points, self.level_water), axis=0)
        if len(self.points):
            _, indices = np.unique(np.round(self.points[:, :2]/.06), axis=0, return_index=True)
            self.points = self.points[indices]
        self.scan = np.full(72, self.cfg["max_range"])
        if len(self.points):
            angle = np.arctan2(self.points[:, 1], self.points[:, 0])
            bins = np.floor((angle+np.pi)/(2*np.pi)*72).astype(int) % 72
            np.minimum.at(self.scan, bins, np.linalg.norm(self.points[:, :2], axis=1))
        forward = self.level_points[(self.level_points[:, 0] > .25) & (self.level_points[:, 0] < 2.) & (np.abs(self.level_points[:, 1]) < .27)]
        low = forward[forward[:, 2] < reference-.18]
        self.gap_distance = float(np.min(low[:, 0])) if len(low) else np.inf
        if len(low) and self.gap_edge_world is None:
            candidates = self.support_returns
            ground = candidates[(np.abs(candidates[:, 2]-reference) < .06) &
                                (candidates[:, 0] > .25) & (candidates[:, 0] < self.gap_distance-.20) &
                                (np.abs(candidates[:, 1]) < .12)]
            if len(ground):
                entry = float(np.max(ground[:, 0])+.04)
                self.gap_edge_world = self.data.qpos[:2] + yaw_rotation[:2, 0]*entry
        if self.gap_edge_world is not None:
            self.gap_distance = float(np.dot(self.gap_edge_world-self.data.qpos[:2], yaw_rotation[:2, 0]))
            if self.gap_distance < -.8:
                self.gap_edge_world = None
                self.gap_distance = np.inf
        raised = forward[(forward[:, 2] > reference+.035) & (forward[:, 2] < reference+.35)]
        self.step_distance = float(np.min(raised[:, 0])) if len(raised) else np.inf

    def _info(self):
        info = super()._info()
        goal_height = getattr(self, "terrain_meta", {}).get("goal_height", 0.)
        contacts = {int(c.geom1) for c in self.data.contact} | {int(c.geom2) for c in self.data.contact}
        stance_count = sum(f in contacts for f in self.foot_ids)
        body_contact = False
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if ((self.model.geom_bodyid[a] == self.base_id and self.model.geom_bodyid[b] == 0) or
                (self.model.geom_bodyid[b] == self.base_id and self.model.geom_bodyid[a] == 0)):
                body_contact = True
        if self.skill.active:
            # Dynamic flight may pitch beyond the walking tilt threshold. A torso
            # strike, inversion, or dropping below support still ends the trial.
            info["fallen"] = bool(body_contact or self.data.qpos[2] < self.support_z+.08 or self.rotation[2, 2] < -.1)
        info["success"] = bool(info["success"] and self.data.qpos[2] > goal_height+.20 and self.rotation[2, 2] > .75 and stance_count >= 2)
        info.update(water_entered=self.water_entered, jumps=self.jump_count,
                    max_airborne_time=self.max_airborne_time, height_peak=self.height_peak,
                    crossed_gap_airborne=self.crossed_gap_airborne,
                    gap_distance=None if not np.isfinite(getattr(self, "gap_distance", np.inf)) else self.gap_distance,
                    step_distance=None if not np.isfinite(getattr(self, "step_distance", np.inf)) else self.step_distance,
                    executed_command=getattr(self, "executed_command", self.command).tolist(),
                    stance_feet=stance_count, body_contact=body_contact,
                    locomotion=self.cfg.get("locomotion", "mlp"))
        if self.water_entered:
            info["success"] = False
        return info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.cfg.get('terminal_hold_s',0.):
            # Only a reached goal may extend the trial; collisions/falls always terminate.
            candidate=bool(info['success'] and (not self.cfg.get('terrain_return') or self.visited_top))
            stable=candidate and np.linalg.norm(self.data.qvel[:2])<.20 and np.linalg.norm(self.data.qvel[3:6])<.5 and self.rotation[2,2]>.98
            self.terminal_stable_s=self.terminal_stable_s+self.cfg['navigation_dt'] if stable else 0.
            info['success']=bool(candidate and self.terminal_stable_s>=self.cfg['terminal_hold_s'])
            info.update(visited_top=self.visited_top,terminal_stable_s=self.terminal_stable_s)
            terminated=bool(info['success'] or info['fallen'] or info['collision'])
        if self.water_entered:
            reward -= 20
            terminated = True
        return obs, reward, terminated, truncated, info
