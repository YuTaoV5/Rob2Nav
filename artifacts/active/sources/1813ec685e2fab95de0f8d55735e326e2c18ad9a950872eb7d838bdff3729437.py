"""Perception-conditioned swing-foot clearance and optional NJU BBC skills.

Only ray returns, proprioception and local odometry reach the controller. Scene
metadata is restricted to inherited evaluation. Cached points expire in seconds;
no occupancy map or prescribed course waypoints are used.
"""
import numpy as np
import mujoco
from .terrain_env import TerrainEnv, terrain_config


class ActiveTerrainEnv(TerrainEnv):
    def __init__(self, config=None, **kwargs):
        self.cloud_memory = []
        self.support_memory = []
        self.landing_confirmed = False
        self.estimated_gap_width = None
        self.jump_guard = None
        self.auth_gap_edge = None
        self.last_deep_seen = -100.
        self.runup_stage = None
        self.runup_yaw = 0.
        self.runup_since = 0.
        self.visited = []
        self.lift_events = []
        self.bbc = None
        self.bbc_start = None
        self.last_bbc_end = -100.
        self.bbc_active = False
        self.bbc_approaching = False
        self.bbc_approach_latch = False
        self.foot_lift = np.zeros(4)
        self.anticipation = np.zeros(4)
        self.last_nominal_feet = None
        self.step_seen = False
        self.riser_impulse = 0.
        self.first_riser_contact = None
        self.lift_count = 0
        cfg = terrain_config() if config is None else dict(config)
        cfg["auto_jump"] = cfg.get("jump_backend", "primitive") == "primitive"
        if cfg.get("jump_backend") in ("nju", "nju_all", "nju_runup"):
            from .nju_bbc import NJUBBC
            self.bbc = NJUBBC(cfg.get("bbc_modes", 3))
        super().__init__(cfg, **kwargs)

    def reset(self, **kwargs):
        self.cloud_memory = []
        self.support_memory = []
        self.landing_confirmed = False
        self.estimated_gap_width = None
        self.jump_guard = None
        self.auth_gap_edge = None
        self.last_deep_seen = -100.
        self.runup_stage = None
        self.runup_since = 0.
        self.visited = []
        self.lift_events = []
        self.last_nominal_feet = None
        self.foot_lift = np.zeros(4)
        self.anticipation = np.zeros(4)
        self.step_seen = False
        self.riser_impulse = 0.
        self.first_riser_contact = None
        self.lift_count = 0
        self.bbc_start = None
        self.last_bbc_end = -100.
        self.bbc_active = False
        self.bbc_approaching = False
        self.bbc_approach_latch = False
        if self.bbc:
            self.bbc.reset()
        return super().reset(**kwargs)

    def _world(self):
        super()._world()
        if self.bbc is not None:
            # RoboGauge intentionally narrows thigh limits for its benchmark.
            # Restore the ranges in NJU's Go2 URDF for the BBC experiment.
            for leg in range(4):
                self.model.jnt_range[self.joint_ids[leg*3+1]] = [-1.5708, 3.4907] if leg < 2 else [-.5236, 4.5379]
        self.kinematics = mujoco.MjData(self.model)

    def _control(self, command):
        command = np.asarray(command).copy()
        auto = self.cfg.get("auto_jump", False)
        edge = getattr(self, "gap_edge_world", None)
        distance = getattr(self, "gap_distance", np.inf)
        self.jump_guard = None
        now = float(self.data.time)
        if not self.settling:
            if not self.visited or np.linalg.norm(self.data.qpos[:2]-self.visited[-1][:2]) > .1:
                self.visited.append(self.data.qpos[:3].copy())
            self.visited = self.visited[-80:]
        if self.cfg.get("prepare_runup", False) and self.bbc is not None and not self.settling:
            if self.runup_stage is None and self.jump_count == 0 and self.landing_confirmed and .5 < distance < 1.5:
                forward = self.rotation[:2, 0]
                safe = [p for p in self.visited if np.dot(self.data.qpos[:2]-p[:2], forward) > 1.6
                        and abs(np.cross(forward, self.data.qpos[:2]-p[:2])) < .30
                        and abs(p[2]-self.data.qpos[2]) < .06]
                if safe:
                    self.runup_stage, self.runup_since = "retreat", now
                    self.runup_yaw = float(np.arctan2(forward[1], forward[0]))
                    self.lift_events.append(dict(event="prepare_runup", time=now, distance=distance))
            yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
            yaw_error = np.arctan2(np.sin(self.runup_yaw-yaw), np.cos(self.runup_yaw-yaw))
            if self.runup_stage == "retreat":
                command = np.array([-.6, 0., np.clip(2*yaw_error, -.5, .5)])
                rear = self.points[(self.points[:, 0] > -.9) & (self.points[:, 0] < .0) &
                                   (np.abs(self.points[:, 1]) < self.cfg["robot_radius"]+.1)]
                if not np.isfinite(distance) or len(rear):
                    self.runup_stage = "abort"
                elif distance > self.cfg.get("runup_distance", 2.7):
                    self.runup_stage, self.runup_since = "brake", now
                if now-self.runup_since > 10.:
                    self.runup_stage = "abort"
            if self.runup_stage == "brake":
                command = np.array([0., 0., np.clip(2*yaw_error, -.5, .5)])
                if now-self.runup_since > .6:
                    self.runup_stage = "runup"
                    self.bbc_approach_latch = True
            if self.runup_stage == "runup":
                command = np.array([self.cfg.get("bbc_approach_speed", 1.2), 0., np.clip(2*yaw_error, -.5, .5)])
            if self.runup_stage == "abort":
                command[:] = 0.
        if not self.settling and edge is not None and not (self.skill.active or self.bbc_active):
            if self.cfg.get("require_landing", True) and not self.landing_confirmed:
                self.cfg["auto_jump"] = False
                if 0 < distance < .9:
                    self.jump_guard = "unobserved_or_out_of_range_landing"
                    command[:] = 0.
            elif .1 < distance < 1.3:
                # Approach at calibrated speed with reduced turning, not a new
                # action selected halfway through a committed flight.
                command[0] = min(command[0], self.cfg.get("jump_approach_speed", .75))
                command[1] = 0.
                command[2] = np.clip(command[2], -.25, .25)
        super()._control(command)
        self.cfg["auto_jump"] = auto
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if any(f in (c.geom1, c.geom2) for f in self.foot_ids) and abs(c.frame[2]) < .5:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                if wrench[0] > 2:
                    self.riser_impulse += float(wrench[0])*self.cfg["control_dt"]
                    if self.first_riser_contact is None:
                        self.first_riser_contact = float(self.data.time)
        # Ground-truth gap bounds are consulted only by the evaluator.
        if self.bbc_active and self.airborne_time >= .06:
            for left, right, _ in getattr(self, "terrain_meta", {}).get("gaps", []):
                if left < self.data.qpos[0] < right:
                    self.crossed_gap_airborne = True

    def _sense(self):
        super()._sense()
        now = float(self.data.time)
        # Geometry originates exclusively from visible sensor returns.
        yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        r = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        reference = self.support_z-float(self.data.qpos[2])
        p = self.level_points
        deep = p[(p[:, 0] > .25) & (p[:, 0] < 3.) & (np.abs(p[:, 1]) < .27) &
                 (p[:, 2] < reference-self.cfg.get("minimum_pit_depth", .40))]
        negative = p[(p[:, 0] > .25) & (p[:, 0] < 3.) & (np.abs(p[:, 1]) < .27) & (p[:, 2] < reference-.18)]
        support = self.support_returns
        lower_floor = support[(support[:, 0] > .25) & (support[:, 0] < 3.) &
                              (np.abs(support[:, 1]) < .27) & (support[:, 2] < reference-.12)]
        # Descending a few stairs is not a pit. A negative step alone must not
        # authorize jumping or create an enduring false cliff in the cache.
        continuous_lower_floor = len(lower_floor) >= 15 and np.ptp(lower_floor[:, 0]) > .50
        pit_visible = len(deep) >= 3 or (len(negative) >= 3 and not continuous_lower_floor)
        if pit_visible and self.gap_edge_world is not None:
            self.auth_gap_edge = self.gap_edge_world.copy()
            self.last_deep_seen = now
        if self.auth_gap_edge is not None:
            distance = float(np.dot(self.auth_gap_edge-self.data.qpos[:2], r[:2, 0]))
            if distance < -.8 or now-self.last_deep_seen > self.cfg.get("pit_memory_seconds", 5.):
                self.auth_gap_edge = None
        self.gap_edge_world = self.auth_gap_edge
        self.gap_distance = np.inf if self.auth_gap_edge is None else float(np.dot(self.auth_gap_edge-self.data.qpos[:2], r[:2, 0]))
        cloud = self.level_points
        cloud = cloud[(cloud[:, 0] > -.6) & (cloud[:, 0] < 2.5) & (np.abs(cloud[:, 1]) < 1.2)]
        if len(cloud):
            _, indices = np.unique(np.round(cloud/.04), axis=0, return_index=True)
            self.cloud_memory.append((now, cloud[indices] @ r.T+self.data.qpos[:3]))
        self.cloud_memory = [(t, p) for t, p in self.cloud_memory if now-t <= self.cfg.get("memory_seconds", 1.5)]
        self.measured_cloud = np.concatenate([p for _, p in self.cloud_memory]) if self.cloud_memory else np.empty((0, 3))
        supports = self.support_returns
        if len(supports):
            self.support_memory.append((now, supports @ r.T+self.data.qpos[:3]))
        self.support_memory = [(t, p) for t, p in self.support_memory if now-t <= 1.5]
        if self.gap_edge_world is not None and self.support_memory:
            observed = np.concatenate([p for _, p in self.support_memory])
            xy = (observed[:, :2]-self.gap_edge_world) @ r[:2, :2]
            flat = (np.abs(observed[:, 2]-self.support_z) < .045) & (np.abs(xy[:, 1]) < .3)
            far = xy[flat & (xy[:, 0] > .10)]
            self.estimated_gap_width = float(np.percentile(far[:, 0], 5)) if len(far) >= 10 else None
            self.landing_confirmed = bool(self.estimated_gap_width is not None and
                self.estimated_gap_width <= self.cfg.get("max_jump_width", .40))
        else:
            self.estimated_gap_width = None
            self.landing_confirmed = False
        if np.isfinite(self.step_distance) and not self.step_seen:
            self.step_seen = True
            self.lift_events.append(dict(event="step_observed", time=now, distance=self.step_distance))

    def modify_targets(self, target, kp, kd, command):
        if self.bbc is not None:
            backend = self.cfg.get("jump_backend")
            now = float(self.data.time)
            distance = getattr(self, "gap_distance", np.inf)
            if not self.settling and self.bbc_start is None and now-self.last_bbc_end > 2.:
                permitted = (not self.cfg.get("require_landing", True) or self.landing_confirmed) and self.runup_stage not in ("retreat", "brake", "abort")
                if permitted and .1 < distance < self.cfg.get("jump_trigger_distance", .55):
                    self.bbc_start = now
                    self.jump_count += 1
                    self.lift_events.append(dict(event="nju_jump_requested", time=now, distance=distance))
            duration = 0 if self.bbc_start is None else now-self.bbc_start
            self.bbc_active = self.bbc_start is not None
            capture = bool(backend == "nju" and not self.bbc_active and
                self.landing_confirmed and .1 < distance < self.cfg.get("bbc_capture_distance", 2.) and
                now-self.last_bbc_end > 2. and self.runup_stage not in ("retreat", "brake", "abort"))
            self.bbc_approach_latch = self.bbc_approach_latch or capture
            self.bbc_approaching = self.bbc_approach_latch and not self.bbc_active
            # Run the observation/history path continuously, also while MoE controls.
            prepare = self.cfg.get("bbc_prepare_time", .0)
            jump = self.bbc_active and prepare <= duration < prepare+self.cfg.get("bbc_jump_command_time", .35)
            velocity = np.asarray(command).copy()
            runup = backend == "nju_runup" and (self.jump_count == 0 or self.bbc_active)
            if backend == "nju_all" or runup or self.bbc_approaching:
                remaining = np.linalg.norm(self.goal-self.data.qpos[:2])
                velocity[0] = min(self.cfg.get("bbc_approach_speed", velocity[0]), max(.0, remaining-.3))
            if self.bbc_active:
                velocity = np.array([self.cfg.get("bbc_speed", 1.), 0., 0.])
            elif self.jump_guard:
                velocity[:] = 0.
            nju_target, nju_kp, nju_kd = self.bbc.targets(self, velocity, jump=jump,
                height=self.cfg.get("bbc_jump_height", .5), epsilon=self.cfg.get("bbc_epsilon", 0.))
            if backend == "nju_all" or runup or self.bbc_active or self.bbc_approaching:
                target, kp, kd = nju_target, nju_kp, nju_kd
            if self.bbc_active and duration > self.cfg.get("bbc_duration", .85):
                contacts = {int(c.geom1) for c in self.data.contact} | {int(c.geom2) for c in self.data.contact}
                if sum(f in contacts for f in self.foot_ids) >= 2 and self.rotation[2, 2] > .8:
                    self.bbc_start = None
                    self.bbc_active = False
                    self.bbc_approach_latch = False
                    self.last_bbc_end = now
                    if self.runup_stage == "runup":
                        self.runup_stage = "complete"
        if not self.settling and self.cfg.get("anticipatory_lift", True) and not (self.skill.active or self.bbc_active or self.bbc_approaching):
            target = self._lift_swing_feet(target)
        return super().modify_targets(target, kp, kd, command)

    def _lift_swing_feet(self, target):
        k = self.kinematics
        k.qpos[:] = self.data.qpos
        k.qpos[self.qids] = target
        mujoco.mj_forward(self.model, k)
        feet = k.geom_xpos[self.foot_ids].copy()
        velocity_z = np.zeros(4) if self.last_nominal_feet is None else (feet[:, 2]-self.last_nominal_feet[:, 2])/self.cfg["control_dt"]
        self.last_nominal_feet = feet.copy()
        actual = self.data.geom_xpos[self.foot_ids]
        forward = self.rotation[:2, 0]
        lateral = self.rotation[:2, 1]
        cloud = getattr(self, "measured_cloud", np.empty((0, 3)))
        lifted = target.copy()
        contacts = {int(c.geom1) for c in self.data.contact} | {int(c.geom2) for c in self.data.contact}
        for leg in range(4):
            delta = cloud[:, :2]-actual[leg, :2]
            dx, dy = delta @ forward, delta @ lateral
            candidates = cloud[(dx > .03) & (dx < self.cfg.get("foot_lookahead", .42)) & (np.abs(dy) < .09)]
            rise = 0.
            if len(candidates):
                raised = candidates[candidates[:, 2] > self.support_z+.045]
                tall = np.count_nonzero(candidates[:, 2] > self.support_z+self.cfg.get("max_lift_step", .22))
                if len(raised) >= 8 and tall < 5:
                    rise = float(np.percentile(raised[:, 2], 25)-self.support_z)
            self.anticipation[leg] = rise if .025 < rise < self.cfg.get("max_lift_step", .22) else 0.
            swing = self.foot_ids[leg] not in contacts and (velocity_z[leg] > .005 or feet[leg, 2] > self.support_z+.055)
            desired = min(max(0., self.support_z+self.anticipation[leg]+.035-feet[leg, 2]),
                          self.cfg.get("max_foot_lift", .05)) if swing and self.anticipation[leg] else 0.
            self.foot_lift[leg] += np.clip(desired-self.foot_lift[leg], -.025, .025)
            if self.foot_ids[leg] in contacts:
                self.foot_lift[leg] = 0.
            if self.foot_lift[leg] < .001:
                continue
            if desired > .01 and len(self.lift_events) < 300:
                self.lift_events.append(dict(event="anticipatory_lift", time=float(self.data.time),
                    leg=leg, rise=rise, lift=float(self.foot_lift[leg]), foot_x=float(actual[leg, 0])))
            if desired > .01:
                self.lift_count += 1
            # Damped Jacobian IK evaluated at the nominal pose, joint-local only.
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacGeom(self.model, k, jp, jr, self.foot_ids[leg])
            cols = self.vids[leg*3:leg*3+3]
            j = jp[:, cols]
            dz = np.array([0., 0., self.foot_lift[leg]])
            dq = j.T @ np.linalg.solve(j @ j.T+.001*np.eye(3), dz)
            lifted[leg*3:leg*3+3] += np.clip(dq, -.45, .45)
        return lifted

    def _info(self):
        info = super()._info()
        if not self.skill.active and not self.bbc_active and info["body_contact"]:
            info["fallen"] = True
        if self.bbc_active:
            body_hit = any((self.model.geom_bodyid[c.geom1] == self.base_id and self.model.geom_bodyid[c.geom2] == 0)
                           or (self.model.geom_bodyid[c.geom2] == self.base_id and self.model.geom_bodyid[c.geom1] == 0)
                           for c in self.data.contact)
            info["fallen"] = bool(body_hit or self.rotation[2, 2] < -.1 or self.data.qpos[2] < self.support_z+.08)
        first_lift = next((e["time"] for e in self.lift_events if e["event"] == "anticipatory_lift"), None)
        info.update(anticipatory_lifts=self.lift_count, riser_contact_impulse=self.riser_impulse,
                    first_riser_contact=self.first_riser_contact, first_lift=first_lift,
                    landing_confirmed=self.landing_confirmed, estimated_gap_width=self.estimated_gap_width,
                    jump_guard=self.jump_guard,
                    runup_stage=self.runup_stage,
                    nju_active=self.bbc_active,
                    foot_lift=self.foot_lift.tolist())
        return info
