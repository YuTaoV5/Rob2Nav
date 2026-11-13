"""Independent R284 sliding dynamics and takeover experiments."""
import torch
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from as2w_isaac_flat_env_all import FlatEnvironment, FlatCommands, configuration, observation

POSE = {"pose", "pose-delay", "combined", "expanded"}
DELAY = {"delay", "pose-delay", "combined", "expanded"}


class TakeoverCommands(FlatCommands):
    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self._env.episode_length_buf < 100] = 0.


def reset_joints(env, env_ids, position_range, velocity_range):
    robot = env.scene["robot"]
    q = robot.data.default_joint_pos.torch[env_ids].clone()
    dq = torch.zeros_like(q)
    legs = env.as2w_joint_ids[:12]
    wheels = env.as2w_joint_ids[12:]
    draw = torch.rand((len(q), 1), device=env.device)
    scale = torch.where(draw < .30, .06, torch.where(draw < .70, .15, .05))
    perturb = (2. * torch.rand_like(q[:, legs]) - 1.) * scale
    flex = torch.rand_like(draw) * .04 + .085
    # A correlated squat/extension proxy until calibrated LowState is available.
    flex *= torch.where(torch.rand_like(draw) < .5, -1., 1.) * (draw >= .70)
    perturb[:, 4:8] += flex
    perturb[:, 8:12] -= 2. * flex
    q[:, legs] += perturb
    q[:, wheels] += (torch.rand_like(q[:, wheels]) - .5) * (4. * torch.pi) * (draw >= .70)
    dq[:, legs] = (torch.rand_like(dq[:, legs]) - .5) * .30 * (draw >= .30)
    limits = robot.data.soft_joint_pos_limits.torch[env_ids]
    q[:, legs] = q[:, legs].clamp(limits[:, legs, 0], limits[:, legs, 1])
    robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)


class DelayedTarget:
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._history = torch.zeros((5, *self.processed_actions.shape), device=env.device)
        self._lag = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._rows = torch.arange(env.num_envs, device=env.device)
        self._cursor = 0

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._lag[ids] = torch.randint(5, self._lag[ids].shape, device=self._lag.device)
        initial = self._initial_target(ids)
        self._history[:, ids] = initial

    def delayed(self):
        self._history[self._cursor] = self.processed_actions
        value = self._history[(self._cursor - self._lag) % 5, self._rows]
        self._cursor = (self._cursor + 1) % 5
        return value


class DelayedPosition(DelayedTarget, JointPositionAction):
    def _initial_target(self, ids):
        return self._asset.data.joint_pos.torch[ids][:, self._joint_ids]

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self.delayed(), joint_ids=self._joint_ids)


class PredictiveDelayedPosition(DelayedPosition):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        limits = env.cfg.as2w_safety["angle_rad"]
        bounds = self.processed_actions.new_tensor([limits[name] for name in self._joint_names])
        self._lower,self._upper = bounds.unbind(dim=1)
        self._margin = getattr(env.cfg, "as2w_predictive_target_margin", .05)

    def apply_actions(self):
        from as2w_predictive_joint_guard import guarded_target

        data = self._asset.data
        target = guarded_target(self.delayed(),data.joint_pos.torch[:,self._joint_ids],
                                data.joint_vel.torch[:,self._joint_ids],self._lower,self._upper,margin=self._margin)
        self._asset.set_joint_position_target_index(target=target,joint_ids=self._joint_ids)


class DelayedVelocity(DelayedTarget, JointVelocityAction):
    def _initial_target(self, ids):
        return torch.zeros_like(self.processed_actions[ids])

    def apply_actions(self):
        self._asset.set_joint_velocity_target_index(target=self.delayed(), joint_ids=self._joint_ids)


class RobustEnvironment(FlatEnvironment):
    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if hasattr(self, "policy_action"):
            self.policy_action[ids] = 0.
            self.observation_ready[ids] = False
        if hasattr(self, "hold_action"):
            robot = self.scene["robot"]
            legs = self.as2w_joint_ids[:12]
            self.hold_action[ids, :12] = (
                robot.data.joint_pos.torch[ids][:, legs] -
                robot.data.default_joint_pos.torch[ids][:, legs]
            ) / .35
            self.observation_lag[ids] = torch.randint(2, self.observation_lag[ids].shape,
                                                     device=self.device)

    def step(self, action):
        if self.cfg.robust_profile == "baseline":
            return super().step(action)
        if not hasattr(self, "policy_action"):
            self.policy_action = torch.zeros_like(action)
            self.hold_action = torch.zeros_like(action)
            self.observation_lag = torch.randint(2, (self.num_envs,), device=self.device)
            robot = self.scene["robot"]
            legs = self.as2w_joint_ids[:12]
            self.hold_action[:, :12] = (
                robot.data.joint_pos.torch[:, legs] -
                robot.data.default_joint_pos.torch[:, legs]
            ) / .35
        self.policy_action.copy_(action)
        if self.cfg.robust_profile != "baseline":
            alpha = (self.episode_length_buf * self.step_dt / 2.).clamp(max=1.)[:, None]
            action = self.hold_action.lerp(action, alpha)
        return super().step(action)


def robust_observation(env):
    frame = observation(env)
    profile = env.cfg.robust_profile
    if profile == "baseline":
        return frame
    if hasattr(env, "policy_action"):
        frame[:, 42:58] = env.policy_action
    measured = torch.cat((frame[:, 3:9], frame[:, 10:42]), dim=1)
    if profile in {"combined", "expanded"}:
        measured[:, :3] += .015 * torch.randn_like(measured[:, :3])
        measured[:, 6:18] += .004 * torch.randn_like(measured[:, 6:18])
        measured[:, 22:] += .04 * torch.randn_like(measured[:, 22:])
    if not hasattr(env, "observation_ready"):
        env.observation_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.previous_measurement = measured.clone()
        env.current_measurement = measured.clone()
        env.measurement_tick = -1
        env.reference = frame[:, 22:26].clone()
        env.previous_zero = torch.zeros_like(env.observation_ready)
    tick = env._sim_step_counter
    if tick != env.measurement_tick:
        env.previous_measurement.copy_(env.current_measurement)
        env.current_measurement.copy_(measured)
        env.measurement_tick = tick
    fresh = ~env.observation_ready | (env.episode_length_buf == 0)
    env.current_measurement[fresh] = measured[fresh]
    env.previous_measurement[fresh] = measured[fresh]
    env.observation_ready.fill_(True)
    if profile == "expanded":
        if not hasattr(env, "bias"):
            env.bias = torch.zeros_like(measured)
            env.delivered = measured.clone()
            env.dropped = torch.zeros_like(fresh)
        env.bias[fresh, :3] = .02 * (2. * torch.rand_like(env.bias[fresh, :3]) - 1.)
        env.bias[fresh, 6:22] = .01 * (2. * torch.rand_like(env.bias[fresh, 6:22]) - 1.)
    if profile in DELAY and hasattr(env, "observation_lag"):
        measured = torch.where(env.observation_lag[:, None] > 0,
                               env.previous_measurement, env.current_measurement)
    else:
        measured = env.current_measurement
    if profile == "expanded":
        # One missed policy packet holds the whole sensor frame coherently.
        miss = (torch.rand_like(fresh, dtype=torch.float32) < .02) & ~env.dropped & ~fresh
        env.delivered.copy_(torch.where(miss[:, None], env.delivered, measured + env.bias))
        env.dropped.copy_(miss)
        measured = env.delivered
    frame[:, 3:9] = measured[:, :6]
    frame[:, 10:42] = measured[:, 6:]
    wheel = frame[:, 22:26]
    zero = frame[:, 64] > .5
    entered = zero & (~env.previous_zero | fresh)
    env.reference[entered] = wheel[entered]
    env.previous_zero.copy_(zero)
    delta = (wheel - env.reference + 2. * torch.pi) % (4. * torch.pi) - 2. * torch.pi
    frame[:, 60:64] = (delta / .20).clamp(-3., 3.) * zero[:, None]
    return frame


def robust_config(args, physics):
    cfg = configuration(args, physics)
    cfg.robust_profile = args.profile
    if args.profile != "baseline":
        cfg.commands.base_velocity.class_type = TakeoverCommands
        cfg.observations.policy.state.func = robust_observation
    if args.profile in POSE:
        cfg.events.reset_robot_joints.func = reset_joints
        cfg.events.reset_base.params["pose_range"] = {
            "z": (-.02, .02), "roll": (-.052, .052), "pitch": (-.052, .052)
        }
    if args.profile in DELAY:
        cfg.actions.joint_pos.class_type = DelayedPosition
        cfg.actions.wheel_vel.class_type = DelayedVelocity
    if args.profile in {"friction", "combined", "expanded"}:
        cfg.events.wheel_friction = EventTermCfg(
            func=mdp.randomize_rigid_body_material, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (.8, 1.2), "dynamic_friction_range": (.64, .80),
                "restitution_range": (0., 0.), "num_buckets": 32,
            })
    if args.profile in {"combined", "expanded"}:
        cfg.events.base_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "mass_distribution_params": (.95, 1.05), "operation": "scale"})
        cfg.events.base_com = EventTermCfg(func=mdp.randomize_rigid_body_com, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "com_range": {"x": (-.01, .01), "y": (-.01, .01)}})
        cfg.events.gains = EventTermCfg(func=mdp.randomize_actuator_gains, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (.9, 1.1),
            "damping_distribution_params": (.9, 1.1), "operation": "scale",
        })
    return cfg
