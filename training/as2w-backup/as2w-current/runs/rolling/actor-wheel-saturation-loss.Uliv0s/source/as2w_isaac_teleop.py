"""Joystick command curricula for the AS2-W sliding policy."""
import torch
from isaaclab.managers import RewardTermCfg
from as2w_isaac_robust_env import TakeoverCommands


class TeleopCommands(TakeoverCommands):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.kind = torch.zeros(env.num_envs, device=self.device)
        self.ramp = torch.zeros_like(self.kind)
        self.ramp_start = torch.zeros_like(self.targets)
        self.previous_target = torch.zeros_like(self.targets)
        self.park_target = torch.zeros_like(self.targets)

    def sample_motion(self, shape):
        values = 2. * torch.rand(shape, device=self.device) - 1.
        planar = values[:, :2]
        planar /= planar.norm(dim=1, keepdim=True).clamp_min(1.)
        planar *= .10
        values[:, 2] *= .30
        mode = torch.rand(len(values), device=self.device)
        straight = mode < .30
        main_axis = planar.abs().argmax(dim=1)
        planar[straight, 1 - main_axis[straight]] = 0.
        values[straight, 2] = 0.
        planar[mode > .90] = 0.
        return values

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.kind[ids] = torch.rand_like(self.kind[ids])
        self.previous_target[ids] = 0.
        self.park_target[ids] = self.sample_motion(self.targets[ids].shape)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        self.ramp_start[env_ids] = self.vel_command_b[env_ids]
        n = self.targets[env_ids].shape[0]
        u = torch.rand((n, 3), device=self.device)
        old = self.previous_target[env_ids]
        super()._resample_command(env_ids)
        move = self.sample_motion(self.targets[env_ids].shape)
        move = torch.where((u[:, :1] < .20) & old.abs().amax(dim=1, keepdim=True).gt(.015),
                           -old, move)
        move = torch.where(u[:, 1:2] < .10, 0., move)
        mixed = (self.kind[env_ids] >= .40) & (self.kind[env_ids] < .80)
        self.targets[env_ids] = torch.where(mixed[:, None], move, self.targets[env_ids])
        self.previous_target[env_ids] = self.targets[env_ids]
        self.ramp[env_ids] = torch.where(mixed & (u[:, 2] < .50), .2 + .65 * u[:, 1], 0.)

    def _update_command(self):
        elapsed = self._env.episode_length_buf * self._env.step_dt
        segment = self.segment_duration - self.time_left
        fraction = (segment / self.ramp.clamp_min(.001)).clamp(0., 1.)
        goal = self.ramp_start.lerp(self.targets, fraction[:, None])
        goal = torch.where(self.ramp[:, None] > 0., goal, self.targets)
        park = self.kind >= .80
        park_until = getattr(self._env.cfg,"as2w_park_until_s",5.)
        park_goal = torch.where((elapsed < park_until)[:, None], self.park_target, 0.)
        goal = torch.where(park[:, None], park_goal, goal)
        goal[elapsed < 2.] = 0.
        if getattr(self._env.cfg,"as2w_ignore_lateral_command",False):
            goal[:,1] = 0.
            self.vel_command_b[:,1] = 0.
        # Apply the same centered-stick handling and 50 Hz filter as deployment.
        goal[:, :2] *= goal[:, :2].norm(dim=1, keepdim=True) >= .008
        goal[:, 2] *= goal[:, 2].abs() >= .024
        self.vel_command_b.lerp_(goal, .12)
        settled = (goal == 0.).all(dim=1) & (self.vel_command_b.abs().amax(dim=1) < .005)
        self.vel_command_b[settled | (elapsed < 2.)] = 0.

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if not hasattr(self, "segment_duration"):
            self.segment_duration = torch.zeros_like(self.time_left)
        self.segment_duration[env_ids] = self.time_left[env_ids]


def axis_error(env):
    command = env.command_manager.get_command("base_velocity")
    robot = env.scene["robot"]
    xy = robot.data.root_lin_vel_b.torch[:, :2]
    yaw = robot.data.root_ang_vel_b.torch[:, 2]
    speed = command[:, :2].norm(dim=1)
    lateral = (xy[:, 0] * command[:, 1] - xy[:, 1] * command[:, 0]).abs()
    lateral = lateral / speed.clamp_min(.015) / .10 * (speed >= .015)
    bad_turn = (-yaw * command[:, 2].sign()).clamp_min(0.) / .30
    return lateral.clamp(max=3.) + 1.5 * bad_turn.clamp(max=3.) * (command[:, 2].abs() >= .025)


def parking_error(env):
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    position = robot.data.root_pos_w.torch[:, :2]
    x, y, z, w = robot.data.root_quat_w.torch.unbind(dim=1)
    heading = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
    zero = (command == 0.).all(dim=1) & (env.episode_length_buf > 100)
    if not hasattr(env, "park_ticks"):
        env.park_ticks = torch.zeros_like(env.episode_length_buf)
        env.park_position = position.clone()
        env.park_heading = heading.clone()
    env.park_ticks = torch.where(zero, env.park_ticks + 1, 0)
    settle_ticks = round(1. / env.step_dt)
    fresh = env.park_ticks <= settle_ticks
    env.park_position[fresh] = position[fresh]
    env.park_heading[fresh] = heading[fresh]
    drift = (position - env.park_position).norm(dim=1) / .02
    yaw = ((heading - env.park_heading + torch.pi) % (2. * torch.pi) - torch.pi).abs() / .05236
    speed = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1) / .02
    return zero * (.4 * speed.clamp(max=4.) + drift.clamp(max=5.) + yaw.clamp(max=5.))


def target_rate(env):
    manager = env.action_manager
    change = (manager.action - manager.prev_action).abs()
    return (change - .18).clamp_min(0.).square().mean(dim=1)


def configure_teleop(cfg, arm):
    cfg.episode_length_s = 17.
    cfg.commands.base_velocity.class_type = TeleopCommands
    cfg.commands.base_velocity.resampling_time_range = (.65, 2.6)
    if arm == "b":
        cfg.rewards.teleop_axis_error = RewardTermCfg(func=axis_error, weight=-.55)
        cfg.rewards.teleop_park_error = RewardTermCfg(func=parking_error, weight=-.24)
        cfg.rewards.teleop_target_rate = RewardTermCfg(func=target_rate, weight=-.12)