"""Expand the AS2-W sliding command radius without dropping low-speed skills."""
import torch
from isaaclab.managers import RewardTermCfg
from as2w_isaac_teleop import TeleopCommands, configure_teleop, axis_error
from as2w_isaac_robust_env import RobustEnvironment


class SpeedRangeCommands(TeleopCommands):
    def sample_motion(self, shape):
        low = super().sample_motion(shape)
        n = shape[0]
        # Equal probability for the cardinal and diagonal directions.
        angle = torch.randint(8, (n,), device=self.device) * (torch.pi / 4.)
        direction = torch.stack((angle.cos(), angle.sin()), dim=1)
        scale = torch.rand(n, device=self.device)
        speed = .10 + (self._env.cfg.teleop_speed_cap - .10) * scale
        fast = torch.cat((direction * speed[:, None], torch.zeros_like(speed[:, None])), dim=1)
        return torch.where((torch.rand(n, 1, device=self.device) < .68), fast, low)

    def _update_command(self):
        previous = self.vel_command_b.clone()
        super()._update_command()
        planar = self.vel_command_b[:, :2]
        limit = 1.2 * self._env.step_dt
        delta = planar - previous[:, :2]
        size = delta.norm(dim=1, keepdim=True)
        high = torch.maximum(planar.norm(dim=1), previous[:, :2].norm(dim=1)) > .10
        planar[:] = torch.where(high[:, None], previous[:, :2] +
                               delta * (limit / size.clamp_min(limit)), planar)
        yaw_scale = ((.18 - planar.norm(dim=1)) / .08).clamp(0., 1.)
        if getattr(self._env.cfg,"as2w_rolling_maneuvers",False):
            carried = .88*previous[:,2]
            self.vel_command_b[:,2] = carried+yaw_scale*(self.vel_command_b[:,2]-carried)
        else:
            self.vel_command_b[:,2] *= yaw_scale
        start = getattr(self._env.cfg,"as2w_motion_start_s",2.)
        self.vel_command_b[self._env.episode_length_buf * self._env.step_dt < start] = 0.


def speed_axis_error(env):
    cmd = env.command_manager.get_command("base_velocity")
    speed = cmd[:, :2].norm(dim=1)
    robot = env.scene["robot"]
    xy = robot.data.root_lin_vel_b.torch[:, :2]
    yaw = robot.data.root_ang_vel_b.torch[:, 2]
    error = (xy - cmd[:, :2]).norm(dim=1) / speed.clamp_min(.10)
    wrong = (-(xy * cmd[:, :2]).sum(dim=1) / speed.clamp_min(.10).square()).clamp_min(0.)
    high_error = error.clamp(max=3.) + wrong.clamp(max=2.) + .5 * yaw.abs().clamp(max=2.)
    return torch.where(speed > .10, high_error, axis_error(env))


class RollingSpeedCommands(SpeedRangeCommands):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.goals = torch.cat((self.goals[:2], self.goals[-2:]))

    def sample_motion(self, shape):
        if getattr(self._env.cfg, "as2w_rolling_low_speed_smooth", False):
            draw = torch.rand((shape[0], 3), device=self.device)
            sign = torch.where(draw[:, 0] < .5, -1., 1.)
            cmd = torch.zeros(shape, device=self.device)
            turn = draw[:, 2] < .40
            arc = (draw[:, 2] >= .40) & (draw[:, 2] < .55)
            cmd[:, 0] = torch.where(turn, 0., sign * (.03 + .17 * draw[:, 1]))
            cmd[:, 2] = torch.where(turn | arc, sign * (.035 + .085 * draw[:, 1]), 0.)
            cmd[arc, 0] = .05
            return cmd
        low = TeleopCommands.sample_motion(self, shape)
        low[:, 1] = 0.
        draw = torch.rand((shape[0], 3), device=self.device)
        speed = .10 + (self._env.cfg.teleop_speed_cap - .10) * draw[:, 0]
        signed = torch.where(draw[:, 1] < .50, -speed, speed)
        fast = torch.zeros_like(low)
        fast[:, 0] = signed
        return torch.where(draw[:, 2:3] < .68, fast, low)


def wheel_support_error(env):
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "rolling_wheel_sensor_ids"):
        env.rolling_wheel_sensor_ids = [
            sensor.body_names.index(name) for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        ]
    load = sensor.data.net_forces_w.torch[:, env.rolling_wheel_sensor_ids, 2]
    return (load < 2.).float().mean(dim=1) * (env.episode_length_buf > 100)


def rolling_brake_error(env):
    term = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    xy = robot.data.root_lin_vel_b.torch[:, :2]
    yaw = robot.data.root_ang_vel_b.torch[:, 2]
    park_until = getattr(env.cfg,"as2w_park_until_s",5.)
    braking = (term.kind >= .80) & (env.episode_length_buf * env.step_dt >= park_until)
    follow = (xy - term.vel_command_b[:, :2]).norm(dim=1) / .05
    return braking * (follow.clamp(max=4.) + (yaw.abs() / .12).clamp(max=3.))


def rolling_turn_error(env):
    command = env.command_manager.get_command("base_velocity")
    robot = env.scene["robot"]
    xy = robot.data.root_lin_vel_b.torch[:, :2]
    yaw = robot.data.root_ang_vel_b.torch[:, 2]
    yaw_command = command[:, 2].abs()
    if getattr(env.cfg, "as2w_rolling_leg_filter_hz", 0.):
        turn = (yaw_command / .04).clamp(max=1.)
    else:
        turn = yaw_command >= .04
    translation_error = (xy-command[:, :2]).norm(dim=1)/.08
    yaw_error = (yaw-command[:, 2]).abs()/.20
    return turn * (translation_error.clamp(max=3.) + yaw_error.clamp(max=3.))


def rolling_unslewed_leg_rate(env):
    return env.rolling_leg_rate_cost


def configure_speed(cfg, cap, rolling=False, *, fixed_exit=False):
    configure_teleop(cfg, "b")
    cfg.teleop_speed_cap = cap
    cfg.episode_length_s = 20.
    cfg.commands.base_velocity.class_type = RollingSpeedCommands if rolling else SpeedRangeCommands
    cfg.commands.base_velocity.resampling_time_range = (1.1, 3.4)
    cfg.scene.env_spacing = 5.
    cfg.rewards.teleop_axis_error = RewardTermCfg(func=speed_axis_error, weight=-.65)
    if fixed_exit:
        cfg.rewards.rolling_turn_error = RewardTermCfg(func=rolling_turn_error, weight=-.9)
    if rolling:
        cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
        cfg.rewards.rolling_brake_error = RewardTermCfg(func=rolling_brake_error, weight=-.42)


class RollingResidualEnvironment(RobustEnvironment):
    """Fixed stance and differential rolling prior plus one learned residual actor."""

    def step(self, action):
        cmd = self.command_manager.get_command("base_velocity")
        command = action.clone()
        command[:, :12] = .34 * action[:, :12].clamp(-1., 1.)
        side = action.new_tensor((1., -1., 1., -1.))
        wheel_residual = action[:,12:].clamp(-1.,1.)
        turning = (cmd[:, 2:3].abs() / .20).clamp(0., 1.)
        steering = (wheel_residual*side).mean(dim=1,keepdim=True)
        gain = .66*(1.+.25*steering)
        common = wheel_residual.mean(dim=1,keepdim=True)
        # This differential sign was checked in PhysX with all wheel residuals zero.
        linear = cmd[:, :1]*(1.+.25*turning*common) - gain*cmd[:,2:3]*side
        desired = linear / (.086*16.)
        residual = .35*(1.-turning)*wheel_residual
        command[:,12:] = (desired+residual).clamp(-6.,6.)
        return super().step(command)


class RollingFixedExitEnvironment(RobustEnvironment):
    """Use the same action mapping as the MuJoCo fixed-exit evaluator."""

    def _reset_idx(self, ids):
        super()._reset_idx(ids)
        if hasattr(self, "rolling_filtered_legs"):
            self.rolling_filtered_legs[ids] = 0.
        if hasattr(self, "rolling_motion_seen"):
            self.rolling_motion_seen[ids] = False
            self.rolling_quiet_time[ids] = 0.
            self.rolling_park_latched[ids] = False
        if hasattr(self, "rolling_slew_legs"):
            self.rolling_slew_legs[ids] = 0.
            self.rolling_leg_rate_cost[ids] = 0.
        if hasattr(self,"rolling_stand_offset"):
            self.rolling_stand_offset[ids] = 0.
        if hasattr(self,"rolling_wheel_hold_frames"):
            self.rolling_wheel_hold_frames[ids] = 0
            self.rolling_wheel_reference[ids] = 0.

    def step(self, action):
        from as2w_rolling_fixed_exit import (parking_odometry, rolling_fixed_exit, rolling_loaded_stance, rolling_precalibrated_stand,
                                            rolling_stand_bias, rolling_wheel_hold)

        cmd = self.command_manager.get_command("base_velocity")
        cutoff = getattr(self.cfg, "as2w_rolling_leg_filter_hz", 0.)
        if cutoff:
            from as2w_rolling_filtered import filter_leg_residual, update_rolling_phase_cost

            if not hasattr(self, "rolling_filtered_legs"):
                self.rolling_filtered_legs = torch.zeros_like(action[:, :12])
            self.rolling_phase_penalty = update_rolling_phase_cost(self, cmd)
            self.rolling_filtered_legs[:] = filter_leg_residual(
                self.rolling_filtered_legs, action[:, :12], self.step_dt, cutoff)
            action = torch.cat((self.rolling_filtered_legs, action[:, 12:]), dim=1)
        legs, wheels = rolling_fixed_exit(action, cmd,
            pure_turn_common_scale=getattr(self.cfg,"as2w_pure_turn_common_scale",1.))
        if self.cfg.as2w_rolling_stand_compensation:
            if not hasattr(self,"rolling_stand_offset"):
                self.rolling_stand_offset = torch.zeros_like(legs)
            sensed = self.delivered
            settled = None
            if self.cfg.as2w_rolling_wheel_hold:
                if not hasattr(self,"rolling_wheel_hold_frames"):
                    self.rolling_wheel_hold_frames = torch.zeros_like(self.episode_length_buf)
                    self.rolling_wheel_reference = torch.zeros_like(wheels)
                wheel_angle = sensed[:,18:22]
                if self.cfg.as2w_park_odometry:
                    wheel_angle = parking_odometry(wheel_angle,sensed[:,6:18],-sensed[:,3:6])
                self.rolling_wheel_reference,self.rolling_wheel_hold_frames,hold = rolling_wheel_hold(
                    self.rolling_wheel_reference,self.rolling_wheel_hold_frames,cmd,
                    wheel_angle,sensed[:,34:38],sensed[:,:3],self.step_dt,
                    common_feedback=self.cfg.as2w_park_odometry,
                    common_gain_scale=getattr(self.cfg, "as2w_park_common_gain_scale", 1.),
                    common_speed_damping=getattr(self.cfg, "as2w_park_common_speed_damping", 0.),
                    leg_speed=sensed[:,22:34] if self.cfg.as2w_park_settle_joints else None)
                wheels = wheels + hold
                settled = self.rolling_wheel_hold_frames*self.step_dt>=.65
            signals = (self.rolling_stand_offset,cmd,sensed[:,6:18],sensed[:,:3],
                       -sensed[:,5],sensed[:,34:38])
            if self.cfg.as2w_park_precalibrated:
                self.rolling_stand_offset,stand = rolling_precalibrated_stand(
                    *signals,self.rolling_wheel_hold_frames,self.episode_length_buf*self.step_dt,self.step_dt,
                    continuous=self.cfg.as2w_park_continuous_bias,straight_only=self.cfg.as2w_park_straight_bias)
            else:
                self.rolling_stand_offset,stand = rolling_stand_bias(
                    *signals,self.step_dt,enabled=settled)
            legs = legs + stand
            if self.cfg.as2w_rolling_loaded_stance:
                legs = legs + rolling_loaded_stance(self.rolling_stand_offset,cmd,-sensed[:,3:6])
        if getattr(self.cfg, "as2w_rolling_low_speed_smooth", False):
            if not hasattr(self, "rolling_slew_legs"):
                self.rolling_slew_legs = torch.zeros_like(legs)
                self.rolling_leg_rate_cost = torch.zeros_like(cmd[:, 0])
            delta = legs - self.rolling_slew_legs
            step_limit = .9 * self.step_dt / .35
            self.rolling_leg_rate_cost[:] = ((delta.abs() / step_limit - 1.).clamp(0., 8.).square()).mean(dim=1)
            self.rolling_slew_legs += delta.clamp(-step_limit, step_limit)
            legs = self.rolling_slew_legs
        return super().step(torch.cat((legs, wheels), dim=1))
