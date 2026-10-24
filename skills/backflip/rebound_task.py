"""Opt-in rebound training task; preserves the 60/165 network layout.

The six phase features now describe the CURRENT cycle. Reusing weights is
initialization, not behavioral equivalence. No simulator state is reset between flips.
"""
from dataclasses import asdict
import torch
from .rebound import ReboundConfig, ReboundPhases


def configure(cfg, cycles, max_rebound_seconds=.45, env_class=None):
    settings = ReboundConfig(cycles=cycles, max_rebound_seconds=max_rebound_seconds)
    if cfg.episode_length_s != 5. or getattr(cfg, 'entry_delay_seconds', None) != 1.:
        raise ValueError('Rebound requires the reviewed delayed-entry parent task')
    if env_class is None:
        from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv
        env_class = Go2BackflipEnv
    if getattr(env_class, '_rob2nav_rebound_installed', False):
        raise RuntimeError('Rebound hooks already installed; use a fresh training process')
    cfg.rebound = asdict(settings)
    cfg.episode_length_s = 1. + 2. * cycles + 2.
    cfg.contact_sensor.update_period = cfg.sim.dt
    cfg.rewards.use_peak_contact_forces = True
    cls = env_class
    init, reset = cls.__init__, cls._reset_idx
    pre, apply, update, dones = cls._pre_physics_step, cls._apply_action, cls._update_flip_state, cls._get_dones

    def ensure(self):
        if hasattr(self, '_rebound'):
            return self._rebound
        r = self._rebound = ReboundPhases(self.num_envs, self.cfg.sim.dt, settings, self.device)
        self._rebound_clock = torch.full_like(r.elapsed, -settings.entry_seconds)
        self._rebound_tick = int(self._sim_step_counter)
        self._rebound_landed = torch.zeros_like(r.just_landed)
        self._rebound_progress = torch.zeros_like(r.elapsed)
        self._rebound_hold = torch.zeros_like(r.elapsed)
        return r

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        ensure(self)

    def sample(self, tick):
        r = ensure(self)
        if tick == self._rebound_tick:
            return
        if tick != self._rebound_tick + 1:
            raise RuntimeError('Rebound physics sample missing or out of order')
        self._rebound_tick = tick
        force = torch.linalg.vector_norm(self._contact_sensor.data.net_forces_w, dim=-1)
        foot = (force[:, self._feet_contact_ids] > self.cfg.rewards.recovery_contact_force).any(dim=-1)
        body = (force[:, self._undesired_contact_sensor_ids] > self.cfg.rewards.unsafe_body_contact_force).any(dim=-1)
        old_angle = r.angle.clone()
        old_phase = r.phase.clone()
        r.step(foot, self._robot.data.root_ang_vel_b, body)
        self._rebound_clock += self.cfg.sim.dt
        # On intermediate landing, enter the launch part of the next cycle,
        # not the one-second entry or final stand. The transition is learned.
        intermediate = r.just_landed & (r.phase == r.REBOUND)
        self._rebound_clock[intermediate] = self.cfg.rewards.takeoff_start
        self._rebound_clock[r.phase == r.RECOVERY] = 2.
        self._rebound_landed |= r.just_landed
        progress = (r.angle - old_angle).clamp_min(0.)
        active = ((old_phase == r.PUSH) | (old_phase == r.REBOUND) | (old_phase == r.FLIGHT))
        self._rebound_progress += torch.where(active & ~foot & ~r.just_landed, progress, 0.)

    def before(self, actions):
        ensure(self)
        self._rebound_landed.zero_()
        self._rebound_progress.zero_()
        pre(self, actions)  # existing entry hold only uses episode time, once

    def apply_action(self):
        sample(self, int(self._sim_step_counter) - 1)
        apply(self)

    def update_state(self):
        sample(self, int(self._sim_step_counter))
        update(self)  # retain upstream motor/contact constraint bookkeeping
        r = self._rebound
        final = r.phase == r.RECOVERY
        self._just_completed[:] = self._rebound_landed
        self._rotation_progress_step[:] = self._rebound_progress
        self._flip_completed[:] = final
        self._landing_detected[:] = (r.phase == r.REBOUND) | final
        self._was_airborne[:] = r.phase == r.FLIGHT
        self._flip_angle[:] = r.angle
        self._max_flip_angle[:] = r.angle
        force = torch.linalg.vector_norm(self._contact_sensor.data.net_forces_w, dim=-1)
        feet = (force[:, self._feet_contact_ids] > self.cfg.rewards.recovery_contact_force).sum(-1)
        q = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        stable = (final & ~self._unsafe_episode & (feet >= 3) &
                  (-self._robot.data.projected_gravity_b[:, 2] >= self.cfg.rewards.recovery_upright_cos) &
                  (self._robot.data.root_link_pos_w[:, 2] >= self.cfg.rewards.recovery_min_height) &
                  (torch.linalg.vector_norm(self._robot.data.root_ang_vel_b, dim=-1) <= self.cfg.rewards.recovery_max_pitch_rate) &
                  (self._robot.data.joint_vel.abs().amax(-1) <= self.cfg.rewards.success_max_joint_speed) &
                  (q.abs().amax(-1) <= self.cfg.rewards.recovery_pose_error))
        self._rebound_hold = torch.where(stable, self._rebound_hold + self.step_dt, 0.)
        success = self._rebound_hold + 1e-6 >= self.cfg.rewards.success_hold_time_s
        previous = getattr(self, '_rebound_success', torch.zeros_like(success))
        self._just_succeeded[:] = success & ~previous
        self._rebound_success = success
        self._flip_success[:] = success

    def get_dones(self):
        terminated, timeout = dones(self)
        return terminated | (ensure(self).phase == ReboundPhases.FAILED), timeout

    def reset_state(self, ids):
        reset(self, ids)
        r = ensure(self)
        ids = slice(None) if ids is None else ids
        r.reset(ids)
        self._rebound_clock[ids] = -settings.entry_seconds
        self._rebound_hold[ids] = 0.
        self._rebound_landed[ids] = False
        self._rebound_progress[ids] = 0.
        if hasattr(self, '_rebound_success'):
            self._rebound_success[ids] = False
        # Global physics clock is never reset for a subset of environments.

    def phases(self):
        ensure(self)
        phase = torch.pi * self._rebound_clock[:, None].clamp(max=2.) / 2.
        return tuple(f(phase / divisor) for divisor in (1., 2., 4.) for f in (torch.sin, torch.cos))

    def recovery(self):
        return ensure(self).phase == ReboundPhases.RECOVERY

    def endpoints(self):
        r = ensure(self)
        return (r.phase == r.ENTRY) | (r.phase == r.RECOVERY)

    def default_pose(self):
        return (self._robot.data.joint_pos - self._robot.data.default_joint_pos).square().sum(-1) * endpoints(self)

    def height(self):
        return (self.cfg.rewards.target_height - self._robot.data.root_link_pos_w[:, 2]).square() * endpoints(self)

    def rewards(self):
        # Replace the old wrapper's unconditional t>=3s stance bonus.
        values = {}
        for name, function in self._reward_functions.items():
            value = function() * self.cfg.rewards.scales[name] * self.step_dt
            self._episode_sums[name] += value
            values[name] = value
        from skills.fixstand import stance_score
        bonus = stance_score(self._robot) * endpoints(self) * 25. * self.step_dt
        self.extras.setdefault('log', {})['Rebound/completed_cycles'] = self._rebound.completed.float().mean()
        self.extras['log']['Rebound/final_hold_s'] = self._rebound_hold.mean()
        return torch.stack(tuple(values.values()), dim=0).sum(0) + bonus

    cls.__init__ = initialize
    cls._reset_idx = reset_state
    cls._pre_physics_step = before
    cls._apply_action = apply_action
    cls._update_flip_state = update_state
    cls._get_dones = get_dones
    cls._get_rewards = rewards
    cls._phase_features = phases
    cls._time = lambda self: self._rebound_clock
    cls._recovery_active = recovery
    cls._reward_default_pose = default_pose
    cls._reward_height_control = height
    cls._rob2nav_rebound_installed = True
