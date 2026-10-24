"""Optional training-only absorption reward; leaves v2 phase/controller sources intact."""
import hashlib
import math
from pathlib import Path


def metadata(upright_weight, height_weight):
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in (upright_weight, height_weight)):
        raise ValueError('Invalid landing shaping weight')
    return dict(version='landing_absorption_v1', upright_weight=upright_weight,
                height_weight=height_weight, phase='LANDING',
                source_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest())


def shaping_terms(phase, upright, height, target_height):
    import torch
    active = phase == 6  # LANDING only; never reward waiting in REBOUND.
    return (torch.exp(-4 * (1 - upright).square()) * active,
            torch.exp(-((height-target_height)/.10).square()) * active)


def configure(cfg, upright_weight=0., height_weight=0., env_class=None):
    record = metadata(upright_weight, height_weight)
    if not upright_weight and not height_weight:
        return
    if getattr(cfg, 'rebound_phase_version', None) != 'landing_v2':
        raise ValueError('Landing shaping requires configured v2')
    if env_class is None:
        from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv
        env_class = Go2BackflipEnv
    if getattr(env_class, '_landing_shaping_installed', False):
        raise RuntimeError('Landing shaping already installed')
    cfg.rebound_landing_shaping = record
    original = env_class._get_rewards
    def rewards(self):
        value = original(self)
        up, height = shaping_terms(self._rebound.phase,
            -self._robot.data.projected_gravity_b[:, 2],
            self._robot.data.root_link_pos_w[:, 2], self.cfg.rewards.target_height)
        self.extras.setdefault('log', {})['Landing/upright_reward'] = up.mean()
        self.extras['log']['Landing/height_reward'] = height.mean()
        return value + self.step_dt * (upright_weight*up + height_weight*height)
    env_class._get_rewards = rewards
    env_class._landing_shaping_installed = True
