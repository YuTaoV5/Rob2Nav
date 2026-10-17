"""Learn bounded 12-joint corrections above a frozen RoboGauge prior."""
import argparse
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from .terrain_env import TerrainEnv, terrain_config


class ResidualLocomotionEnv(TerrainEnv):
    def __init__(self, scene="platform", maximum_height=.20):
        cfg = terrain_config()
        cfg.update(navigation_dt=.02, max_seconds=24)
        cfg["auto_jump"] = scene == "gap"
        self.maximum_height = maximum_height
        self.progress = 0.
        self.last_sensor_time = -np.inf
        super().__init__(cfg, scene=scene)
        self.action_space = spaces.Box(-1, 1, (12,), np.float32)
        self.observation_space = spaces.Box(-5, 5, (87,), np.float32)

    def set_progress(self, progress):
        self.progress = float(np.clip(progress, 0, 1))

    def reset(self, *, seed=None, options=None):
        self.last_sensor_time = -np.inf
        self.cfg["terrain_height"] = .02+self.progress*(self.maximum_height-.02)
        if self.scene == "gap":
            self.cfg["gap_width"] = .08+.22*self.progress
        _, info = super().reset(seed=seed, options=options)
        self.training_speed = float(self.np_random.uniform(.35, .65))
        return self.residual_features(np.array([self.training_speed, 0, 0])), info

    def _sense(self):
        if self.data.time-self.last_sensor_time >= .099:
            super()._sense()
            self.last_sensor_time = float(self.data.time)

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected twelve finite normalized joint residuals")
        previous = self.residual.copy()
        self.residual = np.clip(np.asarray(action), -1, 1)*.25
        yaw = np.arctan2(self.rotation[1, 0], self.rotation[0, 0])
        cmd = np.array([self.training_speed, np.clip(-self.data.qpos[1], -.15, .15), np.clip(-yaw, -.6, .6)])
        normalized = cmd / np.array([self.cfg["max_vx"], self.cfg["max_vy"], self.cfg["max_wz"]])
        _, reward, terminated, truncated, info = super().step(normalized)
        reward -= .02*float(np.sum(self.residual**2))+.1*float(np.sum((self.residual-previous)**2))
        reward -= .05*float(1-self.rotation[2, 2])
        return self.residual_features(cmd), reward, terminated, truncated, info


class Curriculum(BaseCallback):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def _on_step(self):
        if self.n_calls % 128 == 0:
            self.training_env.env_method("set_progress", min(self.num_timesteps/(self.steps*.8), 1.))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["stairs", "platform", "gap"], default="platform")
    p.add_argument("--height", type=float, default=.20)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--output", default="outputs/residual")
    args = p.parse_args()
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(ResidualLocomotionEnv, n_envs=args.envs, seed=42,
                       env_kwargs=dict(scene=args.scene, maximum_height=args.height),
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    try:
        model = PPO("MlpPolicy", env, device="cpu", n_steps=512, batch_size=256,
                    learning_rate=1e-4, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.),
                    verbose=1, tensorboard_log=str(out), seed=42)
        model.learn(total_timesteps=args.steps, callback=Curriculum(args.steps))
        model.save(out/"locomotion_residual")
    finally:
        env.close()


if __name__ == "__main__":
    main()
