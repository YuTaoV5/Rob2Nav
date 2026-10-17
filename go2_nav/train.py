import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from .env import Go2MaplessEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene", choices=["empty", "slalom", "random"], default="random")
    p.add_argument("--sensors", choices=["both", "depth", "lidar"], default="both")
    p.add_argument("--output", default="outputs/ppo")
    p.add_argument("--resume")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(Go2MaplessEnv, n_envs=args.envs, seed=args.seed,
                       env_kwargs=dict(scene=args.scene, sensor_mode=args.sensors),
                       vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    try:
        model = PPO.load(args.resume, env=env, device="cpu") if args.resume else PPO(
            "MlpPolicy", env, device="cpu", n_steps=256, batch_size=256,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
            learning_rate=3e-4, gamma=.99, verbose=1, seed=args.seed, tensorboard_log=str(out))
        model.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.resume))
        model.save(out / "navigation")
    finally:
        env.close()


if __name__ == "__main__":
    main()
