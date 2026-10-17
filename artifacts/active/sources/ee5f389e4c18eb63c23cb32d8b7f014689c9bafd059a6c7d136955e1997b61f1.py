import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import imageio.v2 as imageio
import numpy as np

from .env import Go2MaplessEnv, load_config
from .planner import ReactivePlanner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=["empty", "slalom", "random", "blocked"], default="slalom")
    p.add_argument("--policy", choices=["reactive", "ppo", "straight"], default="reactive")
    p.add_argument("--checkpoint")
    p.add_argument("--sensors", choices=["both", "depth", "lidar"], default="both")
    p.add_argument("--config")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--output", default="outputs/reactive")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = Go2MaplessEnv(load_config(args.config), scene=args.scene, sensor_mode=args.sensors)
    if args.policy == "ppo":
        if not args.checkpoint:
            p.error("--checkpoint is required for PPO")
        from stable_baselines3 import PPO
        policy = PPO.load(args.checkpoint, device="cpu")
    results = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed+episode)
            planner = ReactivePlanner(env.cfg)
            trajectory = []
            writer = imageio.get_writer(out / f"episode_{episode}.mp4", fps=10) if args.video else None
            try:
                while True:
                    if args.policy == "reactive":
                        delta = np.r_[env.goal-env.data.qpos[:2], 0.]
                        goal_body = (env.rotation.T @ delta)[:2]
                        action = planner.act(env.points, goal_body, env.command)
                    elif args.policy == "ppo":
                        action, _ = policy.predict(obs, deterministic=True)
                    else:
                        action = np.array([.7, 0, 0])
                    obs, reward, terminated, truncated, info = env.step(action)
                    trajectory.append(dict(step=env.steps, reward=reward, **info))
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                if writer:
                    writer.close()
            info.update(seed=args.seed+episode, episode=episode, timeout=bool(truncated), scene=args.scene, policy=args.policy)
            results.append(info)
            (out / f"trajectory_{episode}.json").write_text(json.dumps(trajectory, indent=2))
            np.savez_compressed(out / f"sensors_{episode}.npz", depth_m=env.depth,
                                lidar_ranges_m=env.lidar_ranges, points_body_m=env.points, scan_m=env.scan)
            print(json.dumps(info), flush=True)
        summary = dict(episodes=len(results), successes=sum(r["success"] for r in results),
                       collisions=sum(r["collision"] for r in results), falls=sum(r["fallen"] for r in results),
                       results=results, sensor_mode=args.sensors, config=env.cfg)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
