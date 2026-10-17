import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import imageio.v2 as imageio
import numpy as np
from .terrain_env import TerrainEnv, terrain_config
from .planner import ReactivePlanner
from .terrain_planner import TerrainNavigator
from .terrains import TERRAIN_SCENES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", choices=TERRAIN_SCENES, default="water")
    p.add_argument("--config")
    p.add_argument("--locomotion", choices=["mlp", "moe"], default="moe")
    p.add_argument("--height", type=float)
    p.add_argument("--gap-width", type=float)
    p.add_argument("--max-seconds", type=float, default=45)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--straight", action="store_true", help="Privileged-free fixed-command locomotion benchmark")
    p.add_argument("--no-jump", action="store_true")
    p.add_argument("--residual-checkpoint")
    p.add_argument("--video", action="store_true")
    p.add_argument("--output", default="outputs/terrain")
    args = p.parse_args()
    cfg = terrain_config(args.config)
    cfg.update(locomotion=args.locomotion, max_seconds=args.max_seconds)
    cfg["auto_jump"] = not args.no_jump
    cfg["water_recenter"] = args.scene == "mixed"
    if args.residual_checkpoint:
        cfg["residual_checkpoint"] = args.residual_checkpoint
    if args.height is not None:
        cfg["terrain_height"] = args.height
    if args.gap_width is not None:
        cfg["gap_width"] = args.gap_width
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = TerrainEnv(cfg, scene=args.scene)
    results = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed+episode)
            planner = TerrainNavigator(cfg)
            trajectory = []
            writer = imageio.get_writer(out/f"episode_{episode}.mp4", fps=10) if args.video else None
            try:
                while True:
                    delta = np.r_[env.goal-env.data.qpos[:2], 0.]
                    yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
                    goal_body = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ delta[:2]
                    action = np.array([.65, 0., np.clip(1.5*np.arctan2(goal_body[1], goal_body[0]), -1, 1)]) if args.straight else planner.act(env.points, env.level_water, env.data.qpos[:2], yaw, env.goal, env.command)
                    if env.skill.active:
                        action = np.array([.9, 0., 0.])
                    obs, reward, terminated, truncated, info = env.step(action)
                    trajectory.append(dict(step=env.steps, reward=reward, **info))
                    if writer:
                        writer.append_data(env.render())
                    if terminated or truncated:
                        break
            finally:
                if writer:
                    writer.close()
            reached = info["success"]
            if args.scene in ("gap", "mixed"):
                info["success"] = bool(reached and info["crossed_gap_airborne"])
            info.update(seed=args.seed+episode, timeout=bool(truncated), reached_goal=bool(reached),
                        scene=args.scene, terrain_height=cfg["terrain_height"], gap_width=cfg.get("gap_width", .3))
            results.append(info)
            print(json.dumps(info), flush=True)
            (out/f"trajectory_{episode}.json").write_text(json.dumps(trajectory, indent=2))
        (out/"summary.json").write_text(json.dumps(dict(config=cfg, results=results,
              episodes=len(results), successes=sum(r["success"] for r in results),
              collisions=sum(r["collision"] for r in results), falls=sum(r["fallen"] for r in results),
              water_entries=sum(r["water_entered"] for r in results)), indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
