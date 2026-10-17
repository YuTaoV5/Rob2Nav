"""Offline sim-only timing calibration of frozen NJU BBC (not RL training)."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .active_env import ActiveTerrainEnv
from .terrain_env import terrain_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bbc_calibration")
    p.add_argument("--backend", default="nju_all", choices=["nju", "nju_all", "primitive"])
    p.add_argument("--scene", default="gap", choices=["gap", "course"])
    p.add_argument("--walk-mode", type=int, default=0)
    p.add_argument("--approach-speed", type=float)
    p.add_argument("--triggers", nargs="+", type=float)
    p.add_argument("--speeds", nargs="+", type=float)
    p.add_argument("--heights", nargs="+", type=float)
    p.add_argument("--modes", type=int, default=3, choices=[2, 3])
    a = p.parse_args()
    out = Path(a.output);out.mkdir(parents=True, exist_ok=True)
    results = []
    triggers = a.triggers or ([.20, .30, .40, .50, .60] if a.backend == "primitive" else [.20, .35, .50])
    speeds = a.speeds or ([.60, .75, .90] if a.backend == "primitive" else [.8, 1.2, 1.6])
    heights = a.heights or ([.5] if a.backend == "primitive" else [.48, .56])
    grid = itertools.product(triggers, speeds, heights)
    for trigger, speed, height in grid:
        cfg = terrain_config()
        cfg.update(jump_backend=a.backend, bbc_modes=a.modes, jump_trigger_distance=trigger,
                   bbc_speed=speed, bbc_jump_height=height, max_seconds=45., anticipatory_lift=True,
                   bbc_walk_mode=a.walk_mode, jump_approach_speed=speed)
        if a.approach_speed is not None:
            cfg["bbc_approach_speed"] = a.approach_speed
        env = ActiveTerrainEnv(cfg, scene=a.scene)
        try:
            env.reset(seed=200)
            from .terrain_planner import TerrainNavigator
            navigator = TerrainNavigator(cfg)
            while True:
                yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
                action = navigator.act(env.points, env.level_water, env.data.qpos[:2], yaw, env.goal, env.command)
                _, _, done, timeout, info = env.step(action)
                if done or timeout:
                    break
            row = dict(trigger=trigger, speed=speed, height=height, **info)
            row["success"] = bool(info["success"] and info["crossed_gap_airborne"])
            results.append(row)
            print(json.dumps(row), flush=True)
            (out/"trials.json").write_text(json.dumps(results, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
