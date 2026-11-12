"""Go2 MuJoCo keyboard/gamepad session using the existing gait policy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from go2_nav.env import Go2MaplessEnv, load_config


def velocity_command(path, now):
    try:
        signal = json.loads(path.read_text())
        velocity = np.asarray(signal["velocity"], dtype=float)
        if velocity.shape == (3,) and np.isfinite(velocity).all() and 0. <= now - signal["stamp"] <= .35:
            limit = np.array([.35, .20, .50])
            return velocity.clip(-limit, limit)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return np.zeros(3)


def publish(path, phase, velocity, fault="", score=None):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(phase=phase, velocity=velocity.tolist(), fault=fault, score=score)))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(Path("skills/navigation/config.json"))
    cfg.update(locomotion="mlp", navigation_dt=cfg["control_dt"], max_seconds=120)
    scale = np.array([cfg["max_vx"], cfg["max_vy"], cfg["max_wz"]])
    env = Go2MaplessEnv(cfg, scene="empty", sensor_mode="lidar")
    status = args.command.with_name("controller.json")
    command = np.zeros(3)
    publish(status, "SETTLING", command)
    try:
        env.reset(seed=9201)
        dt = cfg["control_dt"]
        for i in range(round(cfg["max_seconds"] / dt)):
            target = velocity_command(args.command, time.monotonic())
            change = np.array([.50, .50, .70]) * dt
            command += np.clip(target - command, -change, change)
            _, reward, _, _, info = env.step(command / scale)
            score = dict(time=float(env.data.time) - env.start_time, step=i + 1,
                         total=reward, terms=info["reward_terms"])
            if info["fallen"] or info["collision"]:
                publish(status, "FAULT", np.zeros(3), "仿真检测到倒地或碰撞", score)
                break
            if i % 5 == 0:
                publish(status, "ACTIVE", command, score=score)
        else:
            publish(status, "COMPLETED", np.zeros(3))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(dict(
            backend="MuJoCo", diagnostic_only=True, hardware_connected=False,
            acceptance_scope="Go2 MLP 手动仿真控制", final_position=info["position"])))
    finally:
        env.close()


if __name__ == "__main__":
    main()
