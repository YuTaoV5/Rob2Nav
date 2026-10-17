"""Offline physical trajectory parameter search, using privileged training resets.

No impulses or root state changes are applied after the initial reset. Search
results must be evaluated again using sensor-triggered terrain_run.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import numpy as np
from .terrain_env import TerrainEnv, terrain_config
from .skills import DEFAULT_JUMP, JumpSkill


def trial(env, params, speed, trigger):
    env.reset(seed=123)
    # Training initial conditions: start close enough to avoid wasting rollouts.
    env.data.qpos[0] = 1.8
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env.skill = JumpSkill(params)
    jumped = False
    for _ in range(230):
        if env.data.qpos[0] >= trigger and not jumped:
            env.skill.trigger(float(env.data.time))
            env.jump_count = 1
            jumped = True
        env._control(np.array([speed, 0., 0.]))
        if env.data.qpos[2] < .12 or env.rotation[2, 2] < .35:
            break
        if env.data.qpos[0] > 4.5 and not env.skill.active:
            break
    success = bool(env.data.qpos[0] > 4. and env.data.qpos[2] > .22 and env.rotation[2, 2] > .8 and env.crossed_gap_airborne)
    score = min(float(env.data.qpos[0]), 5.) + 3*success + min(env.max_airborne_time, .3) - 2*max(0, .8-env.rotation[2, 2])
    return score, dict(success=success, position=env.data.qpos[:3].tolist(), airborne=env.max_airborne_time,
                       crossed_gap_airborne=env.crossed_gap_airborne, height_peak=env.height_peak)


def sensor_trial(env, params, speed, trigger_distance):
    env.cfg.update(auto_jump=True, jump_parameters=params, jump_approach_speed=speed,
                   jump_trigger_distance=trigger_distance)
    env.reset(seed=123)
    info = env._info()
    for k in range(650):
        if k % 5 == 0:
            env._sense()
        yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
        env._control(np.array([.85, 0., -yaw]))
        info = env._info()
        if info["fallen"] or info["collision"]:
            break
        if env.data.qpos[0] > 4.5 and not env.skill.active:
            break
    success = bool(env.data.qpos[0] > 4. and env.data.qpos[2] > .22 and env.rotation[2, 2] > .8 and env.crossed_gap_airborne and not info["fallen"])
    score = min(float(env.data.qpos[0]), 4.5) + 4*success - 2*info["fallen"]
    return score, dict(success=success, position=env.data.qpos[:3].tolist(), airborne=env.max_airborne_time,
                      crossed_gap_airborne=env.crossed_gap_airborne, height_peak=env.height_peak)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--width", type=float, default=.30)
    p.add_argument("--output", default="outputs/jump_search")
    p.add_argument("--trigger-min", type=float, default=2.5)
    p.add_argument("--trigger-max", type=float, default=2.95)
    p.add_argument("--sensor-trigger", action="store_true")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = terrain_config(); cfg["gap_width"] = args.width
    env = TerrainEnv(cfg, scene="gap")
    rng = np.random.default_rng(42)
    best = -np.inf
    records = []
    try:
        for i in range(args.trials):
            params = dict(DEFAULT_JUMP) if i == 0 else dict(
                crouch_height=float(rng.uniform(.18, .27)), crouch_time=float(rng.uniform(.08, .22)),
                extend_height=float(rng.uniform(.35, .415)), extend_time=float(rng.uniform(.06, .18)),
                pushback=float(rng.uniform(.05, .23)), front_lead=float(rng.uniform(0, .03)),
                flight_time=float(rng.uniform(.10, .25)), kp=float(rng.uniform(30, 65)), kd=float(rng.uniform(.5, 1.3)))
            speed = .75 if i == 0 else float(rng.uniform(.5, 1.))
            trigger = 2.75 if i == 0 else float(rng.uniform(args.trigger_min, args.trigger_max))
            if args.sensor_trigger:
                trigger = .30 if i == 0 else float(rng.uniform(.20, .55))
                score, info = sensor_trial(env, params, speed, trigger)
            else:
                score, info = trial(env, params, speed, trigger)
            record = dict(trial=i, score=score, jump_parameters=params, speed=speed,
                          trigger_distance=trigger if args.sensor_trigger else None,
                          trigger_x=trigger if not args.sensor_trigger else None, sensor_trigger=args.sensor_trigger, **info)
            records.append(record)
            if score > best:
                best = score
                (out/"best.json").write_text(json.dumps(record, indent=2))
                if args.sensor_trigger:
                    calibrated = dict(cfg, jump_parameters=params, jump_trigger_distance=trigger,
                                      jump_approach_speed=speed)
                    (out/"best_config.json").write_text(json.dumps(calibrated, indent=2))
                print(json.dumps(record), flush=True)
        (out/"trials.json").write_text(json.dumps(records, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
