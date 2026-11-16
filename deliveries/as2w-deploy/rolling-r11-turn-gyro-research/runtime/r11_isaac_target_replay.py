#!/usr/bin/env python3
"""Replay the SAME final four-wheel target sequence in Isaac and measure what Isaac delivers.

Goal section 3 asks for the two simulators to be run on the same final four-wheel target sequence
before comparing actual yaw.  The operator's item 1 asks the same thing: after confirming what MuJoCo
commands, see how much Isaac delivers under those identical targets.

Input is a MuJoCo trace produced by `--dump-trace`, whose `wheel_applied` field is the actor's wheel
target (already multiplied by the wheel scale).  For every frame we drive the legs with the LOCAL
actor (its observations come from Isaac, so the stance is Isaac's own) and force the four wheel
targets to the recorded values.  That isolates delivery: same wheel targets, local leg feedback.

Reported per case: target differential, measured differential, delivery ratio, and the actual yaw,
plus the support/upright/termination telemetry needed to read the comparison honestly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

VX_CAP, YAW_CAP = 1.2, .15
WHEEL_KD, ACTION_CLIP, WHEEL_SCALE = 6.0, 6.0, 16.0
DT = 0.02
TAIL = 150
LOAD_N = 5.0
WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def build(args, physics):
    from as2w_isaac_flat_env_all import (
        RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_training_safety import apply_safety

    class EvalCommands(TakeoverCommands):
        def _resample_command(self, env_ids):
            pass

    cfg = robust_config(SimpleNamespace(
        **vars(args), profile="pose-delay", direction_focus=True,
        command_stage="all", zero_memory=True, zero_focus=True), physics)
    cmd = cfg.commands.base_velocity
    cmd.class_type = EvalCommands
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.ranges.lin_vel_x = (-VX_CAP, VX_CAP)
    cmd.ranges.lin_vel_y = (0., 0.)
    cmd.ranges.ang_vel_z = (-YAW_CAP, YAW_CAP)
    cfg.episode_length_s = 40.
    cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
    apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    return env, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg


def d4(A):
    A = np.asarray(A, float)
    return A[:, 1] + A[:, 3] - A[:, 0] - A[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True, help="MuJoCo --dump-trace JSON")
    ap.add_argument("--cases", default="turn_left_0.05,turn_right_0.05,turn_left_0.10,turn_right_0.10")
    ap.add_argument("--settle-match", type=float, default=3.0,
                    help="pick the trace whose settle_s matches this")
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt", "trace"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    tj = json.loads(args.trace.read_text())
    want = [c.strip() for c in args.cases.split(",") if c.strip()]
    picks = {}
    for tr in tj["traces"]:
        if tr["case"] in want and abs(float(tr["settle_s"]) - args.settle_match) < 1e-6:
            picks[tr["case"]] = tr
    print("TRACE_PICKS %s" % json.dumps(sorted(picks)), flush=True)

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        env, Wrapper, PPOCfg, check_v, create_runner, handle_dep = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")
            names = list(robot.joint_names)
            widx = [names.index(n) for n in WHEELS]
            calfd = [names.index(n.replace("_foot", "_calf")) for n in WHEELS]
            print("WHEEL_JOINTS %s" % json.dumps([names[i] for i in widx]), flush=True)

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed, "trace": str(args.trace),
                   "settle_match": args.settle_match,
                   "note": "legs driven by the local actor on Isaac observations; the four wheel targets are forced to the recorded MuJoCo values",
                   "cases": {}}
            for case, tr in picks.items():
                cmd = tr["cmd"]
                wz = float(cmd[2])
                tgts = np.array(tr["wheel_applied"], float)          # already scaled
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = torch.tensor([[0.0, 0.0, wz]],
                                                        dtype=torch.float32, device=env.device)
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                mea, yaw, sup, up, term = [], [], [], [], 0
                n = min(len(tgts), 600)
                for k in range(n):
                    cmdterm.vel_command_b[:] = torch.tensor([[0.0, 0.0, wz]],
                                                            dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    a = a.clone()
                    for j in range(4):
                        a[0, 12 + j] = float(tgts[k][j]) / WHEEL_SCALE
                    obs, _r, _d, _i = wrapped.step(a)
                    mea.append(robot.data.joint_vel.torch[0][widx].detach().cpu().numpy().copy())
                    yaw.append(float(robot.data.root_ang_vel_b.torch[0, 2]))
                    up.append(float(-robot.data.projected_gravity_b.torch[0][2]))
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][calfd]
                    sup.append(bool((torch.linalg.norm(cf, dim=1) > LOAD_N).all()))
                    term += int(bool(_d[0]))
                mea = np.array(mea)
                tl = slice(max(0, n - TAIL), n)
                dtg, dme = float(np.mean(d4(tgts[:n])[tl])), float(np.mean(d4(mea)[tl]))
                rep["cases"][case] = {
                    "cmd_wz": wz,
                    "ddq_target_tail": dtg,
                    "ddq_measured_tail": dme,
                    "exec": (abs(dme) / abs(dtg)) if abs(dtg) > 1e-9 else None,
                    "yaw_tail": float(np.mean(yaw[tl])),
                    "yaw_track": (float(np.mean(yaw[tl]) / wz) if abs(wz) > 1e-9 else None),
                    "support_tail": float(np.mean(sup[tl])),
                    "support_all": float(np.mean(sup)),
                    "upright_min": float(np.min(up)),
                    "terminations": term,
                    "n_frames": n,
                }
                r = rep["cases"][case]
                print("== %-16s ddq_tgt=%+.3f ddq_mea=%+.3f exec=%.2f yaw_track=%+.2f sup_tail=%.3f up_min=%.3f term=%d"
                      % (case, dtg, dme, r["exec"] or 0, r["yaw_track"] or 0,
                         r["support_tail"], r["upright_min"], r["terminations"]), flush=True)
            (args.out / "isaac_target_replay.json").write_text(json.dumps(rep, indent=1, default=str))
            print("ISAAC_TARGET_REPLAY_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
