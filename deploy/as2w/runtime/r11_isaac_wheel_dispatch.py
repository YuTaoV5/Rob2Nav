#!/usr/bin/env python3
"""Record the ACTUAL dispatched wheel target, per frame, next to the measured response.

Why this exists: the diagonalisation question ("is the left-right asymmetry in the policy's COMMAND, or
does the contact turn a symmetric command into a diagonal?") cannot be closed without the final
dispatched value.  The state tracer only recorded the previous action scaled by 16, which is NOT what
reaches the actuator.

This probe reads the action term's own state without modifying the training code:
    tgt_delayed    - the value returned by DelayedTarget.delayed()  (post delay-buffer, pre compensation)
    ff            - the gyro feed-forward actually added (S * clip(...))
    dispatched    - the value handed to set_joint_velocity_target_index  (tgt + ff/kd)
and pairs each with the SAME frame's measured wheel speed, wheel load and body yaw rate.

The action term is reached through the env's action manager, so the probe wraps the environment and
installs a recorder on the wheel action term.  It runs the normal policy, so the numbers are directly
comparable with the acceptance matrix.
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

LEGS = ("FL", "FR", "RL", "RR")


class WheelDispatchRecorder:
    """Wraps the wheel action term's apply_actions so every dispatch is captured."""

    def __init__(self, term):
        self.term = term
        self.rows = []
        self._orig = None

    def install(self):
        term = self.term
        # bind the ORIGINAL bound method, not an attribute that may not exist
        orig = term.apply_actions
        self._orig = orig

        def patched(*args, **kwargs):
            before = term.processed_actions.detach().clone()
            try:
                delayed = term.delayed()
            except Exception:
                delayed = None
            out = orig(*args, **kwargs)
            try:
                tgt = term._asset.data.joint_vel_target.torch.detach().clone()
            except Exception:
                tgt = None
            self.rows.append({"processed": before, "delayed": delayed, "dispatched": tgt})
            return out

        term.apply_actions = patched

    def restore(self):
        if self._orig is not None:
            self.term.apply_actions = self._orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=10.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        from as2w_isaac_flat_env_all import (
            RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
            check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
        from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
        from as2w_training_safety import apply_safety

        class EvalCommands(TakeoverCommands):
            def _resample_command(self, env_ids):
                pass

        cfg = robust_config(SimpleNamespace(**vars(args), profile="pose-delay", direction_focus=True,
                                             command_stage="all", zero_memory=True, zero_focus=True), physics)
        cmd = cfg.commands.base_velocity
        cmd.class_type = EvalCommands
        cmd.resampling_time_range = (1e6, 1e6)
        cfg.episode_length_s = 60.
        cfg.scene.robot.actuators["foot"].damping = 6.0
        apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
        env = RobustEnvironment(cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=6.0)
            robot = env.scene["robot"]
            names = list(robot.joint_names)
            wheel_ids = [names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]

            # locate the wheel velocity action term
            am = env.action_manager
            terms = getattr(am, "_terms", None) or getattr(am, "terms", None) or {}
            term = terms.get("wheel_vel")
            if term is None:
                for t in terms.values():
                    if type(t).__name__ == "DelayedGyroVelocity":
                        term = t
                        break
            if term is None:
                print("WHEEL_TERM_NOT_FOUND; terms:", [(k, type(v).__name__) for k, v in terms.items()])
                raise SystemExit(3)
            term_jn = list(getattr(term, "joint_names", []) or [])
            if not term_jn:
                term_jn = [names[i] for i in getattr(term, "_joint_ids", [])]
            print("AS2W_WHEEL_TERM", type(term).__name__, term_jn,
                  "joint_ids", list(getattr(term, "_joint_ids", [])), flush=True)
            rec = WheelDispatchRecorder(term)
            rec.install()

            ppo = UnitreeGo2FlatPPORunnerCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_deprecated_rsl_rl_cfg(ppo, check_rsl_rl_version())
            runner = create_rsl_rl_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            cmdterm = env.command_manager.get_term("base_velocity")
            env.seed(args.seed); env.reset()
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            for _ in range(int(args.settle_s / 0.02)):
                with torch.inference_mode():
                    a = policy(obs)
                obs, _r, _d, _i = wrapped.step(a)
            rec.rows.clear()
            cmdterm.vel_command_b[:] = torch.tensor([[args.vx, 0.0, args.wz]], dtype=torch.float32, device=env.device)
            frames = []
            for _ in range(int(args.turn_s / 0.02)):
                with torch.inference_mode():
                    a = policy(obs)
                obs, _r, _d, _i = wrapped.step(a)
                dq = robot.data.joint_vel.torch[0][wheel_ids].detach().cpu().numpy().copy()
                yaw = float(robot.data.root_ang_vel_b.torch[0, 2])
                frames.append({"dq": [float(x) for x in dq], "yaw": yaw,
                               "act_wheel": [float(x) for x in a[0][12:].detach().cpu().numpy()]})
            rec.restore()

            # pair each frame with the dispatch that produced it (dispatch happens inside the step)
            n = min(len(frames), len(rec.rows))
            out = []
            for k in range(n):
                row = rec.rows[k]
                disp = row["dispatched"]
                out.append({
                    "frame": k,
                    "processed_action": [float(x) for x in row["processed"][0].flatten()[:4].detach().cpu().numpy()],
                    "delayed_target": None if row.get("delayed") is None else
                        [float(x) for x in row["delayed"][0].flatten()[:4].detach().cpu().numpy()],
                    "dispatched_wheel_target": None if disp is None else
                        [float(x) for x in disp[0][wheel_ids].detach().cpu().numpy()],
                    "measured_wheel_speed": frames[k]["dq"],
                    "yaw_rate": frames[k]["yaw"],
                    "act_wheel_action": frames[k]["act_wheel"]})
            rep = {"vx": args.vx, "wz": args.wz, "frames_paired": n,
                   "dispatch_rows_captured": len(rec.rows),
                   "note": "dispatched_wheel_target is the value the articulation received; "
                           "processed_action is the term's own processed action",
                   "frames": out}
            (args.out / "wheel_dispatch.json").write_text(json.dumps(rep))
            print("PAIRED_FRAMES", n, "DISPATCH_ROWS", len(rec.rows))
            for k in range(0, n, max(1, n // 6)):
                r = out[k]
                print("  f%-4d dispatched=%s measured=%s yaw=%+.4f" % (
                    r["frame"],
                    None if r["dispatched_wheel_target"] is None else [round(x, 3) for x in r["dispatched_wheel_target"]],
                    [round(x, 3) for x in r["measured_wheel_speed"]], r["yaw_rate"]))
            print("WHEEL_DISPATCH_DONE")
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
