#!/usr/bin/env python3
"""AS2-W rolling: frozen acceptance matrix v2 (adds the 0.25 m/s tier).

Operator selections this run implements:
  * next_training = A  -> evaluate turn-pilot's endpoint against model_1999 before
    any further training; do not launch training blind.
  * auto_proceed  = authorized -> after evaluating, pick the next training
    direction automatically, but STOP and ask if the fix needs a reward,
    threshold or wrapper change.
  * speed_tier    = add 0.25 m/s straight + arc (plus reverse and release braking).

FROZEN GATES (declared before scoring, not tuned afterwards)
    hold     10 s, last 3 s |vx| < 0.02 m/s and |wz| < 0.02 rad/s
    turn     12 s at +-0.05 / +-0.10 rad/s, last 3 s signed yaw tracking >= 0.70
    straight 12 s at +-0.25 m/s, last 3 s signed vx tracking in [0.70, 1.30]
    arc      12 s at (vx +0.25, wz +-0.05), last 3 s BOTH vx and yaw tracking in [0.70, 1.30]
    brake    10 s: 5 s drive then release; last 3 s |vx| < 0.02 m/s; record stop distance
    all cases: four-wheel support >= 0.95 in the last 3 s, no termination,
               upright (projected gravity z) >= 0.85
Judgement order: support / hard protection FIRST, then signed tracking.

Protocol: `env.seed(); env.reset()` + fixed settle per repeat; results reported as
mean/min/max across seeds.  Snapshot forks are NOT used -- the Isaac/PhysX fork was
shown to be non-reproducible (2.067 rad/s repeat, 3.480 rad/s order-swap), while
MuJoCo is bit-reproducible only because its snapshot carries `qacc_warmstart`.
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

VX_CAP, YAW_CAP = .25, .15
WHEEL_KD, WHEEL_SCALE, ACTION_CLIP = 6.0, 16.0, 6.0
DT = 0.02
SETTLE_S = 2.0
LOAD_N = 5.0
TAIL_S = 3.0
BRAKE_HOLD_S = 5.0

# ---- FROZEN GATES ----
G_HOLD_VX, G_HOLD_WZ = 0.02, 0.02
G_YAW_TRACK_LO, G_YAW_TRACK_HI = 0.70, 1.30
G_VX_TRACK_LO, G_VX_TRACK_HI = 0.70, 1.30
G_SUPPORT = 0.95
G_UPRIGHT = 0.85
G_BRAKE_VX = 0.02
TURN_TIERS = (0.05, 0.10)
STRAIGHT_TIER = 0.25
ARC_TIER = (0.25, 0.05)          # (vx, wz)


def build(args, physics):
    from isaaclab.managers import RewardTermCfg
    from as2w_isaac_flat_env_all import (
        RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_isaac_speed_range import wheel_support_error
    from as2w_training_safety import apply_safety, rolling_knee_margin_cost

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
    cfg.episode_length_s = 32.
    cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
    cfg.rewards.rolling_knee_margin = RewardTermCfg(
        func=rolling_knee_margin_cost, weight=-3., params={"margin_m": .10})
    apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    return (env, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
            check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)


def d4(A):
    A = np.asarray(A, float)
    return A[:, 1] + A[:, 3] - A[:, 0] - A[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoints", required=True, help="label=path,...")
    ap.add_argument("--seeds", default="7393,311")
    # robust_config() reads args.seed; add_launcher_args does not define it
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--skip-speed-tier", action="store_true")
    add_launcher_args(ap)
    args = ap.parse_args()
    for a in ("usd", "urdf"):
        setattr(args, a, getattr(args, a).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"
    cks = []
    for spec in args.checkpoints.split(","):
        lab, _, p = spec.partition("=")
        cks.append((lab, Path(p)))
    seeds = [int(s) for s in args.seeds.split(",")]

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        (env, Wrapper, PPOCfg, check_v, create_runner, handle_dep) = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            jids = env.as2w_joint_ids
            wj = jids[12:]
            cmdterm = env.command_manager.get_term("base_velocity")
            wterm = env.action_manager.get_term("wheel_vel")
            st = {"applied": torch.zeros(1, 4, device=env.device)}
            def patched_apply():
                delayed = wterm.delayed()
                st["applied"] = delayed.clone()
                robot.set_joint_velocity_target_index(target=delayed,
                                                      joint_ids=wterm._joint_ids)
            wterm.apply_actions = patched_apply

            ppo = PPOCfg(); ppo.seed, ppo.device = seeds[0], args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)

            def settle(seed):
                env.seed(seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(SETTLE_S / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, *_ = wrapped.step(a)
                return obs

            def run_case(obs, cmd_fn, seconds):
                n = int(seconds / DT)
                R = {k: [] for k in ("yaw", "vx", "vy", "load", "app", "meas", "up")}
                term = 0
                for k in range(n):
                    c = cmd_fn(k * DT)
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [c], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, rew, don, info = wrapped.step(a)
                    if bool(don[0]):
                        term = 1
                    R["yaw"].append(float(robot.data.root_ang_vel_b.torch[0, 2]))
                    R["vx"].append(float(robot.data.root_lin_vel_b.torch[0, 0]))
                    R["vy"].append(float(robot.data.root_lin_vel_b.torch[0, 1]))
                    R["app"].append(st["applied"][0].detach().cpu().numpy().tolist())
                    R["meas"].append(robot.data.joint_vel.torch[0, wj]
                                     .detach().cpu().numpy().tolist())
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0]
                    cfz = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]][:, 2]
                    R.setdefault("load_z", []).append(cfz.detach().cpu().numpy().tolist())
                    R["load"].append(torch.linalg.norm(cf[[13, 14, 15, 16]], dim=1)
                                     .detach().cpu().numpy().tolist())
                    g = robot.data.projected_gravity_b.torch[0]
                    R["up"].append(float(-g[2]))
                R["term"] = term
                return R

            def judge(R, cmd, seconds, brake=False, brake_at=0.0):
                tail = max(1, int(TAIL_S / DT))
                yaw = np.asarray(R["yaw"], float)
                vx = np.asarray(R["vx"], float)
                L = np.asarray(R["load"], float)
                sup = (L > LOAD_N).all(axis=1)
                app_d = d4(R["app"])
                mea_d = d4(R["meas"])
                wz_cmd, vx_cmd = float(cmd[2]), float(cmd[0])
                out = {
                    "cmd": [float(cmd[0]), float(cmd[1]), float(cmd[2])],
                    "vx_tail": float(vx[-tail:].mean()),
                    "vy_tail": float(np.mean(R["vy"][-tail:])),
                    "wz_tail": float(yaw[-tail:].mean()),
                    "support_tail": float(sup[-tail:].mean()),
                    "unloaded_frac_z2n_tail": (float((np.asarray(R["load_z"], float)[-tail:] < 2.0).mean()) if "load_z" in R else None),  # training definition
                    "unloaded_frac_norm5n_tail": (float((L[-tail:] < LOAD_N).mean())),

                    "support_all": float(sup.mean()),
                    "valid_support_s": float(sup.sum() * DT),
                    "upright_min": float(np.min(R["up"])),
                    "terminated": int(R["term"]),
                    "app_ddq_tail": float(app_d[-tail:].mean()),
                    "meas_ddq_tail": float(mea_d[-tail:].mean()),
                    "app_per_wheel_tail": np.asarray(R["app"], float)[-tail:].mean(axis=0).tolist(),
                    "meas_per_wheel_tail": np.asarray(R["meas"], float)[-tail:].mean(axis=0).tolist(),
                    "worst_1s_wz": float(min(yaw[i:i + int(1.0 / DT)].mean()
                                             for i in range(0, len(yaw) - int(1.0 / DT) + 1))),
                    "worst_1s_vx": float(min(vx[i:i + int(1.0 / DT)].mean()
                                             for i in range(0, len(vx) - int(1.0 / DT) + 1))),
                }
                if abs(wz_cmd) > 1e-9:
                    out["yaw_track"] = out["wz_tail"] / wz_cmd
                if abs(vx_cmd) > 1e-9:
                    out["vx_track"] = out["vx_tail"] / vx_cmd
                if brake:
                    i0 = int(brake_at / DT)
                    seg = vx[i0:]
                    out["brake_stop_dist_m"] = float(np.abs(seg).sum() * DT)
                    out["brake_settle_s"] = None
                    below = np.where(np.abs(seg) < G_BRAKE_VX)[0]
                    if len(below):
                        out["brake_settle_s"] = float(below[0] * DT)
                return out

            def gates(case, j):
                """Return (stage1_ok, tracking_ok, reason)."""
                s1 = (j["terminated"] == 0 and j["support_tail"] >= G_SUPPORT
                      and j["upright_min"] >= G_UPRIGHT)
                if not s1:
                    r = []
                    if j["terminated"]:
                        r.append("terminated")
                    if j["support_tail"] < G_SUPPORT:
                        r.append("support")
                    if j["upright_min"] < G_UPRIGHT:
                        r.append("upright")
                    return s1, None, "+".join(r)
                if case == "hold":
                    ok = (abs(j["vx_tail"]) < G_HOLD_VX and abs(j["wz_tail"]) < G_HOLD_WZ)
                    return s1, ok, "hold" if not ok else "ok"
                if case.startswith("turn"):
                    t = j.get("yaw_track")
                    ok = t is not None and G_YAW_TRACK_LO <= t <= G_YAW_TRACK_HI
                    return s1, ok, "yaw_track" if not ok else "ok"
                if case.startswith("straight"):
                    t = j.get("vx_track")
                    ok = t is not None and G_VX_TRACK_LO <= t <= G_VX_TRACK_HI
                    return s1, ok, "vx_track" if not ok else "ok"
                if case.startswith("arc"):
                    ty, tv = j.get("yaw_track"), j.get("vx_track")
                    ok = (ty is not None and G_YAW_TRACK_LO <= ty <= G_YAW_TRACK_HI
                          and tv is not None and G_VX_TRACK_LO <= tv <= G_VX_TRACK_HI)
                    return s1, ok, "ok" if ok else "yaw_or_vx_track"
                if case.startswith("brake"):
                    ok = abs(j["vx_tail"]) < G_BRAKE_VX
                    return s1, ok, "brake_not_stopped" if not ok else "ok"
                return s1, None, "unknown"

            def const(vx, wz):
                return lambda t: [vx, 0.0, wz]

            def brake_fn(vx):
                return lambda t: [vx, 0.0, 0.0] if t < BRAKE_HOLD_S else [0.0, 0.0, 0.0]

            CASES = [
                ("hold", 10.0, const(0.0, 0.0), False),
                ("turn_left_0.05", 12.0, const(0.0, +0.05), False),
                ("turn_right_0.05", 12.0, const(0.0, -0.05), False),
                ("turn_left_0.10", 12.0, const(0.0, +0.10), False),
                ("turn_right_0.10", 12.0, const(0.0, -0.10), False),
            ]
            if not args.skip_speed_tier:
                CASES += [
                    ("straight_fwd_0.25", 12.0, const(+STRAIGHT_TIER, 0.0), False),
                    ("straight_bwd_0.25", 12.0, const(-STRAIGHT_TIER, 0.0), False),
                    ("arc_left_0.25", 12.0, const(ARC_TIER[0], +ARC_TIER[1]), False),
                    ("arc_right_0.25", 12.0, const(ARC_TIER[0], -ARC_TIER[1]), False),
                    ("brake_fwd_0.25", 10.0, brake_fn(+STRAIGHT_TIER), True),
                    ("brake_bwd_0.25", 10.0, brake_fn(-STRAIGHT_TIER), True),
                ]

            report = {"frozen_gates": {
                          "hold_vx": G_HOLD_VX, "hold_wz": G_HOLD_WZ,
                          "yaw_track": [G_YAW_TRACK_LO, G_YAW_TRACK_HI],
                          "vx_track": [G_VX_TRACK_LO, G_VX_TRACK_HI],
                          "support": G_SUPPORT, "upright": G_UPRIGHT,
                          "brake_vx": G_BRAKE_VX, "tail_s": TAIL_S,
                          "load_n": LOAD_N, "settle_s": SETTLE_S,
                          "brake_hold_s": BRAKE_HOLD_S,
                          "turn_tiers": list(TURN_TIERS),
                          "straight_tier": STRAIGHT_TIER,
                          "arc_tier": list(ARC_TIER),
                          "frozen": True},
                      "protocol": ("env.seed()+env.reset()+fixed settle per repeat; "
                                   "mean/min/max across seeds; NO snapshot forks "
                                   "(Isaac/PhysX fork non-reproducible)"),
                      "seeds": seeds, "cases": [c[0] for c in CASES],
                      "candidates": []}

            for label, path in cks:
                runner.load(str(path), map_location=args.device)
                policy = runner.get_inference_policy(device=args.device)
                it = runner.current_learning_iteration
                print()
                print("########## %s (iter %s) ##########" % (label, it), flush=True)
                per_seed = []
                for sd in seeds:
                    rec = {"seed": sd, "cases": {}}
                    for case, secs, fn, is_brake in CASES:
                        obs = settle(sd)
                        R = run_case(obs, fn, secs)
                        j = judge(R, fn(0.0), secs, brake=is_brake,
                                  brake_at=BRAKE_HOLD_S)
                        s1, tk, reason = gates(case, j)
                        j["stage1_ok"] = bool(s1)
                        j["tracking_ok"] = (None if tk is None else bool(tk))
                        j["verdict"] = ("INVALID_STAGE1" if not s1
                                        else ("PASS" if tk else "FAIL"))
                        j["reason"] = reason
                        rec["cases"][case] = j
                    # zero-command differential bias from the hold case
                    hold_app = rec["cases"]["hold"]["app_ddq_tail"]
                    rec["zero_command_ddq_bias"] = hold_app
                    per_seed.append(rec)
                    print("  seed %d" % sd, flush=True)
                    for case, _s, _f, _b in CASES:
                        j = rec["cases"][case]
                        print("    %-18s %-15s sup=%.3f up=%.3f term=%d "
                              "vx=%+.4f wz=%+.4f trk(vx/yaw)=%s/%s"
                              % (case, j["verdict"], j["support_tail"],
                                 j["upright_min"], j["terminated"], j["vx_tail"],
                                 j["wz_tail"],
                                 ("%+.2f" % j["vx_track"]) if "vx_track" in j else "-",
                                 ("%+.2f" % j["yaw_track"]) if "yaw_track" in j else "-"),
                              flush=True)

                entry = {"label": label, "iteration": int(it),
                         "zero_command_ddq_bias": {
                             "mean": float(np.mean([r["zero_command_ddq_bias"]
                                                    for r in per_seed])),
                             "values": [r["zero_command_ddq_bias"] for r in per_seed]},
                         "cases": {}}
                for case, _s, _f, _b in CASES:
                    js = [r["cases"][case] for r in per_seed]
                    entry["cases"][case] = {
                        "verdicts": [j["verdict"] for j in js],
                        "reasons": [j["reason"] for j in js],
                        "pass_count": int(sum(1 for j in js if j["verdict"] == "PASS")),
                        "support_min": float(min(j["support_tail"] for j in js)),
                        "upright_min": float(min(j["upright_min"] for j in js)),
                        "term_any": int(max(j["terminated"] for j in js)),
                        "vx_tail_mean": float(np.mean([j["vx_tail"] for j in js])),
                        "wz_tail_mean": float(np.mean([j["wz_tail"] for j in js])),
                        "yaw_track_mean": (float(np.mean([j["yaw_track"] for j in js
                                                         if "yaw_track" in j]))
                                           if any("yaw_track" in j for j in js) else None),
                        "vx_track_mean": (float(np.mean([j["vx_track"] for j in js
                                                        if "vx_track" in j]))
                                          if any("vx_track" in j for j in js) else None),
                        "brake_stop_dist_m": (float(np.mean([j["brake_stop_dist_m"] for j in js]))
                                              if "brake_stop_dist_m" in js[0] else None),
                        "app_ddq_tail": float(np.mean([j["app_ddq_tail"] for j in js])),
                        "meas_ddq_tail": float(np.mean([j["meas_ddq_tail"] for j in js])),
                    }
                report["candidates"].append(entry)
                npass = sum(1 for c in entry["cases"].values() if c["pass_count"] == len(seeds))
                print("  ==> %s: %d/%d cases pass on ALL seeds, zero_bias=%+.4f"
                      % (label, npass, len(entry["cases"]),
                         entry["zero_command_ddq_bias"]["mean"]), flush=True)

            # ---- summary + auto decision input ----
            print()
            print("=== SUMMARY (frozen gates, per-case pass on all seeds) ===")
            print("  %-12s %-6s %-10s %-10s %-10s %-10s %s"
                  % ("cand", "pass", "hold", "turn", "straight", "arc", "brake"))
            summary = {}
            for e in report["candidates"]:
                c = e["cases"]
                def pk(pref):
                    ks = [k for k in c if k.startswith(pref)]
                    if not ks:
                        return "-"
                    return "%d/%d" % (sum(1 for k in ks if c[k]["pass_count"] == len(seeds)), len(ks))
                summary[e["label"]] = {"pass_total": sum(
                    1 for k in c if c[k]["pass_count"] == len(seeds)),
                    "n_cases": len(c)}
                print("  %-12s %-6s %-10s %-10s %-10s %-10s %s"
                      % (e["label"], "%d/%d" % (summary[e["label"]]["pass_total"], len(c)),
                         pk("hold"), pk("turn"), pk("straight"), pk("arc"), pk("brake")))
            report["summary"] = summary
            (args.out / "acceptance_v2.json").write_text(json.dumps(report, indent=1, default=str))
            print()
            print("ACCEPTANCE_V2_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
