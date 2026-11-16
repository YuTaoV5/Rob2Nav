#!/usr/bin/env python3
"""AS2-W rolling: DIFFERENTIAL-ONLY wheel torque feedforward, evaluated on the frozen matrix.

Mechanism this addresses (GOAL_STATUS 70, measured with r11_isaac_exec_probe.py)
  On Isaac the wheel drive is NOT torque saturated: 0.0% of frames at the +-13.5 Nm cap,
  peak only 4.47 Nm (33% of budget).  Yet a 0.49 rad/s target error produces just
  2.95 Nm from the kd=6 law, so the wheel settles at 13% of the commanded speed and the
  differential delivery is exec 0.25.  Raising kd was already refuted (kd=12 -> 2/11 plus a
  wheel speed limit cycle) because kd multiplies the ERROR, which amplifies the policy's
  chatter.  A feedforward multiplies the TARGET instead, so it does not sit in the error loop.

Why differential-only
  A naive velocity feedforward (sp = target*(1+c/kd)) biases the steady-state speed of every
  wheel, so straight-line vx tracking would jump (it currently sits at 0.85).  This script
  instead applies feedforward torque ONLY along the differential axis, driven by the policy's
  own differential target:
        D     = d4(target)                    # FR+RR-FL-RL
        tau_i = kd*(target_i - dq_i) + s_i * c * |D|/2 ,  s = (-1,-1,+1,+1) for FL,FR,RL,RR
  Common-mode (straight / hold / brake) has D ~ 0, so those regimes are untouched by
  construction and act as in-run regression controls.

Guarantees
  * the actuator still clamps at +-13.5 Nm, so the 90% torque hard limit is unchanged
  * no integrator, no filter state -> the 4-calls-per-policy-step cadence cannot compound
  * legs untouched
  * frozen gates unchanged; this script only adds an arm
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
_CONTRACT_APPLIED = None  # module-level so build() and the dispatch patch can both read it
WHEEL_LIMIT = 13.5
DT = 0.02
SETTLE_S = 2.0
LOAD_N = 5.0
TAIL_S = 3.0
BRAKE_HOLD_S = 5.0
G_HOLD_VX, G_HOLD_WZ = 0.02, 0.02
G_TRACK_LO, G_TRACK_HI = 0.70, 1.30
G_SUPPORT, G_UPRIGHT, G_BRAKE_VX = 0.95, 0.85, 0.02
STRAIGHT_TIER = 0.25
ARC_TIER = (0.25, 0.05)


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
    # the wheel action scale lives on the ACTION TERM, and robust_config does not carry it
    cfg.actions.wheel_vel.scale = WHEEL_SCALE
    if _CONTRACT_APPLIED is not None and not _CONTRACT_APPLIED["gyro_enabled"]:
        from as2w_isaac_robust_env import DelayedVelocity as _DV
        cfg.actions.wheel_vel.class_type = _DV
    print("AS2W_WHEEL_DISPATCH_CFG scale=%.3f kd=%.3f class=%s" % (WHEEL_SCALE, WHEEL_KD, cfg.actions.wheel_vel.class_type.__name__), flush=True)
    # GOAL_STATUS 74.3 hypothesis: the wheel differential delivers ~half of its target
    # (exec 0.49) yet the body barely yaws, so the reaction is being absorbed somewhere
    # other than the ground.  The wheels hang off position-PD-held legs (KP 55,
    # hip/thigh 54 and calf 81 Nm limits) that can easily absorb the yaw reaction.
    # This knob scales the LEG position stiffness only (wheels untouched) so the
    # hypothesis can be tested as a single variable.
    _lk = float(getattr(args, "leg_kp_scale", 1.0))
    if _lk != 1.0:
        for _grp in ("hip", "thigh", "calf"):
            if _grp in cfg.scene.robot.actuators:
                cfg.scene.robot.actuators[_grp].stiffness *= _lk
        print("AS2W_LEG_KP_SCALE %.3f applied to hip/thigh/calf" % _lk, flush=True)
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
    cfg.rewards.rolling_knee_margin = RewardTermCfg(
        func=rolling_knee_margin_cost, weight=-3., params={"margin_m": .10})
    apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    # Goal section 4 fixed-load tier.  Done as a startup mass event on the base link instead
    # of mutating the PhysX view after spawn: the event API is supported and applies before
    # the first step, and the readback below records the realised mass rather than assuming it.
    _pay = float(getattr(args, "payload_kg", 0.0))
    if abs(_pay) > 1e-9:
        from isaaclab.managers import EventTermCfg as _EvCfg, SceneEntityCfg
        import isaaclab.envs.mdp as _mdp
        cfg.events.as2w_payload = _EvCfg(
            func=_mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "mass_distribution_params": (_pay, _pay), "operation": "add"})
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
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--seeds", default="7393,311")
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--ff-gain", type=float, default=0.0,
                    help="differential feedforward Nm per rad/s of COMMANDED wz; 0 = off")
    ap.add_argument("--ff-cap", type=float, default=4.0,
                    help="hard cap on the feedforward torque itself, Nm")
    ap.add_argument("--leg-kp-scale", type=float, default=1.0,
                    help="scale leg (hip/thigh/calf) position stiffness; 1.0 = contract")
    ap.add_argument("--allow-no-gyro", action="store_true",
                    help="permit yaw_kp=0 and ff_gain=0. Without this flag a zero gyro gain is refused, "
                         "because yaw_kp defaults to 0 and a missing --yaw-kp would otherwise silently run "
                         "the UNMODIFIED baseline while reporting itself as gyro-1")
    ap.add_argument("--contract", default="", help="wheel_contract_v1.json path; sets wheel scale and gain from the named entry")
    ap.add_argument("--contract-name", default="", help="entry name inside --contract")
    ap.add_argument("--yaw-kp", type=float, default=0.0,
                    help="bounded gyro yaw feedback, Nm per (rad/s) of (cmd - measured) yaw rate")
    ap.add_argument("--payload-kg", type=float, default=0.0,
                    help="fixed payload added to the base link (goal section 4: 0 / +6 kg tiers)")
    ap.add_argument("--drop-rate", type=float, default=0.0,
                    help="per-policy-step probability that the action frame is LOST, in which case "
                         "the last valid action is held (goal section 2: whole-frame loss + "
                         "last-valid-frame hold)")
    ap.add_argument("--drop-burst", type=int, default=1,
                    help="when a drop occurs, hold the last valid action for this many consecutive "
                         "policy steps (goal section 2 burst loss; default 1 = IID)")
    ap.add_argument("--drop-seed", type=int, default=1234,
                    help="RNG seed for the frame-drop pattern so a run is reproducible")
    ap.add_argument("--action-delay-steps", type=int, default=0,
                    help="communication latency in POLICY steps: the robot applies the action "
                         "issued N steps ago (goal section 4 latency tier)")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    ap.add_argument("--obs-freeze", type=int, default=0, help="C5 observation distortion: once every N frames the policy is shown the previous observation instead of the fresh one (stalled sensor stream)")
    ap.add_argument("--obs-stale-steps", type=int, default=0, help="C5 observation distortion: the policy is shown the observation from K policy steps ago while the simulator keeps advancing (delayed or duplicated frame stream)")
    ap.add_argument("--leg-bias-rad", default="", help="EXPERIMENTAL posture bias: 12 comma-separated joint-space radians added to the leg action target, divided by the 0.35 action scale; empty disables it; not part of the default contract")
    args = ap.parse_args()
    for a in ("usd", "urdf"):
        setattr(args, a, getattr(args, a).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"
    cks = []
    if float(getattr(args, "yaw_kp", 0.0)) <= 0.0 and float(getattr(args, "ff_gain", 0.0)) <= 0.0 \
            and not bool(getattr(args, "allow_no_gyro", False)):
        raise SystemExit(
            "AS2W_GYRO_GUARD: refusing to run with yaw_kp=%r and ff_gain=%r.\n"
            "  The gyro feed-forward would be UNMODIFIED baseline: the early-return branch fires, tau and ff\n"
            "  are never computed and both print as 0, which is exactly what silently happened before.\n"
            "  Pass --yaw-kp 60 --ff-cap 4 to actually evaluate gyro-1, or --allow-no-gyro if the baseline\n"
            "  is genuinely what you want." % (getattr(args, "yaw_kp", None), getattr(args, "ff_gain", None)))
    print("AS2W_GYRO_GUARD OK yaw_kp=%.2f ff_cap=%.2f ff_gain=%.2f"
          % (float(getattr(args, "yaw_kp", 0.0)), float(getattr(args, "ff_cap", 0.0)),
             float(getattr(args, "ff_gain", 0.0))), flush=True)

    global WHEEL_KD, WHEEL_SCALE, _CONTRACT_APPLIED
    _CONTRACT_APPLIED = None
    if str(getattr(args, "contract", "")).strip():
        import json as _json
        from pathlib import Path as _Path
        _c = _json.loads(_Path(args.contract).read_text(encoding="utf-8"))
        _want = _c["contracts"][args.contract_name]
        WHEEL_KD = float(_want["wheel_kd"])
        WHEEL_SCALE = float(_want["wheel_action_scale"])
        _CONTRACT_APPLIED = {"name": args.contract_name, "wheel_kd": WHEEL_KD, "wheel_scale": WHEEL_SCALE, "gyro_enabled": bool(_want.get("gyro_enabled", False))}
        print("AS2W_CONTRACT_APPLIED " + _json.dumps(_CONTRACT_APPLIED), flush=True)

    for spec in args.checkpoints.split(","):
        lab, _, p = spec.partition("=")
        cks.append((lab, Path(p)))
    seeds = [int(s) for s in args.seeds.split(",")]

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        (env, Wrapper, PPOCfg, check_v, create_runner, handle_dep) = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            payload_info = {"requested_kg": float(args.payload_kg), "applied": bool(abs(float(args.payload_kg)) > 1e-9),
                            "method": "startup mass event on base_link, operation=add"}
            try:
                masses = np.asarray(robot.root_physx_view.get_masses()).astype(np.float64).reshape(-1)
                body_names = list(robot.body_names)
                base_i = body_names.index("base_link") if "base_link" in body_names else 0
                payload_info.update({
                    "base_body_name": body_names[base_i],
                    "base_mass_after_kg": float(masses[base_i]),
                    "total_mass_after_kg": float(masses.sum()),
                    "readback_ok": True,
                })
            except Exception as exc:                      # readback must never break the run
                payload_info["readback_ok"] = False
                payload_info["readback_error"] = "%s: %s" % (type(exc).__name__, exc)
            print("AS2W_PAYLOAD %s" % json.dumps(payload_info), flush=True)
            wj = env.as2w_joint_ids[12:]
            cmdterm = env.command_manager.get_term("base_velocity")
            wterm = env.action_manager.get_term("wheel_vel")
            # left wheels vs right wheels (FL,FR,RL,RR).  See the sign note at tau_ff below:
            # the previous (-1,-1,+1,+1) was a front/rear couple and produced no yaw.
            S = torch.tensor([-1., 1., -1., 1.], device=env.device)

            st = {"applied": torch.zeros(1, 4, device=env.device),
                  "req": torch.zeros(1, 4, device=env.device),
                  "tau": torch.zeros(1, 4, device=env.device),
                  "ff": float(args.ff_gain), "calls": 0,
                  "wz_cmd": 0.0, "wz_meas": 0.0, "ff_cap": float(args.ff_cap),
                  "yaw_kp": float(getattr(args, "yaw_kp", 0.0)),
                  "tau_abs_max": 0.0, "ff_abs_max": 0.0, "cap_hits": 0}

            def patched_apply():
                # No state is carried across calls: the feedforward is a pure function of the
                # current differential target, so the once-per-substep cadence is harmless.
                st["calls"] += 1
                tgt = wterm.delayed()
                st["req"] = tgt.clone()
                dq = robot.data.joint_vel.torch[:, wj]
                C = st["ff"]
                # The early return must also check the gyro gain: with --ff-gain 0 the
                # previous guard skipped the whole path, so --yaw-kp 30/60 silently ran the
                # UNMODIFIED baseline (evidence: ff_max=0.00 and caps=0 in both arms).
                if C <= 0.0 and st["yaw_kp"] <= 0.0:
                    st["applied"] = tgt.clone()
                    robot.set_joint_velocity_target_index(target=tgt,
                                                          joint_ids=wterm._joint_ids)
                    return
                # v2 correction (GOAL_STATUS 73).  The first version drove the feedforward
                # from the POLICY's differential target.  delayed() is in the action scale
                # (x16), so |d4(target)| reached ~42 rad/s and c=6 produced 124 Nm against a
                # 13.5 Nm cap; worse, the policy emits a non-zero differential even at zero
                # command, so the feedforward amplified exactly the zero-command wheel-bias
                # the goal forbids.  The feedforward is now driven by the COMMANDED yaw rate,
                # which is exactly 0 in hold and in every straight case by construction, and
                # it is clamped on its own (not only by the actuator).
                _err = st["wz_cmd"] - st["wz_meas"]
                _raw = C * st["wz_cmd"] + st["yaw_kp"] * _err
                tau_ff = (S.unsqueeze(0) * _raw).clamp(-st["ff_cap"], st["ff_cap"])
                tau_err = WHEEL_KD * (tgt - dq)
                tau = tau_err + tau_ff
                sat = tau.abs() > WHEEL_LIMIT
                st["cap_hits"] += int(sat.any(dim=1).sum().item())
                tau = tau.clamp(-WHEEL_LIMIT, WHEEL_LIMIT)
                st["ff_abs_max"] = max(st["ff_abs_max"], float(tau_ff.abs().max()))
                st["tau_abs_max"] = max(st["tau_abs_max"], float(tau.abs().max()))
                sp = dq + tau / WHEEL_KD
                st["tau"] = tau.clone()
                st["applied"] = sp.clone()
                robot.set_joint_velocity_target_index(target=sp,
                                                      joint_ids=wterm._joint_ids)

            wterm.apply_actions = patched_apply

            ppo = PPOCfg(); ppo.seed, ppo.device = seeds[0], args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)

            from collections import deque
            _bias12 = None
            if str(getattr(args, "leg_bias_rad", "")).strip():
                _v = [float(x) for x in str(args.leg_bias_rad).split(",") if x.strip()]
                assert len(_v) == 12, "leg_bias_rad needs 12 values"
                _bias12 = torch.tensor(_v, dtype=torch.float32, device=env.device) / 0.35
                print("AS2W_LEG_BIAS_ON absmax_rad=%.4f" % max(abs(x) for x in _v), flush=True)
            _delay = int(getattr(args, "action_delay_steps", 0))
            _hist = deque(maxlen=max(1, _delay + 1))

            import random as _random
            _drop = float(getattr(args, "drop_rate", 0.0))
            _rng = _random.Random(int(getattr(args, "drop_seed", 1234)))
            _burst = max(1, int(getattr(args, "drop_burst", 1)))
            _last_valid = {"a": None, "drops": 0, "steps": 0, "remaining": 0}

            def delayed_step(a):
                """Apply a[-(_delay+1)] instead of a; with _delay 0 this is a no-op.

                With --drop-rate > 0 a frame is lost with that probability and the robot keeps
                executing the last VALID action (last-valid-frame hold), which is the failure
                mode the training guide asks to model.  The policy always sees the real
                resulting observation.
                """
                _last_valid["steps"] += 1
                if _bias12 is not None:
                    a = a.clone()
                    a[:, :12] = a[:, :12] + _bias12
                applied = a
                # A burst begins with probability _drop and then holds for _burst consecutive
                # steps.  With _burst == 1 this is exactly the IID drop tested earlier, so the
                # burstiness is the only variable that changes.
                if _last_valid["remaining"] > 0 and _last_valid["a"] is not None:
                    applied = _last_valid["a"]
                    _last_valid["remaining"] -= 1
                    _last_valid["drops"] += 1
                elif _drop > 0.0 and _last_valid["a"] is not None and _rng.random() < _drop:
                    applied = _last_valid["a"]
                    _last_valid["remaining"] = _burst - 1
                    _last_valid["drops"] += 1
                else:
                    _last_valid["a"] = a.clone()
                if _delay <= 0:
                    return wrapped.step(applied)
                _hist.append(applied.clone())
                if len(_hist) <= _delay:
                    return wrapped.step(applied)    # still filling the pipeline
                return wrapped.step(_hist[0])

            def settle(seed):
                env.seed(seed); env.reset()
                _hist.clear()
                cmdterm.vel_command_b[:] = 0.0
                st["calls"] = 0
                obs = wrapped.get_observations()
                for _ in range(int(SETTLE_S / DT)):
                    # C5 observation distortion: the SIMULATOR always advances with the true state, but the
                    # policy may be shown a stale or frozen observation, which is what a stalled, delayed or
                    # duplicated sensor stream looks like to it.
                    if args.obs_stale_steps or args.obs_freeze:
                        _ohist.append(obs)
                        if args.obs_stale_steps and len(_ohist) > args.obs_stale_steps:
                            feed = _ohist[-1 - args.obs_stale_steps]
                        elif args.obs_freeze and _frozen is not None and (k % args.obs_freeze) != 0:
                            feed = _frozen
                        else:
                            feed, _frozen = obs, obs
                        if len(_ohist) > 64:
                            _ohist.pop(0)
                    else:
                        feed = obs
                    with torch.inference_mode():
                        a = policy(feed)
                    obs, _rew, _don, _info = delayed_step(a)
                return obs

            def run_case(obs, cmd_fn, seconds):
                st["calls"] = 0
                c0 = st["calls"]
                n = int(seconds / DT)
                R = {k: [] for k in ("yaw", "vx", "load", "app", "meas", "tau", "up")}
                term = 0
                _ohist, _frozen = [], None
                for k in range(n):
                    c = cmd_fn(k * DT)
                    st["wz_cmd"] = float(c[2])
                    st["wz_meas"] = float(robot.data.root_ang_vel_b.torch[0, 2])
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [c], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _rew, _don, _info = delayed_step(a)
                    # termination must actually be read: the earlier version inspected a
                    # list with isinstance(_, tuple) and therefore never set this flag,
                    # which could have scored a fallen episode as PASS.
                    if bool(_don[0]):
                        term = 1
                    R["yaw"].append(float(robot.data.root_ang_vel_b.torch[0, 2]))
                    R["vx"].append(float(robot.data.root_lin_vel_b.torch[0, 0]))
                    R["app"].append(st["applied"][0].detach().cpu().numpy().tolist())
                    R["meas"].append(robot.data.joint_vel.torch[0, wj]
                                     .detach().cpu().numpy().tolist())
                    R["tau"].append(st["tau"][0].detach().cpu().numpy().tolist())
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]]
                    R["load"].append(torch.linalg.norm(cf, dim=1).detach().cpu()
                                     .numpy().tolist())
                    g = robot.data.projected_gravity_b.torch[0]
                    R["up"].append(float(-g[2]))
                R["term"] = term
                R["cap_hits"] = st["cap_hits"]
                R["tau_abs_max"] = st["tau_abs_max"]
                R["ff_abs_max"] = st["ff_abs_max"]
                return R

            def judge(R, cmd, brake=False, brake_at=0.0):
                tail = max(1, int(TAIL_S / DT))
                yaw = np.asarray(R["yaw"], float); vx = np.asarray(R["vx"], float)
                L = np.asarray(R["load"], float)
                sup = (L > LOAD_N).all(axis=1)
                app_d = d4(R["app"]); mea_d = d4(R["meas"])
                wz_cmd, vx_cmd = float(cmd[2]), float(cmd[0])
                out = {"cmd": [float(x) for x in cmd],
                       "vx_tail": float(vx[-tail:].mean()),
                       "wz_tail": float(yaw[-tail:].mean()),
                       "support_tail": float(sup[-tail:].mean()),
                       "upright_min": float(np.min(R["up"])),
                       "terminated": int(R["term"]),
                       "exec_tail": (float(np.abs(mea_d[-tail:]).mean()
                                           / np.abs(app_d[-tail:]).mean())
                                     if np.abs(app_d[-tail:]).mean() > 1e-9 else None),
                       "ddq_app_tail": float(app_d[-tail:].mean()),
                       "ddq_meas_tail": float(mea_d[-tail:].mean()),
                       "tau_abs_max": float(R["tau_abs_max"]),
                       "ff_abs_max": float(R["ff_abs_max"]),
                       "cap_hits": int(R["cap_hits"])}
                if abs(wz_cmd) > 1e-9:
                    out["yaw_track"] = out["wz_tail"] / wz_cmd
                if abs(vx_cmd) > 1e-9:
                    out["vx_track"] = out["vx_tail"] / vx_cmd
                if brake:
                    i0 = int(brake_at / DT); seg = vx[i0:]
                    out["brake_stop_dist_m"] = float(np.abs(seg).sum() * DT)
                return out

            def gates(case, j):
                s1 = (j["terminated"] == 0 and j["support_tail"] >= G_SUPPORT
                      and j["upright_min"] >= G_UPRIGHT)
                if not s1:
                    return False
                if case == "hold":
                    return abs(j["vx_tail"]) < G_HOLD_VX and abs(j["wz_tail"]) < G_HOLD_WZ
                if case.startswith("turn"):
                    t = j.get("yaw_track")
                    return t is not None and G_TRACK_LO <= t <= G_TRACK_HI
                if case.startswith("straight"):
                    t = j.get("vx_track")
                    return t is not None and G_TRACK_LO <= t <= G_TRACK_HI
                if case.startswith("arc"):
                    ty, tv = j.get("yaw_track"), j.get("vx_track")
                    return (ty is not None and G_TRACK_LO <= ty <= G_TRACK_HI
                            and tv is not None and G_TRACK_LO <= tv <= G_TRACK_HI)
                if case.startswith("brake"):
                    return abs(j["vx_tail"]) < G_BRAKE_VX
                return True

            def const(vx, wz):
                return lambda t: [vx, 0.0, wz]
            def brake_fn(vx):
                return lambda t: [vx, 0.0, 0.0] if t < BRAKE_HOLD_S else [0.0, 0.0, 0.0]

            CASES = [("hold", 10.0, const(0.0, 0.0), False),
                     ("turn_left_0.05", 12.0, const(0.0, +0.05), False),
                     ("turn_right_0.05", 12.0, const(0.0, -0.05), False),
                     ("turn_left_0.10", 12.0, const(0.0, +0.10), False),
                     ("turn_right_0.10", 12.0, const(0.0, -0.10), False),
                     ("straight_fwd_0.25", 12.0, const(+STRAIGHT_TIER, 0.0), False),
                     ("straight_bwd_0.25", 12.0, const(-STRAIGHT_TIER, 0.0), False),
                     ("arc_left_0.25", 12.0, const(ARC_TIER[0], +ARC_TIER[1]), False),
                     ("arc_right_0.25", 12.0, const(ARC_TIER[0], -ARC_TIER[1]), False),
                     ("brake_fwd_0.25", 10.0, brake_fn(+STRAIGHT_TIER), True),
                     ("brake_bwd_0.25", 10.0, brake_fn(-STRAIGHT_TIER), True)]

            report = {"payload": payload_info,
                      "action_delay_steps": int(getattr(args, "action_delay_steps", 0)),
                      "action_delay_ms": int(getattr(args, "action_delay_steps", 0)) * 20,
                      "drop_rate": float(getattr(args, "drop_rate", 0.0)),
                      "drop_burst": int(getattr(args, "drop_burst", 1)),
                      "drop_seed": int(getattr(args, "drop_seed", 1234)),
                      "action_delay_note": "the plant executes the action issued N policy steps ago; the policy sees the real resulting observation",
                      "arm": {"ff_gain_nm_per_rad_s": float(args.ff_gain),
                              "axis": "LEFT-vs-RIGHT differential (s = -1,+1,-1,+1), driven by the COMMANDED wz",
                              "drive": "wz_cmd (exactly 0 in hold and every straight case)",
                              "ff_cap_nm": float(args.ff_cap),
                              "yaw_kp": float(getattr(args, "yaw_kp", 0.0)),
                              "yaw_feedback": "bounded, state-free (no integrator): K*(wz_cmd - wz_meas)",
                              "common_mode_effect": "none by construction",
                              "wheel_limit_nm": WHEEL_LIMIT, "legs_touched": bool(float(getattr(args, "leg_kp_scale", 1.0)) != 1.0),
                              "leg_kp_scale": float(getattr(args, "leg_kp_scale", 1.0)),
                              "state_free": True},
                      "seeds": seeds, "candidates": []}

            for label, path in cks:
                runner.load(str(path), map_location=args.device)
                policy = runner.get_inference_policy(device=args.device)
                print()
                print("########## %s ff=%.1f ##########" % (label, args.ff_gain), flush=True)
                per_seed = []
                for sd in seeds:
                    rec = {"seed": sd, "cases": {}}
                    for case, secs, fn, is_brake in CASES:
                        obs = settle(sd)
                        R = run_case(obs, fn, secs)
                        j = judge(R, fn(0.0), brake=is_brake, brake_at=BRAKE_HOLD_S)
                        j["verdict"] = "PASS" if gates(case, j) else "FAIL"
                        rec["cases"][case] = j
                    per_seed.append(rec)
                    print("  seed %d" % sd, flush=True)
                    for case, _s, _f, _b in CASES:
                        j = rec["cases"][case]
                        print("    %-18s %-5s sup=%.3f up=%.3f vx=%+.4f wz=%+.4f "
                              "trk=%s/%s exec=%s tau_max=%.2f ff_max=%.2f caps=%d term=%d"
                              % (case, j["verdict"], j["support_tail"], j["upright_min"],
                                 j["vx_tail"], j["wz_tail"],
                                 ("%+.2f" % j["vx_track"]) if "vx_track" in j else "-",
                                 ("%+.2f" % j["yaw_track"]) if "yaw_track" in j else "-",
                                 ("%.3f" % j["exec_tail"]) if j["exec_tail"] else "-",
                                 j["tau_abs_max"], j["ff_abs_max"], j["cap_hits"],
                                 j["terminated"]), flush=True)
                entry = {"label": label, "cases": {}}
                for case, _s, _f, _b in CASES:
                    js = [r["cases"][case] for r in per_seed]
                    entry["cases"][case] = {
                        "verdicts": [j["verdict"] for j in js],
                        "pass_all": all(j["verdict"] == "PASS" for j in js),
                        "support_min": float(min(j["support_tail"] for j in js)),
                        "upright_min": float(min(j["upright_min"] for j in js)),
                        "term_any": int(max(j["terminated"] for j in js)),
                        "yaw_track_mean": (float(np.mean([j["yaw_track"] for j in js
                                                         if "yaw_track" in j]))
                                           if any("yaw_track" in j for j in js) else None),
                        "vx_track_mean": (float(np.mean([j["vx_track"] for j in js
                                                        if "vx_track" in j]))
                                          if any("vx_track" in j for j in js) else None),
                        "exec_mean": (float(np.mean([j["exec_tail"] for j in js
                                                    if j["exec_tail"]]))
                                      if any(j["exec_tail"] for j in js) else None),
                        "tau_abs_max": float(max(j["tau_abs_max"] for j in js)),
                        "ff_abs_max": float(max(j["ff_abs_max"] for j in js)),
                        "cap_hits": int(max(j["cap_hits"] for j in js)),
                    }
                entry["frames_dropped"] = int(_last_valid["drops"])
                entry["frames_total"] = int(_last_valid["steps"])
                entry["pass_total"] = sum(1 for c in entry["cases"].values() if c["pass_all"])
                report["candidates"].append(entry)
                print("  ==> %s ff=%.1f: %d/%d pass"
                      % (label, args.ff_gain, entry["pass_total"], len(entry["cases"])),
                      flush=True)

            (args.out / "ff_arm.json").write_text(json.dumps(report, indent=1, default=str))
            print()
            print("FF_ARM_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
