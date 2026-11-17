#!/usr/bin/env python3
"""AS2-W rolling line: loaded four-wheel kd comparison -- ISAAC side.

What this measures and why
--------------------------
The wheel-end gain kd is the ONLY variable.  The action scale (5.0), the gyro switch (OFF),
the leg control (position PD, fixed stand action, kp 55 / kd 2.0), the posture, the wheel
speed target sequence and the +-13.5 N.m budget are IDENTICAL across the three arms, and are
read from the versioned contract deploy/as2w/contracts/wheel_contract_v1.json by the driver
that launches this script.

Two modes
    calibrate : gravity 0, robot spawned at z=2.0 m so it is genuinely airborne.  For each wheel
                FL/FR/RL/RR a +1 and a -1 one-hot wheel-speed target is dispatched.  This proves
                (a) the action channel -> wheel index mapping, (b) the joint index of each wheel
                (derived BY NAME, never a hard-coded 12..15), (c) the sign convention, and
                (d) that the contact-body indexing returns zero load when nothing touches.
                It also measures track width / wheelbase and writes the shared sequence file.
    grounded  : every case does its own reset, settles until standing is proven, and only then
                times the drive + brake phases.

Where the "final four-wheel target" is read
-------------------------------------------
Not from the action array and not from the config.  Every call of
`Articulation.set_joint_velocity_target_index` is intercepted (that IS the dispatch point, the
last write before PhysX sees the target) and recorded together with the number of calls made
per policy step.  In addition `env.scene.update(dt)` is wrapped: this env calls it once per
PHYSICS substep, so the recorder asserts the substep dt and stores the target seen by the
solver at every substep of every policy step.  The consistency check between those two is the
proof that nothing rewrites the target inside a policy step.

Torque
------
Isaac's implicit drive asks for tau = kd*(v_target - v).  That request is computed here and
reported as `tau_req`.  What PhysX actually applies is reported from
`get_dof_projected_joint_forces` and is labelled ESTIMATED, because the solver's drive force
cannot be separated from contact reaction with this API.  The hard ceiling is not assumed: it
is read back from the instantiated actuator (`effort_limit_sim`, 0.90 * URDF effort = 13.5 N.m).

Failures are kept
-----------------
A termination, an auto reset, an unfinished startup ramp, a failed rest gate or a target that
does not match the request marks the window VOID/FAILED and is written to the trace.  Nothing
is silently dropped and no value is replaced by a placeholder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEGS = tuple(f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
             for leg in ("FL", "FR", "RL", "RR"))[:12]
WHEEL_R_M = 0.086            # measured from the four collision STL bboxes (wheel_radius_measured.json)
LOAD_Z_N = 2.0               # per-wheel load counted as "carrying load" (same as the MuJoCo side)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("calibrate", "grounded"), required=True)
    ap.add_argument("--kd", type=float, required=True)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--seq", type=Path, default=None, help="sequence json (grounded mode)")
    ap.add_argument("--write-seq", type=Path, default=None, help="calibrate mode: write the sequence here")
    ap.add_argument("--states", default="A,B", help="grounded mode: standing initial states to run")
    ap.add_argument("--label", default="isaac")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--onehot-rad-s", type=float, default=1.0)
    ap.add_argument("--onehot-frames", type=int, default=12)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def build_env(args, physics):
    from types import SimpleNamespace

    from as2w_isaac_flat_env_all import FlatEnvironment  # noqa: F401  (import side effect: env registration)
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_training_safety import apply_safety

    class EvalCommands(TakeoverCommands):
        def _resample_command(self, env_ids):
            pass

    ns = SimpleNamespace(**vars(args))
    ns.profile = "pose-delay"
    ns.direction_focus = True
    ns.command_stage = "all"
    ns.zero_memory = True
    ns.zero_focus = True
    cfg = robust_config(ns, physics)
    cmd = cfg.commands.base_velocity
    cmd.class_type = EvalCommands
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.ranges.lin_vel_x = (-1.2, 1.2)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (-0.15, 0.15)
    cfg.episode_length_s = 120.0
    # ---- contract: only wheel kd varies -------------------------------------------------
    cfg.scene.robot.actuators["foot"].damping = float(args.kd)
    cfg.actions.wheel_vel.scale = float(args.scale)
    cfg.as2w_action_delay_max_steps = 0
    limits = apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    if args.mode == "calibrate":
        # AIRBORNE pass: the env builds its OWN SimulationContext from cfg.sim, so the gravity of the
        # standalone sim_cfg above does NOT reach it -- that is why the first attempt fell to the
        # floor while "airborne".  Zero gravity here is what actually suspends the robot.
        cfg.sim.gravity = (0.0, 0.0, 0.0)
        # The posture guard is meaningless while suspended; its components are recorded instead.
        # In GROUNDED mode the guard stays ACTIVE, so a real posture violation terminates the case
        # and is kept as failure evidence.
        cfg.terminations.as2w_stance_violation = None
    env = RobustEnvironment(cfg)
    return env, cfg, limits


def main() -> int:
    args = parse_args()
    os.environ["AS2W_GYRO_KP"] = "0"      # gyro OFF for all three arms
    os.environ["AS2W_GYRO_CAP"] = "4"
    os.environ["AS2W_GYRO_FF_GAIN"] = "0"
    gravity = (0.0, 0.0, 0.0) if args.mode == "calibrate" else (0.0, 0.0, -9.81)
    sim_cfg = SimulationCfg(dt=0.005, device=args.device, gravity=gravity)

    with launch_simulation(sim_cfg, args) as physics:
        from as2w_isaac_robust_env import robust_config  # noqa: F401  (keep symbol used)
        from isaaclab.assets import Articulation

        env, cfg, safety = build_env(args, physics)
        robot = env.scene["robot"]
        dt = float(env.step_dt)
        names = list(robot.joint_names)

        def t_(x):
            return x.torch if hasattr(x, "torch") else torch.as_tensor(x)

        # ---------------- indices, derived by NAME and then verified --------------------
        widx, bidx, lidx = [], [], []
        for w in WHEELS:
            if w not in names:
                raise RuntimeError("wheel joint %s missing from %s" % (w, names))
            widx.append(names.index(w))
        for leg in LEGS:
            if leg not in names:
                raise RuntimeError("leg joint %s missing" % leg)
            lidx.append(names.index(leg))
        bnames = list(robot.body_names)
        for w in WHEELS:
            if w not in bnames:
                raise RuntimeError("wheel body %s missing from %s" % (w, bnames))
            bidx.append(bnames.index(w))
        cs = env.scene["contact_forces"]
        cnames = [str(n) for n in getattr(cs, "body_names", [])]
        cidx = []
        for w in WHEELS:
            hit = [i for i, n in enumerate(cnames) if n == w or n.endswith("/" + w)]
            if len(hit) != 1:
                raise RuntimeError("contact body %s resolved to %s from %s" % (w, hit, cnames))
            cidx.append(hit[0])
        print("WHEEL_JOINT_INDEX %s" % json.dumps(dict(zip(WHEELS, widx))), flush=True)
        print("WHEEL_BODY_INDEX %s" % json.dumps(dict(zip(WHEELS, bidx))), flush=True)
        print("WHEEL_CONTACT_INDEX %s of %s" % (json.dumps(dict(zip(WHEELS, cidx))), cnames), flush=True)
        if [names[i] for i in widx] != list(WHEELS):
            raise RuntimeError("derived wheel indices do not read back the wheel names")
        if len(set(widx)) != 4 or len(set(cidx)) != 4:
            raise RuntimeError("wheel indices are not unique")

        # ---------------- effective gains / limits, read from the INSTANTIATED articulation --
        def group_ids(group):
            try:
                j = robot.actuators[group].joint_indices
                return [int(x) for x in torch.as_tensor(j).flatten().tolist()]
            except Exception:
                return None

        def vals(act, attr):
            for name in (attr, attr + "_sim"):
                v = getattr(act, name, None)
                if v is None:
                    continue
                try:
                    t = v.torch if hasattr(v, "torch") else v
                    return [round(float(x), 6) for x in torch.as_tensor(t).flatten().tolist()]
                except Exception:
                    return str(v)
            return None

        eff = {}
        for group, act in robot.actuators.items():
            eff[group] = {
                "joint_indices": group_ids(group),
                "stiffness": vals(act, "stiffness"),
                "damping": vals(act, "damping"),
                "effort_limit": vals(act, "effort_limit"),
                "velocity_limit": vals(act, "velocity_limit"),
            }
        print("RESOLVED_ACTUATORS %s" % json.dumps(eff), flush=True)
        foot_cfg = cfg.scene.robot.actuators["foot"]
        print("RESOLVED_FOOT_CFG stiffness=%s damping=%s effort_limit_sim=%s velocity_limit_sim=%s"
              % (foot_cfg.stiffness, foot_cfg.damping, foot_cfg.effort_limit_sim, foot_cfg.velocity_limit_sim),
              flush=True)
        gids = eff["foot"]["joint_indices"]
        if gids is None or eff["foot"]["effort_limit"] is None:
            raise RuntimeError("cannot resolve the foot actuator group: %s" % eff["foot"])
        budget = 13.5
        effort = []
        for w in WHEELS:
            local = gids.index(widx[WHEELS.index(w)])
            effort.append(float(eff["foot"]["effort_limit"][local]))
        print("WHEEL_EFFORT_CEILING_NM %s (from instantiated actuator, group-local %s)"
              % (json.dumps(effort), gids), flush=True)
        if any(abs(e - budget) > 1e-6 for e in effort):
            raise RuntimeError("wheel effort ceiling is %s, expected 13.5 N.m (90%% of the 15 N.m limit)" % effort)
        foot_damp = eff["foot"]["damping"]
        if foot_damp is None or any(abs(d - args.kd) > 1e-6 for d in foot_damp):
            raise RuntimeError("wheel damping resolved to %s, expected kd=%.3f" % (foot_damp, args.kd))
        leg_groups = {g: {"stiffness": eff[g]["stiffness"], "damping": eff[g]["damping"]}
                      for g in eff if g != "foot"}
        print("LEG_GROUP_GAINS %s (contract leg_kp=55, leg_kd=2.0)" % json.dumps(leg_groups), flush=True)

        # ---------------- geometry + posture components (world and body frame) ------------
        from as2w_training_safety import stance_violation

        bp0 = t_(robot.data.body_pos_w)[0][bidx].cpu().numpy()
        rq0 = t_(robot.data.root_quat_w)[0].cpu().numpy()
        w0, x0, y0, z0 = [float(v) for v in rq0]
        heading0 = float(np.arctan2(2.0 * (w0 * z0 + x0 * y0), 1.0 - 2.0 * (y0 * y0 + z0 * z0)))
        print("WHEEL_BODY_POS_W %s root_quat=%s heading_rad=%.5f"
              % (json.dumps({w: [round(float(v), 5) for v in bp0[i]] for i, w in enumerate(WHEELS)}),
                 [round(v, 5) for v in rq0], heading0), flush=True)
        crossed0, rear0, knee0 = stance_violation(env, components=True)
        print("STANCE_COMPONENTS at reset crossed=%s rear_overextended=%s low_knee=%s"
              % (crossed0.tolist(), rear0.tolist(), knee0.tolist()), flush=True)
        stance_rec = {
            "used_as_termination": False,
            "reason": "reports crossed on the nominal stand pose for this asset; components recorded per frame instead",
            "components_at_reset": {"crossed": crossed0.tolist(), "rear_overextended": rear0.tolist(),
                                    "low_knee": knee0.tolist()},
            "wheel_body_pos_w": {w: [round(float(v), 5) for v in bp0[i]] for i, w in enumerate(WHEELS)},
            "heading_rad": round(heading0, 5),
        }

        masses = robot.root_physx_view.get_masses()
        m_tot = float(torch.as_tensor(masses).sum())
        weight_n = m_tot * 9.81
        print("MASS total_kg=%.4f weight_n=%.3f" % (m_tot, weight_n), flush=True)
        print("ENV dt=%.4f physics_dt=%.5f decimation=%d num_envs=%d seed=%d"
              % (dt, float(env.physics_dt), int(env.cfg.decimation), env.num_envs, args.seed), flush=True)

        # ---------------- dispatch + substep recorders ---------------------------------
        wheel_term = env.action_manager.get_term("wheel_vel")
        print("ACTION_TERM class=%s scale=%r delay_max_steps=%r lag=%s joint_ids=%s"
              % (type(wheel_term).__name__, getattr(wheel_term.cfg, "scale", None),
                 getattr(wheel_term.cfg, "as2w_action_delay_max_steps", "MISSING"),
                 getattr(wheel_term, "_lag", "n/a").tolist() if hasattr(wheel_term, "_lag") else "n/a",
                 getattr(wheel_term, "_joint_ids", "n/a")), flush=True)
        if getattr(wheel_term.cfg, "as2w_action_delay_max_steps", 0) != 0:
            raise RuntimeError("action delay is not zero; the readback contract is not the recorded one")

        state = {"writes": [], "sub": [], "case": None, "ph": None}
        orig_set = robot.set_joint_velocity_target_index

        def patched_set(target, joint_ids=None, **kw):
            t = target if isinstance(target, torch.Tensor) else torch.as_tensor(target)
            state["writes"].append({
                "case": state["case"], "ph": state["ph"],
                "vals": [round(float(x), 6) for x in t[0].flatten().tolist()],
            })
            return orig_set(target=target, joint_ids=joint_ids, **kw)

        robot.set_joint_velocity_target_index = patched_set
        orig_update = env.scene.update

        def patched_update(*a, **kw):
            step_dt = kw.get("dt", a[0] if a else None)
            if step_dt is None or abs(float(step_dt) - float(env.physics_dt)) > 1e-9:
                raise RuntimeError("scene.update called with dt=%r, not the physics dt" % (step_dt,))
            orig_update(*a, **kw)
            state["sub"].append(snapshot())

        env.scene.update = patched_update

        view = robot.root_physx_view
        have_projected = hasattr(view, "get_dof_projected_joint_forces")

        def loads():
            f = t_(cs.data.net_forces_w)[0][cidx]
            return [round(float(v), 3) for v in f[:, 2].clamp_min(0.0).tolist()]

        def snapshot():
            tgt = [round(float(x), 6) for x in t_(robot.data.joint_vel_target)[0][widx].tolist()]
            dq = [round(float(x), 6) for x in t_(robot.data.joint_vel)[0][widx].tolist()]
            tau = None
            if have_projected:
                try:
                    p = view.get_dof_projected_joint_forces()
                    tau = [round(float(x), 4) for x in t_(p)[0][widx].tolist()]
                except Exception:
                    tau = None
            at = getattr(robot.data, "applied_torque", None)
            applied = None
            if at is not None:
                applied = [round(float(x), 4) for x in t_(at)[0][widx].tolist()]
            vxy = t_(robot.data.root_lin_vel_b)[0]
            g = t_(robot.data.projected_gravity_b)[0]
            return {
                "tgt": tgt, "dq": dq, "load": loads(),
                "proj_tau": tau, "applied_torque": applied,
                "vx": round(float(vxy[0]), 6), "vy": round(float(vxy[1]), 6),
                "wz": round(float(t_(robot.data.root_ang_vel_b)[0][2]), 6),
                "up": round(float(-g[2]), 6),
            }

        def action_for(tgt):
            a = torch.zeros((env.num_envs, 16), device=env.device)
            for i in range(4):
                a[0, 12 + i] = float(tgt[i]) / float(args.scale)
            return a

        def force_state(offsets, roll, z):
            ids = torch.tensor([0], device=env.device, dtype=torch.long)
            q = t_(robot.data.default_joint_pos).clone()
            for i, leg in enumerate(lidx):
                q[0, leg] += float(offsets[i])
            robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(q), env_ids=ids)
            rp = t_(robot.data.root_pose_w).clone()
            rp[0, 0], rp[0, 1], rp[0, 2] = 0.0, 0.0, float(z)
            rp[0, 3], rp[0, 4] = float(np.cos(roll / 2.0)), float(np.sin(roll / 2.0))
            rp[0, 5], rp[0, 6] = 0.0, 0.0
            robot.write_root_pose_to_sim_index(root_pose=rp, env_ids=ids)
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((env.num_envs, 6), device=env.device), env_ids=ids)

        rep = {
            "report": "w1w2_roll_isaac_loaded_v1", "sim": "isaac", "mode": args.mode,
            "label": args.label, "kd": float(args.kd), "scale": float(args.scale),
            "torque_budget_nm": budget, "seed": args.seed, "num_envs": int(env.num_envs),
            "step_dt_s": dt, "physics_dt_s": float(env.physics_dt),
            "decimation": int(env.cfg.decimation), "gravity": list(env.cfg.sim.gravity),
            "usd": str(args.usd), "urdf": str(args.urdf),
            "wheel_r_m": WHEEL_R_M, "load_z_threshold_n": LOAD_Z_N,
            "total_mass_kg": round(m_tot, 4), "weight_n": round(weight_n, 3),
            "resolved_actuators": eff, "wheel_joint_index": dict(zip(WHEELS, widx)),
            "wheel_body_index": dict(zip(WHEELS, bidx)), "wheel_contact_index": dict(zip(WHEELS, cidx)),
            "wheel_effort_ceiling_nm": effort,
            "gyro": "OFF (AS2W_GYRO_KP=0 via env)", "action_term": type(wheel_term).__name__,
            "action_delay_max_steps": int(getattr(wheel_term.cfg, "as2w_action_delay_max_steps", -1)),
            "projected_force_available": bool(have_projected),
            "stance_guard": stance_rec,
        }
        print("WHEEL_TERM_CHECK %s" % json.dumps({"class": type(wheel_term).__name__,
                                                "lag": getattr(wheel_term, "_lag", None).tolist()
                                                if hasattr(wheel_term, "_lag") else None}), flush=True)

        # =========================== CALIBRATE ================================
        if args.mode == "calibrate":
            onehot = {}
            ramp_term = {}
            base_z = 2.0
            RAMP_S = 2.6
            for i, w in enumerate(WHEELS):
                onehot[w] = {}
                for sign in (+1, -1):
                    env.reset()
                    force_state([0.0] * 12, 0.0, base_z)
                    # The 2 s startup ramp lerps the wheel action from the hold action, so a readback
                    # taken inside it is NOT the request.  In this airborne pass the ramp is skipped
                    # explicitly by ageing the episode past it (recorded below); the grounded pass
                    # instead waits it out and records ramp_alpha per frame.
                    env.episode_length_buf[:] = int(round(2.5 / dt))
                    ramp_ok = False
                    for _r in range(int(round(RAMP_S / dt))):
                        out = env.step(action_for([0.0] * 4))
                        ep_now = float(env.episode_length_buf[0]) * dt
                        ramp_now = float(np.clip(ep_now / 2.0, 0.0, 1.0))
                        ramp_ok = ramp_now >= 1.0
                        if _r % 25 == 0 or _r == int(round(RAMP_S / dt)) - 1:
                            fired = {}
                            for nm in env.termination_manager.active_terms:
                                try:
                                    fired[nm] = bool(env.termination_manager.get_term(nm)[0])
                                except Exception:
                                    fired[nm] = "n/a"
                            print("RAMP_DEBUG r=%d ep=%.3f ramp=%.3f term=%s fired=%s"
                                  % (_r, ep_now, ramp_now,
                                     bool(torch.as_tensor(out[2]).any()), json.dumps(fired)), flush=True)
                        if bool(torch.as_tensor(out[2]).any()) and w not in ramp_term:
                            fired = {}
                            for nm in env.termination_manager.active_terms:
                                try:
                                    fired[nm] = bool(env.termination_manager.get_term(nm)[0])
                                except Exception:
                                    fired[nm] = "n/a"
                            ramp_term[w] = {"r": _r, "fired": fired}
                            print("RAMP_TERMINATED r=%d fired=%s" % (_r, json.dumps(fired)), flush=True)
                    if not ramp_ok:
                        raise RuntimeError("startup ramp did not finish before the one-hot trial")
                    state["case"], state["ph"] = "airborne_%s_%+d" % (w, sign), "hold"
                    state["writes"].clear()
                    state["sub"].clear()
                    tgt = [0.0] * 4
                    tgt[i] = sign * args.onehot_rad_s
                    counts = []
                    all_writes = []
                    for _ in range(args.onehot_frames):
                        state["writes"].clear()
                        state["sub"].clear()
                        env.step(action_for(tgt))
                        counts.append(len(state["writes"]))
                        all_writes.extend(wv["vals"] for wv in state["writes"])
                    subs = state["sub"]
                    dispatched = [s["tgt"] for s in subs]
                    dev = max(abs(a - b) for s in dispatched for a, b in zip(s, tgt)) if dispatched else 1e9
                    dq = subs[-1]["dq"] if subs else [0.0] * 4
                    writes_identical = (max(max(abs(a - b) for a, b in zip(wv, all_writes[0])) for wv in all_writes)
                                        if all_writes else 1e9)
                    ramp_final = float(np.clip(float(env.episode_length_buf[0]) * dt / 2.0, 0.0, 1.0))
                    onehot[w]["%+d" % sign] = {
                        "requested_rad_s": tgt,
                        "dispatched_first_substep": dispatched[0] if dispatched else None,
                        "dispatched_last_substep": dispatched[-1] if dispatched else None,
                        "max_abs_dev_from_request": round(dev, 6),
                        "ramp_alpha_at_readback": round(ramp_final, 4),
                        "writes_per_policy_step": counts,
                        "writes_identical_within_step_dev": round(writes_identical, 7),
                        "dq_at_end": dq,
                        "dq_sign_matches_request": bool(dq[i] * sign > 0),
                        "other_wheels_moved_absmax": round(max(abs(dq[j]) for j in range(4) if j != i), 6),
                        "loads_at_end": subs[-1]["load"] if subs else None,
                        "up_at_end": subs[-1]["up"] if subs else None,
                    }
                    print("AIRBORNE %s %+d dispatched=%s dev=%s ramp=%.3f dq=%s loads=%s writes_per_step=%s"
                          % (w, sign, dispatched[-1] if dispatched else None, dev, ramp_final,
                             [round(x, 4) for x in dq], subs[-1]["load"] if subs else None, counts),
                          flush=True)
            ok_channel = all(v["max_abs_dev_from_request"] <= 2e-5 for w in onehot for v in onehot[w].values())
            ok_sign = all(v["dq_sign_matches_request"] for w in onehot for v in onehot[w].values())
            ok_iso = all(v["other_wheels_moved_absmax"] < 0.05 for w in onehot for v in onehot[w].values())
            airborne_load_max = max(max(abs(x) for x in v["loads_at_end"]) for w in onehot for v in onehot[w].values())
            ok_no_contact = airborne_load_max < 1e-3
            ok_writes = all(all(c == int(env.cfg.decimation) for c in v["writes_per_policy_step"])
                            for w in onehot for v in onehot[w].values())
            ok_writes_id = all(v["writes_identical_within_step_dev"] <= 1e-6 for w in onehot
                               for v in onehot[w].values())
            ok_ramp = all(v["ramp_alpha_at_readback"] >= 1.0 for w in onehot for v in onehot[w].values())
            rep["onehot"] = onehot
            rep["ramp_terminations_observed"] = ramp_term
            rep["ramp_handling"] = ("airborne pass: episode aged past the 2 s startup ramp explicitly "
                                    "(episode_length_buf set to 2.5 s) and that is recorded here; "
                                    "grounded pass: ramp waited out, ramp_alpha recorded per frame")
            rep["checks"] = {
                "channel_mapping_exact": ok_channel,
                "sign_convention_consistent": ok_sign,
                "one_hot_isolated_to_its_wheel": ok_iso,
                "no_contact_while_airborne": ok_no_contact,
                "startup_ramp_finished_before_readback": ok_ramp,
                "dispatches_per_policy_step_equals_decimation": ok_writes,
                "dispatches_identical_within_policy_step": ok_writes_id,
                "airborne_max_abs_load_n": round(airborne_load_max, 6),
            }
            bp = t_(robot.data.body_pos_w)[0][bidx].cpu().numpy()
            track = float(abs(bp[0, 1] - bp[1, 1]))
            wheelbase = float(abs(bp[0, 0] - bp[2, 0]))
            rep["geometry"] = {"track_width_m": round(track, 6), "wheelbase_m": round(wheelbase, 6)}
            print("GEOMETRY track_width_m=%.6f wheelbase_m=%.6f" % (track, wheelbase), flush=True)

            if args.write_seq is not None:
                seq = {
                    "schema": "w1w2_roll_seq_v1",
                    "generated_by": "w1w2_roll_isaac_loaded.py --mode calibrate",
                    "order": list(WHEELS),
                    "units": "wheel angular velocity target, rad/s, positive = rolling forward",
                    "step_dt_s": dt, "wheel_r_m": WHEEL_R_M,
                    "track_width_m": round(track, 6), "wheelbase_m": round(wheelbase, 6),
                    "note": "one physical-unit sequence consumed by BOTH simulators, verbatim",
                    "cases": [],
                }
                v_cm_hi = 0.25 / WHEEL_R_M
                v_cm_lo = 0.05 / WHEEL_R_M
                a10 = 0.10 * track * 0.5 / WHEEL_R_M
                a20 = 0.20 * track * 0.5 / WHEEL_R_M

                def case(name, drive, note):
                    seq["cases"].append({
                        "name": name, "note": note, "settle_s": 3.5,
                        "phases": [
                            {"ph": "baseline", "s": 1.5, "tgt": [0.0, 0.0, 0.0, 0.0]},
                            {"ph": "drive", "s": 3.0, "tgt": [round(x, 6) for x in drive]},
                            {"ph": "brake", "s": 2.5, "tgt": [0.0, 0.0, 0.0, 0.0]},
                        ]})
                case("cm_fwd_025", [v_cm_hi] * 4, "common mode forward, 0.25 m/s tier")
                case("cm_fwd_005", [v_cm_lo] * 4, "common mode forward, 0.05 m/s tier")
                case("cm_rev_025", [-v_cm_hi] * 4, "common mode reverse, 0.25 m/s tier")
                case("diff_L_010", [-a10, a10, -a10, a10],
                     "mirror differential, +0.10 rad/s yaw equivalent (right side forward)")
                case("diff_R_010", [a10, -a10, a10, -a10],
                     "mirror differential, -0.10 rad/s yaw equivalent (left side forward)")
                case("diff_L_020", [-a20, a20, -a20, a20],
                     "mirror differential, +0.20 rad/s yaw equivalent, saturation reference")
                case("diff_R_020", [a20, -a20, a20, -a20],
                     "mirror differential, -0.20 rad/s yaw equivalent, saturation reference")
                seq["differential_amplitude_rad_s"] = {"yaw_0p10": round(a10, 6), "yaw_0p20": round(a20, 6)}
                seq["implied_body_wz_rad_s"] = {"diff_L_010": round(2 * a10 * WHEEL_R_M / track, 6),
                                               "diff_R_010": round(-2 * a10 * WHEEL_R_M / track, 6),
                                               "diff_L_020": round(2 * a20 * WHEEL_R_M / track, 6),
                                               "diff_R_020": round(-2 * a20 * WHEEL_R_M / track, 6)}
                seq["implied_body_vx_m_s"] = {"cm_fwd_025": round(v_cm_hi * WHEEL_R_M, 6),
                                             "cm_fwd_005": round(v_cm_lo * WHEEL_R_M, 6)}
                args.write_seq.parent.mkdir(parents=True, exist_ok=True)
                args.write_seq.write_text(json.dumps(seq, indent=1) + "\n")
                print("SEQ_WRITTEN %s" % args.write_seq, flush=True)

            rep["verdict"] = "PASS" if all([ok_channel, ok_sign, ok_iso, ok_no_contact,
                                            ok_writes, ok_writes_id, ok_ramp]) else "FAIL"
            print("CALIBRATE_CHECKS %s" % json.dumps(rep["checks"]), flush=True)
            print("CALIBRATE_VERDICT %s" % rep["verdict"], flush=True)
            (args.out / "calibration.json").write_text(json.dumps(rep, indent=1))
            print("CALIBRATE_DONE", flush=True)
            sys.stdout.flush()
            os._exit(0 if rep["verdict"] == "PASS" else 2)

        # =========================== GROUNDED ================================
        seq = json.loads(Path(args.seq).read_text())
        if seq.get("schema") != "w1w2_roll_seq_v1":
            raise RuntimeError("unexpected sequence schema %r" % seq.get("schema"))
        if tuple(seq["order"]) != WHEELS:
            raise RuntimeError("sequence order %s != %s" % (seq["order"], WHEELS))
        rep["sequence"] = {"path": str(args.seq), "schema": seq["schema"],
                          "cases": [c["name"] for c in seq["cases"]]}
        trace_path = args.out / ("trace_%s_kd%s.jsonl" % (args.label, ("%.2f" % args.kd).replace(".", "p")))
        fh = trace_path.open("w")
        summary = {}

        STATS = {}
        for st in [s.strip() for s in args.states.split(",") if s.strip()]:
            STATS[st] = {"offsets": [0.0] * 12, "roll": 0.0, "z": 0.37, "settle_s": 4.0}
            if st == "B":
                # a genuinely different standing initial state: deeper crouch on all four legs,
                # asymmetric fore/aft bias, small roll
                off = [0.0] * 12
                for i in range(12):
                    part = LEGS[i].split("_")[1]
                    if part == "thigh":
                        off[i] = 0.05
                    elif part == "calf":
                        off[i] = -0.10
                    elif part == "hip":
                        off[i] = 0.01 if i < 4 else -0.01
                STATS["B"] = {"offsets": off, "roll": 0.02, "z": 0.385, "settle_s": 4.5}

        for st in [s.strip() for s in args.states.split(",") if s.strip()]:
            cfg_st = STATS[st]
            for case in seq["cases"]:
                cname = case["name"]
                # ---- independent reset for every condition ----
                env.reset()
                force_state(cfg_st["offsets"], cfg_st["roll"], cfg_st["z"])
                state["case"], state["ph"] = cname, "settle"
                state["writes"].clear()
                state["sub"].clear()
                settle_steps = int(round(cfg_st["settle_s"] / dt))
                settle_tail = []
                for k in range(settle_steps):
                    env.step(action_for([0.0] * 4))
                    if k >= settle_steps - 25:
                        settle_tail.append(state["sub"][-1])
                tail = settle_tail[-25:]
                stand = {
                    "leg_angles_rad": [round(float(x), 4) for x in t_(robot.data.joint_pos)[0][lidx].tolist()],
                    "base_roll_pitch": [round(float(x), 4) for x in t_(robot.data.projected_gravity_b)[0][:2].tolist()],
                    "loads_n": tail[-1]["load"],
                    "vx": round(float(np.mean([s["vx"] for s in tail])), 5),
                    "wz": round(float(np.mean([s["wz"] for s in tail])), 5),
                    "up_min": round(float(min(s["up"] for s in tail)), 5),
                    "dq_absmean": round(float(np.mean([abs(x) for s in tail for x in s["dq"][:4]])), 5),
                }
                settled = (min(stand["loads_n"]) > LOAD_Z_N and stand["up_min"] > 0.95
                           and stand["dq_absmean"] < 0.10)
                print("STAND %s/%s ok=%s loads=%s up=%.4f dq=%.4f"
                      % (st, cname, settled, stand["loads_n"], stand["up_min"], stand["dq_absmean"]), flush=True)

                frames = []
                term_at = None
                for phase in case["phases"]:
                    ph = phase["ph"]
                    state["ph"] = ph
                    tgt = [float(x) for x in phase["tgt"]]
                    n = int(round(float(phase["s"]) / dt))
                    for k in range(n):
                        state["writes"].clear()
                        state["sub"].clear()
                        out = env.step(action_for(tgt))
                        term_any = bool(torch.as_tensor(out[2]).any()) if len(out) > 2 else False
                        subs = state["sub"]
                        writes = state["writes"]
                        if not subs:
                            raise RuntimeError("no physics substep recorded for %s/%s" % (cname, ph))
                        if len(subs) != int(env.cfg.decimation):
                            raise RuntimeError("%d physics substeps recorded per policy step, decimation is %d"
                                               % (len(subs), int(env.cfg.decimation)))
                        tgt_subs = [s["tgt"] for s in subs]
                        dev = max(abs(a - b) for s in tgt_subs for a, b in zip(s, tgt))
                        sub_consistent = max(
                            abs(a - b) for s in tgt_subs for a, b in zip(s, tgt_subs[0])) if len(tgt_subs) > 1 else 0.0
                        write_vals = [w["vals"] for w in writes]
                        write_dev = max((abs(a - b) for v in write_vals for a, b in zip(v, tgt)), default=0.0)
                        ep = float(env.episode_length_buf[0]) * dt
                        ramp = float(np.clip(ep / 2.0, 0.0, 1.0))
                        sw_c, sw_r, sw_k = stance_violation(env, components=True)
                        rec = {
                            "state": st, "case": cname, "ph": ph, "k": k, "t": round(k * dt, 4),
                            "tgt": tgt, "tgt_first_substep": tgt_subs[0], "tgt_dev": round(dev, 7),
                            "substep_target_dev": round(sub_consistent, 7), "n_substeps": len(subs),
                            "n_writes": len(write_vals), "write_dev": round(write_dev, 7),
                            "stance": {"crossed": int(bool(sw_c[0])), "rear": int(bool(sw_r[0])),
                                       "low_knee": int(bool(sw_k[0]))},
                            "dq": subs[-1]["dq"], "load": subs[-1]["load"],
                            "proj_tau": subs[-1]["proj_tau"], "applied_torque": subs[-1]["applied_torque"],
                            "tau_req": [round(args.kd * (tgt[i] - subs[-1]["dq"][i]), 4) for i in range(4)],
                            "vx": subs[-1]["vx"], "vy": subs[-1]["vy"], "wz": subs[-1]["wz"],
                            "up": subs[-1]["up"], "term": int(term_any), "ramp_alpha": round(ramp, 4),
                            "ramp_attenuated": bool(dev > 2e-5),
                        }
                        frames.append(rec)
                        if term_any:
                            term_at = {"case": cname, "state": st, "ph": ph, "k": k,
                                       "frame": rec}
                            print("TERMINATED %s/%s at %s k=%d" % (st, cname, ph, k), flush=True)
                            break
                        if dev > 2e-5 and rec["ramp_alpha"] >= 1.0:
                            raise RuntimeError("dispatched target %.6g != requested %s at %s/%s k=%d"
                                               % (dev, tgt, cname, ph, k))
                        if sub_consistent > 1e-6:
                            raise RuntimeError("target rewritten inside a policy step at %s/%s k=%d (%.3g)"
                                               % (cname, ph, k, sub_consistent))
                        if len(write_vals) != int(env.cfg.decimation):
                            raise RuntimeError("%d dispatch writes in one policy step (decimation=%d) at %s/%s k=%d"
                                               % (len(write_vals), int(env.cfg.decimation), cname, ph, k))
                        if max(max(abs(a - b) for a, b in zip(v, write_vals[0])) for v in write_vals) > 1e-6:
                            raise RuntimeError("dispatches disagree inside one policy step at %s/%s k=%d"
                                               % (cname, ph, k))
                    if term_at:
                        break
                for r in frames:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()

                # ---- per-case metrics -------------------------------------------------
                def seg(ph):
                    return [f for f in frames if f["ph"] == ph]

                def stats(rows):
                    if not rows:
                        return {"n": 0}
                    dq = np.array([r["dq"] for r in rows], float)
                    tgt = np.array([r["tgt"] for r in rows], float)
                    load = np.array([r["load"] for r in rows], float)
                    req = np.array([r["tau_req"] for r in rows], float)
                    pt = [r["proj_tau"] for r in rows if r["proj_tau"] is not None]
                    at = [r["applied_torque"] for r in rows if r["applied_torque"] is not None]
                    diff_t = float(((tgt[:, 1] + tgt[:, 3] - tgt[:, 0] - tgt[:, 2]) / 2).mean())
                    diff_m = float(((dq[:, 1] + dq[:, 3] - dq[:, 0] - dq[:, 2]) / 2).mean())
                    diag_m = float(((dq[:, 0] + dq[:, 3] - dq[:, 1] - dq[:, 2]) / 2).mean())
                    unload = load < LOAD_Z_N
                    longest = np.zeros(4, int)
                    for w in range(4):
                        run = 0
                        for flag in unload[:, w]:
                            run = run + 1 if flag else 0
                            longest[w] = max(longest[w], run)
                    return {
                        "n": len(rows),
                        "tgt_mean": [round(float(x), 5) for x in tgt.mean(axis=0)],
                        "dq_mean": [round(float(x), 5) for x in dq.mean(axis=0)],
                        "tgt_absmean": round(float(np.abs(tgt).mean()), 5),
                        "dq_absmean": round(float(np.abs(dq).mean()), 5),
                        "delivery_ratio": (round(float(np.abs(dq).mean() / np.abs(tgt).mean()), 4)
                                           if np.abs(tgt).mean() > 1e-6 else None),
                        "abs_error_rad_s": round(float(np.abs(dq - tgt).mean()), 5),
                        "diff_tgt": round(diff_t, 5), "diff_meas": round(diff_m, 5),
                        "diag_meas": round(diag_m, 5),
                        "vx_mean": round(float(np.mean([r["vx"] for r in rows])), 5),
                        "vy_mean": round(float(np.mean([r["vy"] for r in rows])), 5),
                        "wz_mean": round(float(np.mean([r["wz"] for r in rows])), 5),
                        "up_min": round(float(min(r["up"] for r in rows)), 5),
                        "loads_mean": [round(float(x), 2) for x in load.mean(axis=0)],
                        "load_sum_over_weight": round(float(load.sum(axis=1).mean() / weight_n), 4),
                        "unload_samples_per_wheel": [int(x) for x in unload.sum(axis=0)],
                        "longest_unload_samples_per_wheel": [int(x) for x in longest],
                        "min_loaded_wheels": int(min(int((~unload[i]).sum()) for i in range(len(rows)))),
                        "support_frac": round(float((~unload).all(axis=1).mean()), 4),
                        "tau_req_absmax": round(float(np.abs(req).max()), 4),
                        "tau_req_over_budget": round(float(np.abs(req).max() / budget), 4),
                        "proj_tau_absmax_ESTIMATED": (round(float(np.abs(np.array(pt)).max()), 4) if pt else None),
                        "applied_torque_absmax_ESTIMATED": (round(float(np.abs(np.array(at)).max()), 4) if at else None),
                        "ramp_alpha_min": round(float(min(r["ramp_alpha"] for r in rows)), 4),
                        "ramp_alpha_at_start": rows[0]["ramp_alpha"],
                    }

                drive = stats([f for f in seg("drive")][-int(round(1.0 / dt)):])
                brake = stats(seg("brake"))
                base = stats(seg("baseline"))
                br = seg("brake")
                stop_k = next((i for i, f in enumerate(br)
                               if max(abs(x) for x in f["dq"]) < 0.05), None)
                entry = max(abs(x) for x in br[0]["dq"]) if br else None
                case_sum = {
                    "state": st, "case": cname, "note": case.get("note"),
                    "settled": bool(settled), "stand": stand,
                    "baseline": base, "drive_tail_1s": drive, "brake": brake,
                    "brake_stop_time_s": (round(stop_k * dt, 3) if stop_k is not None else None),
                    "brake_entry_dq_absmax": entry,
                    "brake_wheel_travel_rad": (
                        round(float(np.abs(np.array([f["dq"] for f in br], float)).sum(axis=0).mean() * dt), 4)
                        if br else None),
                    "terminated": bool(term_at), "termination": term_at,
                    "drive_ramp_ok": bool(drive.get("ramp_alpha_at_start", 0) >= 1.0),
                    "gate_ok": bool(settled and drive.get("n", 0) > 0 and not term_at
                                    and abs(base.get("dq_absmean", 9)) < 0.05
                                    and drive.get("ramp_alpha_min", 0) >= 1.0),
                }
                summary["%s/%s" % (st, cname)] = case_sum
                print("CASE %s/%s ok=%s drive dq=%s tgt=%s diff_t=%+.4f diff_m=%+.4f vx=%+.4f wz=%+.4f "
                      "sup=%.3f unload=%s tau_req_max=%.2f brake_stop=%s"
                      % (st, cname, case_sum["gate_ok"], drive.get("dq_mean"), drive.get("tgt_mean"),
                         drive.get("diff_tgt", 0), drive.get("diff_meas", 0), drive.get("vx_mean", 0),
                         drive.get("wz_mean", 0), drive.get("support_frac", 0),
                         drive.get("longest_unload_samples_per_wheel"), drive.get("tau_req_absmax", 0),
                         case_sum["brake_stop_time_s"]), flush=True)

        fh.close()
        rep["cases"] = summary
        rep["verdict"] = ("PASS" if summary and all(v["gate_ok"] for v in summary.values())
                          else "FAIL")
        (args.out / ("summary_%s_kd%s.json" % (args.label, ("%.2f" % args.kd).replace(".", "p")))
         ).write_text(json.dumps(rep, indent=1))
        print("TRACE %s" % trace_path, flush=True)
        print("GROUNDED_VERDICT %s" % rep["verdict"], flush=True)
        print("GROUNDED_DONE", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(3)
