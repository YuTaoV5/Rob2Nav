#!/usr/bin/env python3
"""Three-foot load-bearing confirmation, then lift-in-place with a verified wheel-height curve.

Operator item 4, in their order:
  1. only cases with a large residual after settling continue to a single-leg trial
  2. the unload instant is not suitable for commanding a leg off the ground, so FIRST confirm the
     settled state can bear load on three feet
  3. lift IN PLACE and land IN PLACE using wheel-centre inverse kinematics
  4. check the PLANNED wheel height against the MEASURED wheel height - both must show the wheel leaves
     the ground while the other three stay continuously loaded
  5. only then move to a target foothold
  6. keep this trajectory so the redo can be compared at the same angle

Two things this deliberately does differently from my earlier failed attempts:

  * the target rise is small (default 10 mm), because the validated support margin leaves only
    1-24 mm of headroom once a leg is removed - asking for 30 mm was asking for something the
    geometry does not permit, which is a plausible reason the earlier IK only reached 15%.
  * the leg is chosen by POSITIVE support margin from the validated table, not by largest joint error.
    Picking by joint error is why the earlier runs chose legs that were already outside the triangle.

The output keeps a per-frame wheel-height curve for the lifted leg and the load of all four, so the
"did the wheel actually leave the ground" question is answered by data rather than by the plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mujoco

DT = 0.000625
SUBSTEPS = 32
STEP_DT = DT * SUBSTEPS
LEG_KD, WHEEL_KD = 2.0, 6.0
KP_LEG = 55.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
KNEE = [f"{L}_calf" for L in LEGS]
LOAD_N = 20.0


def build_actor(sd):
    lin = {}
    for k, v in sd.items():
        if not k.startswith("mlp."):
            continue
        _, idx, param = k.split(".")
        if param in ("weight", "bias"):
            lin.setdefault(int(idx), {})[param] = v
    layers, order = [], sorted(lin)
    for n, i in enumerate(order):
        w, b = lin[i]["weight"], lin[i]["bias"]
        mm = nn.Linear(w.shape[1], w.shape[0]); mm.weight.data = w.clone(); mm.bias.data = b.clone()
        layers.append(mm)
        if n != len(order) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def support_margin(com, pts):
    """Signed distance from com to the convex hull of pts; positive inside.  Same rule that was
    self-validated against analytic cases (square, triangle, corner, order independence)."""
    from scipy.spatial import ConvexHull  # noqa: F401  (fallback below if absent)
    return _hull_margin(com, pts)


def _hull_margin(com, pts):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return float("-inf")
    # order the points around their centroid, then walk the edges
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    poly = pts[order]
    best = 1e9
    for i in range(len(poly)):
        a = poly[i]; b = poly[(i + 1) % len(poly)]
        e = b - a
        nrm = np.array([-e[1], e[0]])
        ln = np.hypot(*nrm)
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        d = float(nrm @ (com - a))
        if d < best:
            best = d
    # if the centroid-ordered polygon came out clockwise the sign flips; normalise by the centroid
    if best < 0:
        d0 = 0.0
        for i in range(len(poly)):
            a = poly[i]; b = poly[(i + 1) % len(poly)]
            e = b - a
            nrm = np.array([-e[1], e[0]]); ln = np.hypot(*nrm)
            if ln < 1e-12:
                continue
            nrm = nrm / ln
            d0 = max(d0, float(nrm @ (c - a)))
        if d0 > 0:
            best = -best
    return float(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--rise-m", type=float, default=0.010,
                    help="target wheel rise; the validated margin leaves only 1-24 mm of headroom")
    ap.add_argument("--leg", default="auto", help="auto = pick by positive support margin")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--stand-s", type=float, default=4.0)
    ap.add_argument("--lift-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--land-s", type=float, default=2.0)
    ap.add_argument("--ik-iters", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, KP_LEG), np.full(4, 0.0)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, WHEEL_KD)]
    wb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{L}_foot") for L in LEGS]
    kb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KNEE]
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))
    cmd = {"v": np.zeros(3)}

    def obs_of(prev, ep, zr):
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        R = quat_to_mat(d.qpos[3:7])
        qrel = q - DEF
        qrel[12:] = (qrel[12:] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        o = np.zeros(65)
        o[0:3] = cmd["v"] - np.array([0.4, 0.0, 0.0])
        o[3:6] = R.T @ d.qvel[3:6]
        o[6:9] = R.T @ np.array([0.0, 0.0, -1.0])
        o[9] = 0.4
        o[10:26] = qrel
        o[26:42] = dq
        o[42:58] = prev
        clk = 2 * np.pi * ep * STEP_DT / 0.62
        o[58], o[59] = np.sin(clk), np.cos(clk)
        zero = float(np.all(cmd["v"] == 0.0))
        if zr[0] is None:
            zr[0] = qrel[12:].copy()
        dd = (qrel[12:] - zr[0] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        o[60:64] = np.clip(dd / 0.20, -3, 3) * zero
        o[64] = zero
        return o

    prev = np.zeros(16); zr = [None]; ep = 0
    override = {"leg_target": None}

    def step_ctrl(act16):
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        leg = act16[:12] * LEG_SCALE + DEF[:12]
        if override["leg_target"] is not None:
            leg = override["leg_target"].copy()
        tgt = np.r_[leg, act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            d.ctrl[aid] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)

    def loads():
        out = []
        for b in wb:
            tot = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if m.geom_bodyid[c.geom1] == b or m.geom_bodyid[c.geom2] == b:
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv)
                    tot += abs(fv[0])
            out.append(tot)
        return np.array(out)

    def run_policy(secs, vx, wz):
        nonlocal ep
        cmd["v"] = np.array([vx, 0.0, wz])
        n = int(secs / STEP_DT)
        for _ in range(n):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    run_policy(a.settle_s, 0.0, 0.0)
    run_policy(a.turn_s, a.vx, a.wz)
    run_policy(a.release_s, 0.0, 0.0)
    run_policy(a.stand_s, 0.0, 0.0)

    # ---- settled state: choose the leg by POSITIVE support margin ----
    com = np.zeros(2); tm = 0.0
    for i in range(m.nbody):
        com += m.body_mass[i] * d.xpos[i][:2]; tm += m.body_mass[i]
    com /= tm
    pts = np.array([d.xpos[b][:2] for b in wb])
    margins = {}
    for i, L in enumerate(LEGS):
        keep = [j for j in range(4) if j != i]
        margins[L] = round(support_margin(com, pts[keep]), 4)
    if a.leg == "auto":
        leg = max(margins, key=margins.get)
    else:
        leg = a.leg
    li = LEGS.index(leg)
    settled_loads = loads()
    settle_q = d.qpos[qadr][:12].copy()
    z0 = float(d.xpos[wb[li]][2])
    print("settled margins to 90%%: %s" % margins)
    print("chosen leg %s (margin %+.4f m); settled loads %s; wheel z0=%.5f"
          % (leg, margins[leg], np.round(settled_loads, 1).tolist(), z0), flush=True)

    # ---- IK: raise the wheel by rise_m using only that leg's three joints ----
    idx = slice(li * 3, li * 3 + 3)
    q_start = settle_q[idx].copy()
    target = np.array([d.xpos[wb[li]][0], d.xpos[wb[li]][1], z0 + a.rise_m])
    lam, plan = 0.05, None

    qidx = qadr[li * 3:li * 3 + 3]        # real qpos indices for this leg's three joints

    def set_leg(qv):
        # d.qpos[qadr] returns a COPY, so d.qpos[qadr][idx] = qv would silently write to the copy.
        for k in range(3):
            d.qpos[qidx[k]] = qv[k]

    def get_leg():
        return np.array([d.qpos[qidx[k]] for k in range(3)])

    def fk_try(qv):
        set_leg(qv)
        mujoco.mj_kinematics(m, d)
        return d.xpos[wb[li]].copy()

    cur = q_start.copy()
    best_err = 1e9
    for _ in range(a.ik_iters):
        set_leg(cur)
        mujoco.mj_kinematics(m, d)
        p = d.xpos[wb[li]].copy()
        err = target - p
        e = float(np.linalg.norm(err))
        if e < best_err:
            best_err, plan = e, cur.copy()
        if e < 1e-5:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            qp = cur.copy(); qp[k] += 1e-4
            set_leg(qp)
            mujoco.mj_kinematics(m, d)
            J[:, k] = (d.xpos[wb[li]] - p) / 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        cur = cur + dq
        # respect the 90% angle envelope
        for k in range(3):
            jn = JOINTS[li * 3 + k]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if m.jnt_limited[jid]:
                lo, hi = m.jnt_range[jid]; span = hi - lo
                cur[k] = float(np.clip(cur[k], lo + 0.05 * span, hi - 0.05 * span))
    set_leg(settle_q[idx])
    mujoco.mj_kinematics(m, d)
    planned_rise = float(fk_try(plan)[2] - z0) if plan is not None else float("nan")
    fk_try(settle_q[idx]); mujoco.mj_kinematics(m, d)
    print("IK: residual %.6f m, PLANNED wheel rise %+.5f m (target %+.5f m)" % (best_err, planned_rise, a.rise_m), flush=True)

    curve = []
    def hold_leg(qv, secs, tag):
        nonlocal ep
        override["leg_target"] = settle_q.copy(); override["leg_target"][idx] = qv
        cmd["v"] = np.zeros(3)
        n = int(secs / STEP_DT)
        for _ in range(n):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
            curve.append({"tag": tag,
                           "wheel_z": float(d.xpos[wb[li]][2]),
                           "rise_m": float(d.xpos[wb[li]][2] - z0),
                           "loads": [round(float(x), 1) for x in loads()],
                           "q_err_leg_max": float(np.abs(d.qpos[qadr][idx] - settle_q[idx]).max())})

    hold_leg(settle_q[idx], 0.5, "pre")
    hold_leg(plan if plan is not None else settle_q[idx], a.lift_s, "lift")
    hold_leg(plan if plan is not None else settle_q[idx], a.hold_s, "hold")
    hold_leg(settle_q[idx], a.land_s, "land")
    override["leg_target"] = None

    lift = [c for c in curve if c["tag"] == "lift"]
    held = [c for c in curve if c["tag"] == "hold"]
    meas_rise_max = max((c["rise_m"] for c in lift + held), default=float("nan"))
    others_ok = all(min(c["loads"][j] for j in range(4) if j != li) > LOAD_N for c in lift + held)
    lifted = [c for c in lift + held if c["loads"][li] < LOAD_N]
    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "target_rise_m": a.rise_m, "chosen_leg": leg, "margins": margins,
           "settled_loads_N": [round(float(x), 1) for x in settled_loads],
           "ik_residual_m": float(best_err), "planned_rise_m": planned_rise,
           "measured_rise_max_m": float(meas_rise_max),
           "plan_vs_measured_error_m": float(abs(meas_rise_max - planned_rise)),
           "other_three_continuously_loaded": bool(others_ok),
           "lifted_frames": len(lifted), "total_lift_hold_frames": len(lift) + len(held),
           "wheel_left_ground": bool(len(lifted) > 0.5 * (len(lift) + len(held))),
           "curve": curve}
    verdict = ("PASS" if (others_ok and len(lifted) > 0.5 * (len(lift) + len(held))
                          and meas_rise_max > 0.5 * a.rise_m) else "FAIL")
    rep["verdict"] = verdict
    print("measure: measured rise %+.5f m vs planned %+.5f m (err %.5f)"
          % (meas_rise_max, planned_rise, abs(meas_rise_max - planned_rise)))
    print("other three continuously loaded: %s | lifted frames %d/%d | wheel left ground: %s"
          % (others_ok, len(lifted), len(lift) + len(held), rep["wheel_left_ground"]))
    print("LIFT_IN_PLACE_%s" % verdict)
    (a.out / "lift_in_place.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
