#!/usr/bin/env python3
"""Regression under the posture bias: support, rear clearance, torque margin, non-wheel contact.

The goal requires that any leg-pose change pass standing, four-wheel support and all hard-protection
regressions before it may enter the next tier.  The posture bias was shown to cut the post-turn standing
residual from about 0.198 to 0.061 rad, but none of those regressions had been run, so it was explicitly not
usable as a delivery configuration.  This runs them.

Measured during the biased hold, with the same bias calibration the earlier test used (4 iterations of
"measure residual, add it to the bias"):

  support          fraction of frames with all four wheels loaded above the threshold
  rear clearance   minimum vertical distance from the rear calf / knee region to the ground, by ray cast
  torque           peak |tau| per joint class against the 90% limits (hip/thigh 54, calf 81, wheel 13.5 N*m)
  non-wheel        any contact involving a non-wheel body with the ground
  posture          the leg error against the default stance, to confirm the bias still works here

Two holds are compared: plain fixed stand and biased fixed stand, so the cost of the bias is visible.
CPU-only, so it does not contend with training.
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
KP_LEG, KP_WHEEL = 55.0, 0.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
LOAD_N = 20.0
WHEEL_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--bias-rounds", type=int, default=4)
    ap.add_argument("--clearance-bodies", default="RL_calf,RR_calf,RL_thigh,RR_thigh")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, KP_LEG), np.full(4, KP_WHEEL)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, WHEEL_KD)]
    wb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in WHEEL_BODIES]
    clr_b = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
             for n in a.clearance_bodies.split(",") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
    # ground geom = the plane whose name is 'ground' if present, else the largest plane
    ground_gid = -1
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "ground" or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground_gid = g
            break
    wheel_body_set = set(wb)
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

    tau_peak = np.zeros(16)

    def step_ctrl(act16, leg_bias=None):
        nonlocal tau_peak
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        leg_t = act16[:12] * LEG_SCALE + DEF[:12]
        if leg_bias is not None:
            leg_t = leg_t + leg_bias
        tgt = np.r_[leg_t, act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            tau = np.clip(tau, -LIM, LIM)
            tau_peak = np.maximum(tau_peak, np.abs(tau))
            d.ctrl[aid] = tau
            mujoco.mj_step(m, d)

    def loads():
        # MjContact exposes geoms, not bodies; map through geom_bodyid
        out = []
        for b in wb:
            tot = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                b1 = m.geom_bodyid[c.geom1]
                b2 = m.geom_bodyid[c.geom2]
                if b1 == b or b2 == b:
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv)
                    tot += abs(fv[0])
            out.append(tot)
        return np.array(out)

    def clearance():
        best = 1e9
        for b in clr_b:
            p = d.xpos[b].copy()
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
            if dist >= 0:
                best = min(best, dist)
        return float(best) if best < 1e8 else None

    def nonwheel_contact():
        n = 0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 == ground_gid or c.geom2 == ground_gid:
                other = m.geom_bodyid[c.geom2] if c.geom1 == ground_gid else m.geom_bodyid[c.geom1]
                if other not in wheel_body_set and other != 0:
                    n += 1
        return n

    def run_hold(n_frames, leg_bias, collect):
        stats = {"support": [], "clearance": [], "nonwheel": 0}
        for _ in range(n_frames):
            step_ctrl(np.zeros(16), leg_bias=leg_bias)
            if collect:
                stats["support"].append(bool((loads() > LOAD_N).all()))
                c = clearance()
                if c is not None:
                    stats["clearance"].append(c)
                stats["nonwheel"] += nonwheel_contact()
        return stats

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    prev = np.zeros(16); zr = [None]; ep = 0
    for _ in range(int(a.settle_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        step_ctrl(prev); ep += 1
    cmd["v"] = np.array([0.0, 0.0, a.wz])
    for _ in range(int(a.turn_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        step_ctrl(prev); ep += 1

    n_hold = int(a.hold_s / STEP_DT)
    tau_peak[:] = 0.0
    plain = run_hold(n_hold, None, collect=True)
    plain_err = float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())
    plain_tau = tau_peak.copy()

    # calibrate bias by iteration, then collect regression stats under it
    bias = np.zeros(12)
    for _ in range(a.bias_rounds):
        run_hold(int(1.0 / STEP_DT), bias, collect=False)
        acc, n = np.zeros(12), 0
        for _ in range(int(0.5 / STEP_DT)):
            step_ctrl(np.zeros(16), leg_bias=bias)
            acc += d.qpos[qadr][:12] - DEF[:12]; n += 1
        bias = bias - acc / max(1, n)
    tau_peak[:] = 0.0
    biased = run_hold(n_hold, bias, collect=True)
    biased_err = float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())
    biased_tau = tau_peak.copy()

    def summarize(st, err, tp):
        sup = np.mean(st["support"]) if st["support"] else None
        cl = min(st["clearance"]) if st["clearance"] else None
        hipmax = float(tp[:4].max()); thmax = float(tp[4:8].max())
        camelmax = float(tp[8:12].max()); whmax = float(tp[12:].max())
        return {"support_frac": sup, "clearance_min_m": cl,
                "nonwheel_contact_frames": st["nonwheel"], "leg_err_max_rad": err,
                "torque_peak": {"hip": hipmax, "thigh": thmax, "calf": camelmax, "wheel": whmax},
                "torque_frac_of_90pct": {"hip": hipmax / (0.9 * 54), "thigh": thmax / (0.9 * 54),
                                          "calf": camelmax / (0.9 * 81), "wheel": whmax / (0.9 * 13.5)}}

    rep = {"checkpoint": str(a.ckpt), "wz": a.wz, "load_threshold_n": LOAD_N,
           "clearance_bodies": a.clearance_bodies,
           "bias_absmax": float(np.abs(bias).max()),
           "plain": summarize(plain, plain_err, plain_tau),
           "biased": summarize(biased, biased_err, biased_tau)}
    (a.out / "bias_regression.json").write_text(json.dumps(rep, indent=1))
    for k in ("plain", "biased"):
        v = rep[k]
        print("%-6s sup=%s clr=%s nonwheel=%d leg_err=%.3f tau/90%%: hip=%.2f thigh=%.2f calf=%.2f wheel=%.2f"
              % (k, "n/a" if v["support_frac"] is None else "%.3f" % v["support_frac"],
                 "n/a" if v["clearance_min_m"] is None else "%.4f" % v["clearance_min_m"],
                 v["nonwheel_contact_frames"], v["leg_err_max_rad"],
                 v["torque_frac_of_90pct"]["hip"], v["torque_frac_of_90pct"]["thigh"],
                 v["torque_frac_of_90pct"]["calf"], v["torque_frac_of_90pct"]["wheel"]))
    print("BIAS_REGRESSION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
