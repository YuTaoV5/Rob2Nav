#!/usr/bin/env python3
"""Q4: what does the posture bias actually improve, side by side with and without it?

The operator asked for the bias-on / bias-off comparison to give, in parallel:

  per-axis joint error            which joint is worst and by how much
  wheel-centre contact error      fore/aft and lateral, because a joint error is not a footfall error
  settle drift                    how far the body creeps after it has come to rest
  worst safety margin             minimum knee clearance and peak torque fraction of the 90% limits

and to use 0.10 rad as a SIMULATION SCREENING LINE, reporting the worst axis and how long each arm spends
above that line - not to treat 0.10 rad as the formal tolerance, which the operator deferred until the Isaac
same-protocol result exists.

Two arms from the same turn, same settled state: bias off (legs commanded to the default stance) and bias on
(iterated 2*DEF - q_eq compensation).  CPU-only.
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
LINE_RAD = 0.10
KNEE_BODIES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")


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
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=4.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--bias-rounds", type=int, default=4)
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
    kb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in KNEE_BODIES]
    root_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if root_b < 0:
        root_b = 1
    ground_gid = -1
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "ground" or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground_gid = g; break
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))
    cmd = {"v": np.zeros(3)}
    tau_peak = np.zeros(16)

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

    def wheel_pos_body():
        Rb = quat_to_mat(d.qpos[3:7]); root = d.xpos[root_b]
        return np.array([Rb.T @ (d.xpos[b] - root) for b in wb])

    def clear_min():
        best = None
        for b in kb:
            p = d.xpos[b].copy()
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, -1.0]), None, 1, ground_gid, gid)
            if dist >= 0:
                best = dist if best is None else min(best, dist)
        return best

    def hold(n_frames, leg_bias, record=False):
        tau_peak[:] = 0.0
        clr, above, xy = [], 0, []
        x0, y0 = float(d.qpos[0]), float(d.qpos[1])
        for _ in range(n_frames):
            step_ctrl(np.zeros(16), leg_bias=leg_bias)
            if record:
                e = float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())
                if e > LINE_RAD:
                    above += 1
                c = clear_min()
                if c is not None:
                    clr.append(c)
                xy.append([float(d.qpos[0]), float(d.qpos[1])])
        return {"clr": clr, "above": above, "xy": xy, "x0": x0, "y0": y0}

    def settle_run():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)
        prev = np.zeros(16); zr = [None]; ep = 0
        cmd["v"] = np.zeros(3)
        for _ in range(int(a.settle_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        ref = wheel_pos_body()
        cmd["v"] = np.array([a.vx, 0.0, a.wz])
        for _ in range(int(a.turn_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        cmd["v"] = np.zeros(3)
        for _ in range(int(a.release_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        return ref

    def summarize(st, ref, bias, n_frames):
        q_err = np.abs(d.qpos[qadr][:12] - DEF[:12])
        worst_i = int(q_err.argmax())
        pos = wheel_pos_body()
        dpos = pos - ref
        clr = st["clr"] if st["clr"] else [None]
        xy = st["xy"]
        drift = float(np.max([np.hypot(p[0] - st["x0"], p[1] - st["y0"]) for p in xy])) if xy else 0.0
        return {"bias_absmax": float(np.abs(bias).max()) if bias is not None else 0.0,
                "bias_vector_rad": ([] if bias is None else [round(float(x), 6) for x in bias]),
                "q_err_max": float(q_err.max()),
                "worst_axis": JOINTS[worst_i],
                "per_axis_q_err": {JOINTS[i]: round(float(q_err[i]), 4) for i in range(12)},
                "wheel_dpos_fwd_m": [round(float(dpos[i][0]), 4) for i in range(4)],
                "wheel_dpos_lat_m": [round(float(dpos[i][1]), 4) for i in range(4)],
                "worst_wheel_dpos_m": float(np.abs(dpos[:, :2]).max()),
                "settle_drift_m": drift,
                "clearance_min_m": (None if clr[0] is None else round(float(min(clr)), 4)),
                "torque_frac_90pct": {"hip": float(tau_peak[:4].max() / (0.9 * 54)),
                                       "thigh": float(tau_peak[4:8].max() / (0.9 * 54)),
                                       "calf": float(tau_peak[8:12].max() / (0.9 * 81)),
                                       "wheel": float(tau_peak[12:].max() / (0.9 * 13.5))},
                "frames_above_line": int(st["above"]),
                "seconds_above_line": round(st["above"] * STEP_DT, 3),
                "frames_total": int(n_frames)}

    n = int(a.hold_s / STEP_DT)
    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "calibration_protocol": {"method": "iterated 2*DEF - q_eq", "rounds": a.bias_rounds,
                                     "hold_per_round_s": 1.0, "residual_avg_window_s": 0.5,
                                     "gain": 1.0, "leg_action_scale": LEG_SCALE},
           "line_rad": LINE_RAD,
           "line_status": "SIMULATION SCREENING LINE only - the formal tolerance waits for the Isaac same-protocol result",
           "arms": {}}
    ref = settle_run()
    st = hold(n, None, record=True)
    rep["arms"]["bias_off"] = summarize(st, ref, None, n)

    # calibrate the bias by iteration from this same settled state
    bias = np.zeros(12)
    for _ in range(a.bias_rounds):
        hold(int(1.0 / STEP_DT), bias, record=False)
        acc, k = np.zeros(12), 0
        for _ in range(int(0.5 / STEP_DT)):
            step_ctrl(np.zeros(16), leg_bias=bias)
            acc += d.qpos[qadr][:12] - DEF[:12]; k += 1
        bias = bias - acc / max(1, k)
    st = hold(n, bias, record=True)
    rep["arms"]["bias_on"] = summarize(st, ref, bias, n)

    for name in ("bias_off", "bias_on"):
        r = rep["arms"][name]
        print("%-9s q_err=%.3f worst=%s | wheel_dpos_max=%.4f m | drift=%.4f m | clr=%s | "
              "above %.2f rad: %.2fs/%d | tau/90%% max=%.2f"
              % (name, r["q_err_max"], r["worst_axis"], r["worst_wheel_dpos_m"], r["settle_drift_m"],
                 "n/a" if r["clearance_min_m"] is None else "%.4f" % r["clearance_min_m"],
                 LINE_RAD, r["seconds_above_line"], r["frames_total"],
                 max(r["torque_frac_90pct"].values())))
    (a.out / "bias_compare.json").write_text(json.dumps(rep, indent=1))
    print("BIAS_COMPARE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
