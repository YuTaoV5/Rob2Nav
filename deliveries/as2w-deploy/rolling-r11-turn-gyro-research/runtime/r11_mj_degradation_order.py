#!/usr/bin/env python3
"""Order of degradation during a spin: which of wheel load, leg pose, measured differential or yaw goes first?

The operator asked two things that this one trace answers:

  item 4  why does RL reach 0 N during a left spin, before anything may be commanded off the ground
  training  within valid support windows, which quantity degrades FIRST - wheel load, leg pose, measured
            differential or yaw - because a global gain change would trade one against another

Method: run one spin with per-frame records of
    per-wheel ground load, measured wheel speeds and their differential, yaw rate, body roll/pitch,
    per-leg joint error against the fixed stand, knee clearance

then locate the first frame at which any wheel falls below the load threshold and report, for a window
BEFORE that frame, which quantity had already departed from its settled baseline.  That ordering is the
answer: the quantity that departs first is the one to fix, and the ones after it are consequences.

CPU-only.
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
LOAD_N = 20.0
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
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--pre-window-s", type=float, default=1.5,
                    help="how far before the first unload to look for the earliest departure")
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
    ground_gid = -1
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "ground" or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground_gid = g; break
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

    def step_ctrl(act16):
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        tgt = np.r_[act16[:12] * LEG_SCALE + DEF[:12], act16[12:] * WHEEL_SCALE]
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

    cmd["v"] = np.array([a.vx, 0.0, a.wz])
    rec = {k: [] for k in ("load", "dq", "yaw", "roll", "pitch", "legq", "knee")}
    for _ in range(int(a.turn_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        step_ctrl(prev); ep += 1
        R = quat_to_mat(d.qpos[3:7])
        rec["load"].append(loads().tolist())
        rec["dq"].append(d.qvel[vadr][12:].tolist())
        rec["yaw"].append(float(d.qvel[5]))
        rec["roll"].append(float(np.arctan2(R[2, 1], R[2, 2])))
        rec["pitch"].append(float(np.arcsin(np.clip(-R[2, 0], -1, 1))))
        rec["legq"].append((d.qpos[qadr][:12] - DEF[:12]).tolist())
        rec["knee"].append([float(d.xpos[b][2]) for b in kb])
    n = len(rec["yaw"])
    load = np.array(rec["load"]); dq = np.array(rec["dq"]); yaw = np.array(rec["yaw"])
    roll = np.array(rec["roll"]); pitch = np.array(rec["pitch"])
    legq = np.abs(np.array(rec["legq"]))
    knee = np.array(rec["knee"])
    d4 = dq[:, 1] + dq[:, 3] - dq[:, 0] - dq[:, 2]

    unload = (load < LOAD_N).any(axis=1)
    first = int(np.argmax(unload)) if unload.any() else None
    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "load_threshold_n": LOAD_N, "frames": n, "dt": STEP_DT,
           "first_unload_frame": first,
           "first_unload_time_s": (first * STEP_DT) if first is not None else None,
           "first_unload_wheels": ([LEGS[i] for i in range(4) if load[first, i] < LOAD_N] if first is not None else None),
           "unload_fraction": float(unload.mean()),
           "per_wheel_min_load_N": {LEGS[i]: float(load[:, i].min()) for i in range(4)},
           "per_wheel_unload_frames": {LEGS[i]: int((load[:, i] < LOAD_N).sum()) for i in range(4)},
           "yaw_tail": float(yaw[max(0, n - 150):].mean()),
           "ddq_tail": float(d4[max(0, n - 150):].mean())}
    if first is not None:
        b0 = max(0, first - int(a.pre_window_s / STEP_DT))
        base = slice(max(0, b0 - 100), b0)      # baseline just before the pre-window
        pre = slice(b0, first)
        def depart(series, name):
            b = series[base].mean() if len(series[base]) > 0 else series[0]
            seg = series[pre]
            if len(seg) == 0:
                return None
            dev = float(np.max(np.abs(seg - b)))
            scale = float(np.max(np.abs(series))) + 1e-9
            return {"name": name, "baseline": float(b) if np.ndim(b) == 0 else None,
                     "max_abs_dev_in_prewindow": dev, "rel_dev": dev / scale,
                     "first_exceed_frame": None}
        # per-quantity first frame in the pre-window where deviation exceeds 20% of the whole-run range
        def first_exceed(series, frac=0.2):
            rng = float(series.max() - series.min())
            if rng < 1e-9:
                return None
            b = float(series[:max(1, b0)].mean()) if b0 > 0 else float(series[0])
            thr = frac * rng
            idx = np.where(np.abs(series[:first] - b) > thr)[0]
            return int(idx[0]) if len(idx) else None
        metrics = {
            "wheel_load_spread": np.abs(load - load.mean(axis=1, keepdims=True)).max(axis=1),
            "leg_joint_error": legq.max(axis=1),
            "measured_differential": np.abs(d4),
            "yaw_rate": np.abs(yaw),
            "roll": np.abs(roll),
            "pitch": np.abs(pitch),
            "knee_height_min": knee.min(axis=1),
        }
        order = {}
        for name, series in metrics.items():
            order[name] = first_exceed(series)
        valid = {k: v for k, v in order.items() if v is not None}
        ranked = sorted(valid.items(), key=lambda kv: kv[1])
        rep["degradation"] = {
            "baseline_frames": int(b0), "first_unload_frame": int(first),
            "first_exceed_frame_per_quantity": order,
            "ranked_earliest_first": [{"quantity": k, "frame": int(v),
                                        "time_s": round(v * STEP_DT, 3)} for k, v in ranked],
            "note": "frame is relative to the START OF THE SPIN command; compare orderings, not absolute values"}
        print("first unload at frame %d (%.2f s), wheels %s, unload fraction %.3f"
              % (first, first * STEP_DT, rep["first_unload_wheels"], rep["unload_fraction"]))
        for k, v in ranked:
            print("  %-26s first exceeds at frame %5d (%.2f s)" % (k, v, v * STEP_DT))
    else:
        print("no unload detected in %.1f s of spin" % a.turn_s)
    print("per-wheel min load N:", {LEGS[i]: round(float(load[:, i].min()), 1) for i in range(4)})
    print("per-wheel unload frames:", rep["per_wheel_unload_frames"])
    (a.out / "degradation_order.json").write_text(json.dumps(rep, indent=1))
    print("DEGRADATION_ORDER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
