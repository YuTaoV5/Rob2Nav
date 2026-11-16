#!/usr/bin/env python3
"""Instant e-stop: from a driving state, zero the wheels and freeze the legs with NO ramp.

C5 lists instant e-stop as not_run, and the reason matters: what has been tested is a RELEASE, where the
command walks to zero through the 0.12-rate lerp, so the wheels decelerate gradually.  An e-stop is a
different condition - the wheel target jumps to zero in one control step and the legs are frozen where
they are, with no transition to the standing pose.  The sharper transient can excite the very support and
torque limits the goal cares about, so it needs its own case.

Measured from the instant of the stop command:
  stop distance and time to |vx| < 0.02
  peak body deceleration and peak pitch/roll excursion
  peak |tau| per joint class against the 90% limits (hip/thigh 54, calf 81, wheel 13.5 N*m)
  four-wheel support fraction, upright minimum, non-wheel ground contacts
  whether it settles without falling

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
KP_LEG = 55.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
LOAD_N = 20.0
WHEEL_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
STOP_VX = 0.02
UPRIGHT_MIN = 0.85
SUPPORT_MIN = 0.95


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
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--drive-s", type=float, default=4.0)
    ap.add_argument("--hold-s", type=float, default=5.0)
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
    wb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in WHEEL_BODIES]
    wheel_set = set(wb)
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

    def step_ctrl(act16):
        nonlocal tau_peak
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        tgt = np.r_[act16[:12] * LEG_SCALE + DEF[:12], act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            tau = np.clip(tau, -LIM, LIM)
            tau_peak = np.maximum(tau_peak, np.abs(tau))
            d.ctrl[aid] = tau
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

    def nonwheel():
        n = 0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 == ground_gid or c.geom2 == ground_gid:
                other = m.geom_bodyid[c.geom2] if c.geom1 == ground_gid else m.geom_bodyid[c.geom1]
                if other not in wheel_set and other != 0:
                    n += 1
        return n

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    prev = np.zeros(16); zr = [None]; ep = 0
    for _ in range(int(a.settle_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        step_ctrl(prev); ep += 1
    cmd["v"] = np.array([a.vx, 0.0, a.wz])
    for _ in range(int(a.drive_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        step_ctrl(prev); ep += 1

    # --- instant e-stop: wheels to zero, legs frozen at the current joint positions, no ramp ---
    vx_at_stop = float(d.qvel[0])
    x_at_stop = float(d.qpos[0])
    frozen_legs = d.qpos[qadr][:12].copy()
    frozen_act = (frozen_legs - DEF[:12]) / LEG_SCALE     # hold legs where they are
    tau_peak[:] = 0.0
    stop_idx = None
    sup, ups, nw, pitch_max, roll_max = [], [], 0, 0.0, 0.0
    n_total = int(a.hold_s / STEP_DT)
    for k in range(n_total):
        step_ctrl(np.r_[frozen_act, np.zeros(4)])
        vx = float(d.qvel[0])
        R = quat_to_mat(d.qpos[3:7])
        up = float(-(R.T @ np.array([0.0, 0.0, -1.0]))[2])
        ups.append(up)
        sup.append(bool((loads() > LOAD_N).all()))
        nw += nonwheel()
        pitch_max = max(pitch_max, abs(float(np.arcsin(np.clip(-R[2, 0], -1, 1)))))
        roll_max = max(roll_max, abs(float(np.arctan2(R[2, 1], R[2, 2]))))
        if stop_idx is None and abs(vx) < STOP_VX:
            stop_idx = k + 1
    x_final = float(d.qpos[0])
    tail = slice(max(0, n_total - 150), n_total)
    rep = {
        "checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
        "estop": {"mode": "instant zero wheel target + legs frozen at current joint positions, no ramp"},
        "at_stop": {"vx": vx_at_stop, "x": x_at_stop},
        "stop_time_s": (stop_idx * STEP_DT) if stop_idx else None,
        "stop_distance_m": abs(x_final - x_at_stop),
        "tail_vx": float(np.mean([d.qvel[0]])) if False else None,
        "upright_min": float(min(ups)),
        "support_frac": float(np.mean(sup)),
        "support_frac_tail": float(np.mean(sup[tail])),
        "nonwheel_ground_contacts": nw,
        "pitch_max_rad": pitch_max, "roll_max_rad": roll_max,
        "fall": bool(min(ups) < UPRIGHT_MIN),
        "torque_frac_of_90pct": {
            "hip": float(tau_peak[:4].max() / (0.9 * 54)),
            "thigh": float(tau_peak[4:8].max() / (0.9 * 54)),
            "calf": float(tau_peak[8:12].max() / (0.9 * 81)),
            "wheel": float(tau_peak[12:].max() / (0.9 * 13.5))},
    }
    rep["pass"] = bool(not rep["fall"] and rep["support_frac_tail"] >= SUPPORT_MIN
                       and rep["nonwheel_ground_contacts"] == 0
                       and max(rep["torque_frac_of_90pct"].values()) <= 1.0)
    (a.out / "estop.json").write_text(json.dumps(rep, indent=1))
    print("vx=%.2f wz=%+.2f | at_stop vx=%.3f | stop_t=%s s stop_d=%.4f m | sup=%.3f (tail %.3f) "
          "up_min=%.3f nonwheel=%d pitch=%.3f roll=%.3f | tau/90%% hip=%.2f thigh=%.2f calf=%.2f wheel=%.2f | pass=%s"
          % (a.vx, a.wz, vx_at_stop,
             "n/a" if rep["stop_time_s"] is None else "%.2f" % rep["stop_time_s"],
             rep["stop_distance_m"], rep["support_frac"], rep["support_frac_tail"],
             rep["upright_min"], nw, pitch_max, roll_max,
             rep["torque_frac_of_90pct"]["hip"], rep["torque_frac_of_90pct"]["thigh"],
             rep["torque_frac_of_90pct"]["calf"], rep["torque_frac_of_90pct"]["wheel"], rep["pass"]))
    print("ESTOP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
