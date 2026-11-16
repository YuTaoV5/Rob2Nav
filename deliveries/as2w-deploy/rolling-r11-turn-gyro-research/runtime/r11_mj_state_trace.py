#!/usr/bin/env python3
"""Dump a per-step closed-loop state trace from MuJoCo, in the same schema as the Isaac twin.

Why this exists: the same-target replay showed Isaac's plant produces 0.72-1.36 yaw from MuJoCo's wheel
targets while Isaac's closed loop produces only 0.05-0.31, and a field-by-field audit showed the 65-dim
observation is built identically on both sides.  So the closed-loop gap must come from the STATES
diverging.  To find the first diverging field we need the same quantities recorded step by step on both
sides and aligned on the same phase.

Schema (identical in r11_isaac_state_trace.py):
    obs     65 values  - exactly what the policy sees
    act     16 values  - the clipped policy output
    q       16 values  - measured joint positions in contract order
    dq      16 values  - measured joint velocities
    quat     4 values  - root orientation
    angvel   3 values  - root angular velocity in the WORLD frame (both sides convert the same way)
    yaw      1 value   - root angular velocity z, the scoring quantity

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
LEG_KD, WHEEL_KD, KP_LEG = 2.0, 6.0, 55.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])


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

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    prev = np.zeros(16); zr = [None]; ep = 0

    trace = {"sim": "mujoco", "checkpoint": str(a.ckpt), "cmd": [a.vx, 0.0, a.wz],
             "step_dt": STEP_DT, "settle_s": a.settle_s, "settle": [], "turn": []}

    def record(bucket):
        bucket.append({
            "obs": [round(float(x), 6) for x in obs_of(prev, ep, zr)],
            "act": [round(float(x), 6) for x in prev],
            "q": [round(float(x), 6) for x in d.qpos[qadr]],
            "dq": [round(float(x), 6) for x in d.qvel[vadr]],
            "quat": [round(float(x), 6) for x in d.qpos[3:7]],
            "angvel": [round(float(x), 6) for x in d.qvel[3:6]],
            "yaw": round(float(d.qvel[5]), 6)})

    cmd["v"] = np.zeros(3)
    for _ in range(int(a.settle_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        record(trace["settle"])
        step_ctrl(prev); ep += 1

    cmd["v"] = np.array([a.vx, 0.0, a.wz])
    for _ in range(int(a.turn_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
        record(trace["turn"])
        step_ctrl(prev); ep += 1

    (a.out / "state_trace.json").write_text(json.dumps(trace))
    f = trace["turn"][len(trace["turn"]) // 2]
    print("settle frames %d, turn frames %d" % (len(trace["settle"]), len(trace["turn"])))
    print("mid-turn yaw=%.5f  wheel dq=%s" % (f["yaw"], [round(x, 2) for x in f["dq"][12:]]))
    print("MJ_STATE_TRACE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
