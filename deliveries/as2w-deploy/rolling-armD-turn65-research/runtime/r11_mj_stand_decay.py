#!/usr/bin/env python3
"""Does the post-turn leg offset decay once the legs are commanded back to fixed stand?

The operator observed that after turning, the legs sit offset from fixed stand, and asked whether stepping
could restore them.  Analysis showed the zero-command target already IS fixed stand (the stepping action
semantics are q = a*0.35 + DEFAULT, so a zero action targets the default pose), which means the residual
is either (a) a slow tracking response that will decay if given time, or (b) the statically loaded
equilibrium of a compliant PD, in which case no amount of time will remove it and only unloading each leg
(stepping) can.

This distinguishes the two.  Sequence, on one continuous trajectory:

    1. settle, then run a turn (wz = +-0.05) for 12 s with the policy
    2. hand over: legs commanded to the fixed-stand pose (action 0) and wheels to zero, held for 6 s
    3. report the max leg joint error against the default stance at the moment of handover and at
       +0.5, +1, +2, +4 and +6 s afterwards

If the error decays toward zero the residual is just slow settling.  If it plateaus at a non-zero value,
that plateau is the loaded equilibrium and stepping is the right remedy.  CPU-only, so it does not contend
with training.
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
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
MARKS_S = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]


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
        m = nn.Linear(w.shape[1], w.shape[0]); m.weight.data = w.clone(); m.bias.data = b.clone()
        layers.append(m)
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
    ap.add_argument("--hold-s", type=float, default=6.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, 55.0), np.full(4, 0.0)]
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
        tgt = np.r_[act16[:12] * LEG_SCALE + DEF[:12], act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            d.ctrl[aid] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)

    def leg_err():
        return float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())

    reset()
    cmd["v"] = np.zeros(3)
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

    err_at_handover = leg_err()
    # handover: legs to the fixed-stand pose (action 0), wheels to zero, policy no longer drives
    samples = []
    total = int(a.hold_s / STEP_DT)
    for k in range(total):
        cmd["v"] = np.zeros(3)
        step_ctrl(np.zeros(16))      # legs target DEF, wheels target 0
        t = (k + 1) * STEP_DT
        if any(abs(t - ms) < STEP_DT / 2 for ms in MARKS_S):
            samples.append({"t_s": round(t, 3), "leg_err_max": leg_err()})
    # make sure the last mark exists
    if not samples or abs(samples[-1]["t_s"] - a.hold_s) > STEP_DT:
        samples.append({"t_s": round(a.hold_s, 3), "leg_err_max": leg_err()})

    rep = {"checkpoint": str(a.ckpt), "wz": a.wz, "settle_s": a.settle_s,
           "turn_s": a.turn_s, "hold_s": a.hold_s,
           "err_at_handover": err_at_handover, "samples": samples,
           "note": "legs are commanded to the DEFAULT pose from handover on; a decaying error means slow settling, a plateau means a loaded equilibrium"}
    (a.out / "stand_decay.json").write_text(json.dumps(rep, indent=1))
    print("HANDOVER err=%.3f rad" % err_at_handover)
    for s in samples:
        print("  t=+%.2fs  leg_err_max=%.3f rad" % (s["t_s"], s["leg_err_max"]))
    print("STAND_DECAY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
