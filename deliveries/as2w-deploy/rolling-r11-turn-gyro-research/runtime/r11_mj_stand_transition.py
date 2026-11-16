#!/usr/bin/env python3
"""Q2: does the return-to-stand transition drag the wheels, and is 1.11x90% the DEMAND or the APPLIED torque?

Two things the operator asked for explicitly:

1. The reported "1.11 x 90%" must be classified.  In every earlier script the torque was computed and then
   clamped before being written to the actuator, and the peak was recorded AFTER the clamp - so 1.11 is the
   APPLIED value (clamped at the 13.5 N*m cap, which is 111% of the 12.15 N*m 90% budget) while the DEMAND
   was far larger.  This script records both separately so the distinction is on the record rather than
   inferred from reading code.

2. Why the current return-to-stand might drag the wheels: during the transition, compare the wheel speed
   TARGET against the MEASURED speed while recording the applied torque and the body displacement, and run a
   control arm that instead does a small four-wheel-grounded correction (a much smaller posture step) from the
   same settled initial state.

Trajectory per trial: settle, then a turn to a settled state, then one of two return strategies:

    current_stand    legs commanded straight to the default stance, wheels zeroed
    grounded_trim    only a SMALL fraction of the posture error is corrected per step, wheels zeroed,
                     so the feet stay planted instead of being dragged

Judged on: wheel slip (target vs measured), wheel torque demand vs applied, body displacement during the
transition, and the residual posture error afterwards.
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
    ap.add_argument("--stand-s", type=float, default=3.0)
    ap.add_argument("--trim-frac", type=float, default=0.25,
                    help="fraction of the posture error corrected per step in the grounded_trim arm")
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

    tdemand = np.zeros(16)
    tapplied = np.zeros(16)

    def step_ctrl(act16, leg_target_abs=None):
        nonlocal tdemand, tapplied
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        if leg_target_abs is None:
            leg_t = act16[:12] * LEG_SCALE + DEF[:12]
        else:
            leg_t = leg_target_abs
        tgt = np.r_[leg_t, act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            tdemand = np.maximum(tdemand, np.abs(tau))
            tau_c = np.clip(tau, -LIM, LIM)
            tapplied = np.maximum(tapplied, np.abs(tau_c))
            d.ctrl[aid] = tau_c
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

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)

    def run_to_settled():
        reset()
        prev = np.zeros(16); zr = [None]; ep = 0
        cmd["v"] = np.zeros(3)
        for _ in range(int(a.settle_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
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

    def transition(mode):
        tdemand[:] = 0.0; tapplied[:] = 0.0
        x0 = float(d.qpos[0]); y0 = float(d.qpos[1])
        slip, body_disp = [], []
        n = int(a.stand_s / STEP_DT)
        for k in range(n):
            if mode == "current_stand":
                step_ctrl(np.zeros(16))
            else:
                err = d.qpos[qadr][:12] - DEF[:12]
                leg_t = d.qpos[qadr][:12] - a.trim_frac * err      # correct a fraction only
                step_ctrl(np.zeros(16), leg_target_abs=leg_t)
            wt = np.zeros(4)                                        # wheel target is zero in both arms
            wmeas = d.qvel[vadr][12:]
            slip.append(float(np.abs(wmeas).max()))
            body_disp.append(float(np.hypot(d.qpos[0] - x0, d.qpos[1] - y0)))
        post_err = float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())
        post_load = loads()
        return {"mode": mode,
                "wheel_speed_max_rad_s": float(np.mean(slip)),
                "wheel_speed_peak_rad_s": float(np.max(slip)),
                "body_disp_max_m": float(np.max(body_disp)),
                "body_disp_end_m": float(body_disp[-1]),
                "posture_err_after_rad": post_err,
                "post_load_N": [round(float(x), 1) for x in post_load],
                "legs_unloaded_after": int((post_load <= LOAD_N).sum()),
                "torque_demand_peak": {"hip": float(tdemand[:4].max()), "thigh": float(tdemand[4:8].max()),
                                        "calf": float(tdemand[8:12].max()), "wheel": float(tdemand[12:].max())},
                "torque_applied_peak": {"hip": float(tapplied[:4].max()), "thigh": float(tapplied[4:8].max()),
                                         "calf": float(tapplied[8:12].max()), "wheel": float(tapplied[12:].max())},
                "frac_of_90pct_demand": {"hip": float(tdemand[:4].max() / (0.9 * 54)),
                                          "thigh": float(tdemand[4:8].max() / (0.9 * 54)),
                                          "calf": float(tdemand[8:12].max() / (0.9 * 81)),
                                          "wheel": float(tdemand[12:].max() / (0.9 * 13.5))},
                "frac_of_90pct_applied": {"hip": float(tapplied[:4].max() / (0.9 * 54)),
                                           "thigh": float(tapplied[4:8].max() / (0.9 * 54)),
                                           "calf": float(tapplied[8:12].max() / (0.9 * 81)),
                                           "wheel": float(tapplied[12:].max() / (0.9 * 13.5))}}

    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "note": "torque_demand_peak is recorded BEFORE the 90%/hard clamp; torque_applied_peak AFTER it, "
                   "so the 1.11x90% figure can be classified directly instead of inferred",
           "arms": {}}
    for mode in ("current_stand", "grounded_trim"):
        run_to_settled()
        rep["arms"][mode] = transition(mode)
        r = rep["arms"][mode]
        print("%-14s wheel_speed_peak=%.2f rad/s  body_disp=%.4f m  posture_err=%.3f rad  "
              "tau_wheel demand=%.2f applied=%.2f (x90%%: demand %.2f applied %.2f)"
              % (mode, r["wheel_speed_peak_rad_s"], r["body_disp_max_m"], r["posture_err_after_rad"],
                 r["torque_demand_peak"]["wheel"], r["torque_applied_peak"]["wheel"],
                 r["frac_of_90pct_demand"]["wheel"], r["frac_of_90pct_applied"]["wheel"]))
    for k in ("hip", "thigh", "calf"):
        print("  %-5s demand_peak=%6.2f  applied_peak=%6.2f  N*m" % (k,
              rep["arms"]["current_stand"]["torque_demand_peak"][k],
              rep["arms"]["current_stand"]["torque_applied_peak"][k]))
    (a.out / "stand_transition.json").write_text(json.dumps(rep, indent=1))
    print("STAND_TRANSITION_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
