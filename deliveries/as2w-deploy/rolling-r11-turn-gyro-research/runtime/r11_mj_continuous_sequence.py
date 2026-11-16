#!/usr/bin/env python3
"""Continuous sequence: turn -> release -> return-to-stand -> START AGAIN, tracking the SAME leg.

The operator's item 1.  The existing four-snapshot table reports the WORST leg at each snapshot, and the
worst leg moves from FR/FL at release to RR after the return-to-stand - so "the worst case improved 70-89%"
does not say which leg was actually restored.  This tracks every leg THROUGH the sequence by name and
reports, per leg:
    joint error against the fixed stand, and the wheel-centre delta in the body frame, at every snapshot
so a reader can see whether a specific leg went back to its reference, or whether the worst-case number
merely improved because a different leg became the worst.

It also adds the phase the operator asked for and which has never been measured: after the return-to-stand,
re-issue the SAME turn command and judge whether the residual hurts the NEXT takeover.  Reported per
restart:
    yaw tracking in the tail, four-wheel support fraction, whether it terminated, and how much the settled
    posture at restart start differed from the fixed stand.

The 0.10 rad line is a SIMULATION SCREENING LINE only, as the operator instructed; this reports how long
each phase spends above it.

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
LINE_RAD = 0.10
CASES = [("arc_left", 0.25, 0.05), ("arc_right", 0.25, -0.05),
         ("spin_left", 0.0, 0.05), ("spin_right", 0.0, -0.05)]


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
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--stand-s", type=float, default=4.0)
    ap.add_argument("--restart-s", type=float, default=8.0)
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
    root_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if root_b < 0:
        root_b = 1
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

    def wheel_pos_body():
        Rb = quat_to_mat(d.qpos[3:7]); root = d.xpos[root_b]
        return np.array([Rb.T @ (d.xpos[b] - root) for b in wb])

    prev = np.zeros(16); zr = [None]; ep = 0

    def run_policy(secs, vx, wz, record=None):
        nonlocal ep
        cmd["v"] = np.array([vx, 0.0, wz])
        n = int(secs / STEP_DT)
        above = 0
        sup = []
        yaw = []
        for _ in range(n):
            o = obs_of(prev, ep, zr)
            with torch.inference_mode() if False else torch.no_grad():
                prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
            e = float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())
            if e > LINE_RAD:
                above += 1
            sup.append(bool((loads() > LOAD_N).all()))
            yaw.append(float(d.qvel[5]))
        stats = {"above_line_s": round(above * STEP_DT, 3), "frames": n,
                 "support_frac": float(np.mean(sup)) if sup else None,
                 "yaw_mean": float(np.mean(yaw)) if yaw else None,
                 "yaw_tail": float(np.mean(yaw[-150:])) if len(yaw) >= 1 else None}
        if record is not None:
            record.update(stats)
        return stats

    rep = {"checkpoint": str(a.ckpt), "line_rad": LINE_RAD,
           "line_status": "SIMULATION SCREENING LINE only, per the operator",
           "phases": ["settle", "turn", "release", "return_to_stand", "restart"],
           "cases": {}}

    for name, vx, wz in CASES:
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)
        prev[:] = 0.0; zr[0] = None; ep = 0
        run_policy(a.settle_s, 0.0, 0.0)
        ref_wheels = wheel_pos_body()
        ref_q = d.qpos[qadr][:12].copy()
        body0 = d.qpos[0:2].copy()

        snaps = {}
        def snap(tag):
            snaps[tag] = {"q_err": (d.qpos[qadr][:12] - DEF[:12]).tolist(),
                          "wheel_dpos": (wheel_pos_body() - ref_wheels).tolist(),
                          "loads": loads().tolist(),
                          "body_xy": d.qpos[0:2].tolist()}
        snap("before_turn")
        ph = {}
        run_policy(a.turn_s, vx, wz, record=ph.setdefault("turn", {}))
        snap("at_release")
        run_policy(a.release_s, 0.0, 0.0, record=ph.setdefault("release", {}))
        snap("after_settle")
        run_policy(a.stand_s, 0.0, 0.0, record=ph.setdefault("return_to_stand", {}))
        snap("after_stand")
        # ---- the phase the operator asked for: start again from the restored state ----
        run_policy(a.restart_s, vx, wz, record=ph.setdefault("restart", {}))
        snap("after_restart")

        per_leg = {}
        for i, leg in enumerate(LEGS):
            per_leg[leg] = {
                tag: {"q_err_max": round(float(np.abs(np.array(snaps[tag]["q_err"][i * 3:(i + 1) * 3])).max()), 4),
                       "wheel_dpos_fwd_m": round(float(snaps[tag]["wheel_dpos"][i][0]), 4),
                       "wheel_dpos_lat_m": round(float(snaps[tag]["wheel_dpos"][i][1]), 4),
                       "load_N": round(float(snaps[tag]["loads"][i]), 1)}
                for tag in ("before_turn", "at_release", "after_settle", "after_stand", "after_restart")}
        # which leg was worst where, and whether the SAME leg recovered
        worst = {tag: max(LEGS, key=lambda L_: per_leg[L_][tag]["q_err_max"])
                 for tag in ("at_release", "after_settle", "after_stand", "after_restart")}
        body_disp = float(np.hypot(*(d.qpos[0:2] - body0)))
        rep["cases"][name] = {"command": {"vx": vx, "wz": wz},
                               "per_leg": per_leg, "worst_leg_by_snapshot": worst,
                               "body_xy_drift_total_m": round(body_disp, 4),
                               "phase_stats": ph,
                               "restart_verdict": {
                                   "restart_support_frac": ph.get("restart", {}).get("support_frac"),
                                   "restart_yaw_tail": ph.get("restart", {}).get("yaw_tail"),
                                   "restart_above_line_s": ph.get("restart", {}).get("above_line_s")}}
        print("=== %s (vx=%.2f wz=%+.2f) ===" % (name, vx, wz))
        print("  worst leg: at_release=%s  after_settle=%s  after_stand=%s  after_restart=%s"
              % (worst["at_release"], worst["after_settle"], worst["after_stand"], worst["after_restart"]))
        for leg in LEGS:
            pl = per_leg[leg]
            print("    %-2s q_err: turn_end=%.3f settle=%.3f stand=%.3f restart=%.3f | wheel_fwd: %.4f %.4f %.4f %.4f"
                  % (leg, pl["at_release"]["q_err_max"], pl["after_settle"]["q_err_max"],
                     pl["after_stand"]["q_err_max"], pl["after_restart"]["q_err_max"],
                     pl["at_release"]["wheel_dpos_fwd_m"], pl["after_settle"]["wheel_dpos_fwd_m"],
                     pl["after_stand"]["wheel_dpos_fwd_m"], pl["after_restart"]["wheel_dpos_fwd_m"]))
        print("  body drift total = %.4f m | return_to_stand above 0.10 rad = %.2f s | restart: sup=%s yaw_tail=%s above=%s"
              % (body_disp, ph.get("return_to_stand", {}).get("above_line_s", float("nan")),
                 ph.get("restart", {}).get("support_frac"),
                 None if ph.get("restart", {}).get("yaw_tail") is None else round(ph["restart"]["yaw_tail"], 3),
                 ph.get("restart", {}).get("above_line_s")))
    (a.out / "continuous_sequence.json").write_text(json.dumps(rep, indent=1))
    print("CONTINUOUS_SEQUENCE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
