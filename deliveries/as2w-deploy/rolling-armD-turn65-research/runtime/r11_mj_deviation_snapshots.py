#!/usr/bin/env python3
"""Four-snapshot per-leg deviation dump: is the WHEEL CONTACT POINT of each leg displaced, or only
the joint angles and body pose?

The operator's key question.  Joint-angle error alone cannot answer it: a leg can sit far from the fixed
stand in joint space while its wheel still touches the ground in the same place (the body moved instead),
and conversely the joints can look close while the contact point has slid.  So this records, per leg, the
WHEEL CENTRE POSITION IN THE BODY FRAME - fore/aft and lateral - against the fixed-stand reference, next to
the joint error, the wheel load and the knee clearance.

One continuous trajectory per case, four snapshots:

    before_turn    after settling at zero command    <- this IS the fixed-stand reference
    at_release     the instant the turn command is dropped
    after_settle   after the release has settled
    after_stand    after the legs are commanded back to the default stance

That ordering is what separates "already deviated before release" (a turning problem) from "dragged
during the return-to-stand transition" (a transition problem) - the two need different fixes, and the
operator asked for exactly that split.

Cases: left arc, right arc, left spin, right spin - same checkpoint for all four.

CPU-only so it does not contend with training.
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
CASES = [("arc_left", 0.25, 0.05), ("arc_right", 0.25, -0.05),
         ("spin_left", 0.0, 0.05), ("spin_right", 0.0, -0.05)]
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
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--stand-s", type=float, default=3.0)
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
    wheel_set = set(wb)
    ground_gid = -1
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm == "ground" or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            ground_gid = g; break
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))
    cmd = {"v": np.zeros(3)}
    applied = np.zeros(16)

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
        nonlocal applied
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        applied = act16.copy()
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

    def wheel_body_pos():
        Rb = quat_to_mat(d.qpos[3:7])
        root = d.xpos[root_b]
        out = []
        for b in wb:
            rel = d.xpos[b] - root
            out.append(Rb.T @ rel)
        return np.array(out)          # (4,3) wheel centre in body frame

    def knee_clear():
        Rb = quat_to_mat(d.qpos[3:7]); root = d.xpos[root_b]
        out = []
        for b in kb:
            p = d.xpos[b].copy()
            gid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, -1.0]), None, 1, ground_gid if ground_gid >= 0 else -1, gid)
            out.append(float(dist) if dist >= 0 else None)
        return out

    def snap(tag):
        return {"tag": tag,
                "q_err": (d.qpos[qadr][:12] - DEF[:12]).tolist(),
                "wheel_body_pos": wheel_body_pos().tolist(),
                "load_N": loads().tolist(),
                "knee_clear_m": knee_clear(),
                "body_xyz": d.qpos[0:3].tolist(),
                "body_up": float(-(quat_to_mat(d.qpos[3:7]).T @ np.array([0.0, 0.0, -1.0]))[2]),
                "applied_wheel": applied[12:].tolist()}

    rep = {"checkpoint": str(a.ckpt), "load_threshold_n": LOAD_N,
           "note": "wheel_body_pos is the wheel CENTRE in the body frame; comparing it against the "
                   "before_turn reference separates a displaced contact point from a body-pose change",
           "cases": {}}
    for name, vx, wz in CASES:
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
        S = {"before_turn": snap("before_turn")}
        ref_pos = np.array(S["before_turn"]["wheel_body_pos"])
        cmd["v"] = np.array([vx, 0.0, wz])
        for _ in range(int(a.turn_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        S["at_release"] = snap("at_release")
        cmd["v"] = np.zeros(3)
        for _ in range(int(a.release_s / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        S["after_settle"] = snap("after_settle")
        for _ in range(int(a.stand_s / STEP_DT)):
            step_ctrl(np.zeros(16))
        S["after_stand"] = snap("after_stand")

        out = {}
        for tag, s in S.items():
            pos = np.array(s["wheel_body_pos"])
            dpos = pos - ref_pos
            out[tag] = {
                "per_leg": {
                    leg: {"q_err_per_joint": [round(v, 4) for v in s["q_err"][i * 3:(i + 1) * 3]],
                           "q_err_max": round(float(np.abs(s["q_err"][i * 3:(i + 1) * 3]).max()), 4),
                           "wheel_dpos_fwd_m": round(float(dpos[i][0]), 4),
                           "wheel_dpos_lat_m": round(float(dpos[i][1]), 4),
                           "wheel_dpos_vert_m": round(float(dpos[i][2]), 4),
                           "load_N": round(float(s["load_N"][i]), 2),
                           "knee_clear_m": (None if s["knee_clear_m"][i] is None
                                             else round(float(s["knee_clear_m"][i]), 4))}
                    for i, leg in enumerate(LEGS)},
                "body_up": round(float(s["body_up"]), 4),
                "body_xyz": [round(float(v), 4) for v in s["body_xyz"]],
                "any_leg_up": int(sum(1 for x in s["load_N"] if x <= LOAD_N)),
            }
        rep["cases"][name] = {"command": {"vx": vx, "wz": wz}, "snapshots": out}
        print("=== %s (vx=%.2f wz=%+.2f) ===" % (name, vx, wz))
        for tag in ("before_turn", "at_release", "after_settle", "after_stand"):
            o = out[tag]
            worst = max(LEGS, key=lambda L_: o["per_leg"][L_]["q_err_max"])
            wp = o["per_leg"][worst]
            dmax = max(abs(o["per_leg"][L_]["wheel_dpos_fwd_m"]) for L_ in LEGS)
            dlat = max(abs(o["per_leg"][L_]["wheel_dpos_lat_m"]) for L_ in LEGS)
            print("  %-13s worst_leg=%s q_err=%.3f  |d_fwd|max=%.4f |d_lat|max=%.4f  loads=%s  legs_up=%d"
                  % (tag, worst, wp["q_err_max"], dmax, dlat,
                     ["%.0f" % x for x in [o["per_leg"][L_]["load_N"] for L_ in LEGS]], o["any_leg_up"]))
    (a.out / "snapshots.json").write_text(json.dumps(rep, indent=1))
    print("SNAPSHOTS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
