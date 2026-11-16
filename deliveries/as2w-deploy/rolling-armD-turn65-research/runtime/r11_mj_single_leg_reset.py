#!/usr/bin/env python3
"""Operator work-order item 3: can the worst settled state take a safe first step, one leg at a time?

The operator narrowed this: only run single-leg reset for cases where the correction drags wheels, hits
limits or lowers clearance.  Items 1 and 2 showed the four plain turning cases do not qualify - return to
stand already recovers 70-89% of their contact-point deviation - so the candidate set is the worst SETTLED
state per side, and the leg with the largest contact-point error in that state.

Per trial:
  settle -> turn -> release to a settled state
  pick the target leg (largest contact-point error)
  record the three-foot support margin BEFORE lifting: load on the three remaining wheels, and the
    horizontal CoM distance to the nearest edge of the triangle those three wheels form
  swing that leg up to a reset target and back down
  record the swing leg's minimum ground clearance, the post-landing load on that wheel, and the body
    yaw gained during the step (a single-leg reset must not add heading error)

Also dumps a top-down geometry record per frame (wheel centres, support triangle, CoM) so a fixed top-down
video can be rendered with the current footfall, the target footfall and the support triangle marked.
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
LIFT_S = 0.35
HOLD_S = 0.35
LOWER_S = 0.35


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


def point_tri_dist(p, a, b):
    """Signed distance from p to segment ab, positive on the left."""
    ab = b - a
    n = np.array([-ab[1], ab[0]])
    nn_ = np.linalg.norm(n)
    if nn_ < 1e-9:
        return float(np.linalg.norm(p - a))
    return float(np.dot(p - a, n) / nn_)


def tri_support_margin(p, pts):
    """Minimum signed distance from p to the three edges of the triangle; negative means outside."""
    d = [point_tri_dist(p, pts[i], pts[(i + 1) % 3]) for i in range(3)]
    return float(min(d)) if all(x >= 0 or x <= 0 for x in d) and not (min(d) < 0 < max(d)) else float(min(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--lift-m", type=float, default=0.06,
                    help="how high to raise the target wheel above its current height")
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

    def step_ctrl(act16, leg_target_abs=None):
        act16 = np.clip(np.asarray(act16, float), -CLIP, CLIP)
        leg_t = act16[:12] * LEG_SCALE + DEF[:12] if leg_target_abs is None else leg_target_abs
        tgt = np.r_[leg_t, act16[12:] * WHEEL_SCALE]
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

    def world_wheels():
        return np.array([d.xpos[b][:2] for b in wb])

    def com_xy():
        tot = 0.0; acc = np.zeros(2)
        for i in range(m.nbody):
            mm = m.body_mass[i]
            tot += mm; acc += mm * d.xpos[i][:2]
        return acc / max(tot, 1e-9)

    prev = np.zeros(16); zr = [None]; ep = 0

    def run_policy(secs, vx, wz):
        nonlocal ep
        cmd["v"] = np.array([vx, 0.0, wz])
        for _ in range(int(secs / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                pv = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            prev[:] = pv
            step_ctrl(prev); ep += 1

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    run_policy(a.settle_s, 0.0, 0.0)
    ref_wheels = world_wheels()
    run_policy(a.turn_s, a.vx, a.wz)
    run_policy(a.release_s, 0.0, 0.0)

    settled_loads = loads()
    settled_wheels = world_wheels()
    dpos = settled_wheels - ref_wheels
    target = int(np.argmax(np.abs(dpos).max(axis=1)))
    others = [i for i in range(4) if i != target]

    # pre-lift: three-foot support margin using the three remaining wheels
    tri = settled_wheels[others]
    # order the triangle consistently (by angle around its centroid)
    c = tri.mean(axis=0)
    order = np.argsort(np.arctan2(tri[:, 1] - c[1], tri[:, 0] - c[0]))
    tri = tri[order]
    com = com_xy()
    margin = tri_support_margin(com, tri)

    yaw0 = float(np.arctan2(quat_to_mat(d.qpos[3:7])[1, 0], quat_to_mat(d.qpos[3:7])[0, 0]))
    z0 = float(d.qpos[2])
    ground_z = float(d.xpos[wb[target]][2])

    poses = []
    leg_ids = slice(target * 3, target * 3 + 3)
    base_q = d.qpos[qadr][leg_ids].copy()
    min_clear = 1e9
    # lift, hold, lower
    for phase, secs in (("lift", LIFT_S), ("hold", HOLD_S), ("lower", LOWER_S)):
        n = int(secs / STEP_DT)
        for k in range(n):
            if phase == "lift":
                frac = (k + 1) / n
            elif phase == "hold":
                frac = 1.0
            else:
                frac = 1.0 - (k + 1) / n
            tgt = d.qpos[qadr][:12].copy()
            # raise the target wheel by moving its hip/thigh/calf target vertically via a small offset
            tgt[leg_ids] = base_q + frac * np.array([0.0, -0.35, 0.55])
            step_ctrl(np.zeros(16), leg_target_abs=tgt)
            clr = float(d.xpos[wb[target]][2])
            min_clear = min(min_clear, clr - ground_z + a.lift_m * 0)
            poses.append({"phase": phase, "wheel_xy": d.xpos[wb[target]][:2].tolist(),
                          "com_xy": com_xy().tolist(), "wheels_xy": world_wheels().tolist()})
    post_loads = loads()
    yaw1 = float(np.arctan2(quat_to_mat(d.qpos[3:7])[1, 0], quat_to_mat(d.qpos[3:7])[0, 0]))
    dyaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
    post_wheels = world_wheels()

    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "target_leg": LEGS[target], "supporting_legs": [LEGS[i] for i in others],
           "settled_contact_dpos_m": {LEGS[i]: [round(float(dpos[i][0]), 4), round(float(dpos[i][1]), 4)]
                                       for i in range(4)},
           "pre_lift": {"loads_N": [round(float(x), 1) for x in settled_loads],
                         "three_foot_support_margin_m": round(margin, 4),
                         "com_xy": [round(float(x), 4) for x in com],
                         "support_triangle": [[round(float(v), 4) for v in p] for p in tri]},
           "swing": {"planned_lift_m": a.lift_m,
                      "min_wheel_height_relative_to_start_m": round(min_clear, 4)},
           "post_land": {"loads_N": [round(float(x), 1) for x in post_loads],
                          "target_wheel_reloade_OK": bool(post_loads[target] > LOAD_N),
                          "legs_unloaded": int((post_loads <= LOAD_N).sum())},
           "body": {"yaw_added_rad": dyaw, "com_shift_xy_m": [round(float(x), 4) for x in (com_xy() - com)],
                     "target_wheel_shift_xy_m": [round(float(x), 4) for x in (post_wheels[target] - settled_wheels[target])]},
           "topdown_frames": poses[:120]}
    (a.out / "single_leg_reset.json").write_text(json.dumps(rep, indent=1))
    print("target=%s  pre_lift margin=%.4f m  loads=%s" % (LEGS[target], margin,
          ["%.0f" % x for x in settled_loads]))
    print("  swing min wheel height rel start = %.4f m (planned lift %.3f m)" % (min_clear, a.lift_m))
    print("  post-land loads=%s  target reloaded=%s  unloaded=%d"
          % (["%.0f" % x for x in post_loads], rep["post_land"]["target_wheel_reloade_OK"],
             rep["post_land"]["legs_unloaded"]))
    print("  yaw added = %+.4f rad  target wheel shift = %s m"
          % (dyaw, rep["body"]["target_wheel_shift_xy_m"]))
    print("SINGLE_LEG_RESET_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
