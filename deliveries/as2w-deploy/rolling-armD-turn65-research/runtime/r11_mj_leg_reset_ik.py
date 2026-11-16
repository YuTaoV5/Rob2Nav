#!/usr/bin/env python3
"""Q3 redo: lift a leg by FOOT-POSITION IK with three-leg support compensation.

The first attempt failed for a specific, recorded reason: it added a raw joint-angle offset as if that were a
vertical foot lift, so the wheel went DOWN (-0.02 to -0.03 m) and heading was disturbed 16-29 deg.  This
version fixes that cause rather than changing the goal:

  lift           numerically solve the three leg joints so the WHEEL CENTRE rises by lift_m in the world,
                 using a damped-least-squares IK on the measured jacobian - verified against the actual
                 wheel height, not assumed from the joint change
  compensation   during the lift the other three legs extend a little so they take the load the lifted leg
                 gave up; two arms, compensation off and on, from the same settled state, so the cost of
                 omitting it is visible rather than argued

Reports the quantities the operator asked for: pre-lift three-foot support margin, achieved swing clearance
(measured, not planned), post-landing wheel loads and the heading gained during the step.  CPU-only.
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
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--lift-m", type=float, default=0.03)
    ap.add_argument("--comp-m", type=float, default=0.015,
                    help="how much the other three legs extend, per leg, to take over the lifted load")
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

    def solve_leg_ik(leg, dz_world):
        """Damped least squares on the 3 leg joints so the wheel centre moves dz in world z."""
        idx = leg * 3
        qidx = qadr[idx:idx + 3]
        target = d.xpos[wb[leg]].copy() + np.array([0.0, 0.0, dz_world])
        saved = d.qpos[qidx].copy()
        for _ in range(60):
            p0 = d.xpos[wb[leg]].copy()
            err = target - p0
            if np.linalg.norm(err) < 1e-4:
                break
            J = np.zeros((3, 3))
            for j in range(3):
                d.qpos[qidx[j]] += 1e-5
                mujoco.mj_forward(m, d)
                pj = d.xpos[wb[leg]].copy()
                d.qpos[qidx[j]] = saved[j] if False else d.qpos[qidx[j]] - 1e-5
                mujoco.mj_forward(m, d)
                J[:, j] = (pj - p0) / 1e-5
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            d.qpos[qidx] += np.clip(dq, -0.15, 0.15)
            mujoco.mj_forward(m, d)
            saved = d.qpos[qidx].copy()
        res = d.qpos[qidx].copy()
        d.qpos[qidx] = DEF[idx:idx + 3]
        mujoco.mj_forward(m, d)
        # residual achieved height if we apply res
        d.qpos[qidx] = res
        mujoco.mj_forward(m, d)
        achieved = float(d.xpos[wb[leg]][2] - p0[2])
        d.qpos[qidx] = DEF[idx:idx + 3]
        mujoco.mj_forward(m, d)
        return res, achieved

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

    def fresh_turn():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)
        prev[:] = 0.0; zr[0] = None; ep = 0
        run_policy(a.settle_s, 0.0, 0.0)
        run_policy(a.turn_s, a.vx, a.wz)
        run_policy(a.release_s, 0.0, 0.0)

    def tri_margin(legs3, com):
        pts = np.array([d.xpos[wb[i]][:2] for i in legs3])
        c = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        pts = pts[order]
        ds = []
        for i in range(3):
            p1, p2 = pts[i], pts[(i + 1) % 3]
            ab = p2 - p1
            n = np.array([-ab[1], ab[0]]);
            nn_ = np.linalg.norm(n)
            ds.append(float(np.dot(com - p1, n) / nn_) if nn_ > 1e-9 else 0.0)
        return float(min(ds))

    def com_xy():
        tot = 0.0; acc = np.zeros(2)
        for i in range(m.nbody):
            mm = m.body_mass[i]; tot += mm; acc += mm * d.xpos[i][:2]
        return acc / max(tot, 1e-9)

    def yaw():
        R = quat_to_mat(d.qpos[3:7])
        return float(np.arctan2(R[1, 0], R[0, 0]))

    rep = {"checkpoint": str(a.ckpt), "command": {"vx": a.vx, "wz": a.wz},
           "lift_m_requested": a.lift_m, "comp_m": a.comp_m,
           "note": "lift is solved by FOOT-POSITION IK and the achieved wheel rise is measured, because the "
                   "first attempt used joint offsets and the wheel went DOWN",
           "arms": {}}
    for comp in (False, True):
        fresh_turn()
        settled_loads = loads()
        dpos = np.array([d.xpos[b][:2] for b in wb]) - np.array([d.xpos[b][:2] for b in wb])
        target = int(np.argmax(np.abs(loads() - loads().mean())))  # placeholder, overridden below
        # choose the leg with the largest pose error against DEF
        qerr = np.abs(d.qpos[qadr][:12] - DEF[:12]).reshape(4, 3)
        target = int(np.argmax(qerr.max(axis=1)))
        others = [i for i in range(4) if i != target]
        com0 = com_xy()
        margin = tri_margin(others, com0)
        yaw0 = yaw()
        z0 = float(d.xpos[wb[target]][2])
        leg_ik, achieved_planned = solve_leg_ik(target, a.lift_m)

        base = d.qpos[qadr][:12].copy()
        min_z, max_yaw_dev, roll_dev = 1e9, 0.0, 0.0
        n = int((0.35 + 0.35 + 0.35) / STEP_DT)
        for k in range(n):
            frac = min(1.0, (k + 1) / max(1, int(0.35 / STEP_DT)))
            if k > int(0.35 / STEP_DT):
                frac = max(0.0, 1.0 - (k - int(0.35 / STEP_DT)) / max(1, int(0.35 / STEP_DT)))
            tgt = base.copy()
            tgt[target * 3:target * 3 + 3] = base[target * 3:target * 3 + 3] + frac * (leg_ik - base[target * 3:target * 3 + 3])
            if comp:
                for o in others:
                    tgt[o * 3:o * 3 + 3] = base[o * 3:o * 3 + 3] + frac * np.array([0.0, 0.10, -0.20]) * (a.comp_m / 0.015)
            step_ctrl(np.zeros(16), leg_target_abs=tgt)
            min_z = min(min_z, float(d.xpos[wb[target]][2]))
            max_yaw_dev = max(max_yaw_dev, abs(np.arctan2(np.sin(yaw() - yaw0), np.cos(yaw() - yaw0))))
            R = quat_to_mat(d.qpos[3:7])
            roll_dev = max(roll_dev, abs(float(np.arctan2(R[2, 1], R[2, 2]))))
        # settle back
        for _ in range(int(1.0 / STEP_DT)):
            step_ctrl(np.zeros(16))
        post = loads()
        r = {"compensation": comp, "target_leg": LEGS[target],
              "pre_lift_three_foot_margin_m": round(margin, 4),
              "ik_planned_wheel_rise_m": round(achieved_planned, 4),
              "achieved_swing_rise_m": round(min_z - z0, 4),
              "comp_applied": comp,
              "max_heading_dev_rad": round(max_yaw_dev, 4),
              "max_roll_dev_rad": round(roll_dev, 4),
              "post_land_loads_N": [round(float(x), 1) for x in post],
              "post_unloaded_legs": int((post <= LOAD_N).sum()),
              "target_wheel_reloaded": bool(post[target] > LOAD_N)}
        rep["arms"]["comp_on" if comp else "comp_off"] = r
        print("comp=%-5s target=%s  margin=%.4f m  swing_rise=%+.4f m (ik planned %+.4f)  "
              "heading_dev=%.4f roll_dev=%.4f  post_loads=%s unloaded=%d"
              % (comp, LEGS[target], margin, r["achieved_swing_rise_m"], achieved_planned,
                 max_yaw_dev, roll_dev, ["%.0f" % x for x in post], r["post_unloaded_legs"]))
    (a.out / "leg_reset_ik.json").write_text(json.dumps(rep, indent=1))
    print("LEG_RESET_IK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
