#!/usr/bin/env python3
"""Self-test for the per-wheel load channel before it is used for anything.

The operator's requirement: run once standing and once with ONE wheel suspended, confirm the four wheels
are aligned to FL/FR/RL/RR, that the suspended wheel's reading drops, and that the refresh timing is right.
Without this, a misaligned or stale load channel would produce data that cannot localise anything.

Cases:
    A  standing            -> all four wheels carry a share of the weight, sum close to m*g
    B  robot lifted 0.5 m  -> ALL four read zero (proves the channel is not just echoing body weight)
    C  each leg raised in turn -> ONLY that wheel reads zero while the other three increase, and the raised
       wheel must be the one named, which is what proves the FL/FR/RL/RR alignment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

DT = 0.000625
SUBSTEPS = 32
KP_LEG, LEG_KD = 55.0, 2.0
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
# Small offsets: the earlier +-0.35/1.4/1.2 rad class tipped the robot over, which invalidated the test.
LEG_GAIN = {"hip": np.array([-0.12, 0.12, -0.12, 0.12]),
            "thigh": np.array([-0.15, -0.15, -0.15, -0.15]),
            "calf": np.array([0.15, 0.15, 0.15, 0.15])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="checkpoint for the POLICY that settles the robot - holding DEF rigidly collapses it")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch, torch.nn as nn

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

    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, KP_LEG), np.full(4, 0.0)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, 0.0)]
    LIMW = 13.5
    wb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{L}_foot") for L in LEGS]
    total_mass = float(m.body_mass.sum())
    weight = total_mass * 9.81

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
        return [round(float(x), 2) for x in out]

    def diag(tag):
        print("    [diag %s] z=%.4f ncon=%d loads=%s" % (tag, float(d.qpos[2]), int(d.ncon), loads()))

    prev = np.zeros(16); zr = [None]; ep = 0

    def obs_of(prev, ep, zero_ref):
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        w, x, y, z = d.qpos[3:7]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        qrel = q - DEF
        qrel[12:] = (qrel[12:] + 2*np.pi) % (4*np.pi) - 2*np.pi
        o = np.zeros(65)
        o[0:3] = np.array([0.0, 0.0, 0.0]) - np.array([0.4, 0.0, 0.0])
        o[3:6] = R.T @ d.qvel[3:6]
        o[6:9] = R.T @ np.array([0.0, 0.0, -1.0])
        o[9] = 0.4
        o[10:26] = qrel
        o[26:42] = dq
        o[42:58] = prev
        clk = 2*np.pi*ep*0.02/0.62
        o[58], o[59] = np.sin(clk), np.cos(clk)
        if zero_ref[0] is None:
            zero_ref[0] = qrel[12:].copy()
        dd = (qrel[12:] - zero_ref[0] + 2*np.pi) % (4*np.pi) - 2*np.pi
        o[60:64] = np.clip(dd/0.20, -3, 3)
        o[64] = 1.0
        return o

    def settle(n=150):
        nonlocal prev, ep
        for _ in range(n):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev[:] = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -6, 6)
            leg_t = prev[:12] * 0.35 + DEF[:12]
            tgtw = prev[12:] * 16.0
            for _ in range(SUBSTEPS):
                q = d.qpos[qadr]; v = d.qvel[vadr]
                tau = np.r_[kp[:12]*(leg_t - q[:12]) - kd[:12]*v[:12],
                            6.0*(tgtw - v[12:])]
                d.ctrl[aid] = np.clip(tau, -100, 100)
                mujoco.mj_step(m, d)
            ep += 1

    def fresh(z=0.381395, lift_leg=None, lift_after=True):
        nonlocal prev, ep
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, z)
        d.qpos[3:7] = (1.0, 0, 0, 0)
        d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)
        prev = np.zeros(16); ep = 0; zr[0] = None
        diag("after reset")
        settle()
        diag("after settle")
        if lift_leg is not None and lift_after:
            i = LEGS.index(lift_leg)
            base = d.qpos[qadr][:12].copy()
            for k, jt in enumerate(("hip", "thigh", "calf")):
                idx = k * 4 + i      # contract layout is part-major
                base[idx] = base[idx] + LEG_GAIN[jt][i]
            # hold this offset for a short while so the load shift is observable
            for _ in range(120):
                leg_t = base
                tgtw = np.zeros(4)
                for _ in range(SUBSTEPS):
                    q = d.qpos[qadr]; v = d.qvel[vadr]
                    tau = np.r_[kp[:12]*(leg_t - q[:12]) - kd[:12]*v[:12], 6.0*(tgtw - v[12:])]
                    d.ctrl[aid] = np.clip(tau, -100, 100)
                    mujoco.mj_step(m, d)
            diag("after lift %s" % lift_leg)

    results = {}
    fresh()
    results["A_standing"] = {"loads_N": loads(), "sum_N": round(sum(loads()), 1),
                             "expected_weight_N": round(weight, 1)}
    print("A standing            loads=%s sum=%.1f (weight %.1f)" % (results["A_standing"]["loads_N"],
                                                                  results["A_standing"]["sum_N"], weight))
    fresh(z=0.9)
    results["B_lifted"] = {"loads_N": loads()}
    print("B lifted 0.9 m        loads=%s" % results["B_lifted"]["loads_N"])

    per_leg = {}
    for leg in LEGS:
        fresh(lift_leg=leg)
        lo = loads()
        per_leg[leg] = {"loads_N": lo}
        print("C raise %-2s            loads=%s" % (leg, lo))
    results["C_per_leg_raised"] = per_leg

    # verdict
    ok = True
    reasons = []
    st = results["A_standing"]["loads_N"]
    if not all(x > 20 for x in st):
        ok = False; reasons.append("standing: not all four wheels loaded (%s)" % st)
    if abs(sum(st) - weight) > 0.35 * weight:
        ok = False; reasons.append("standing: load sum %.1f far from weight %.1f" % (sum(st), weight))
    if not all(x < 1.0 for x in results["B_lifted"]["loads_N"]):
        ok = False; reasons.append("lifted: wheels still report load (%s)" % results["B_lifted"]["loads_N"])
    for i, leg in enumerate(LEGS):
        lo = per_leg[leg]["loads_N"]
        others = [x for j, x in enumerate(lo) if j != i]
        if not (lo[i] < 1.0):
            ok = False; reasons.append("raise %s: that wheel still reads %.2f" % (leg, lo[i]))
        if not all(x > 0 for x in others):
            ok = False; reasons.append("raise %s: another wheel reads zero (%s)" % (leg, others))
    results["verdict"] = "PASS" if ok else "FAIL"
    results["reasons"] = reasons
    (a.out / "loads_selftest.json").write_text(json.dumps(results, indent=1))
    print()
    if ok:
        print("LOAD_CHANNEL_SELFTEST_PASS  (4 wheels aligned, suspended wheel drops, refresh live)")
    else:
        print("LOAD_CHANNEL_SELFTEST_FAIL")
        for r in reasons:
            print("  -", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
