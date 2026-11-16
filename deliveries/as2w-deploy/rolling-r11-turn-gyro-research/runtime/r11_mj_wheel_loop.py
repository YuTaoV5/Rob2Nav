#!/usr/bin/env python3
"""Wheel-loop isolation: fixed base, wheels in the air, no policy, no gyro, no delay.

Operator step 1.  Both simulators get the identical treatment so the only thing under test is the wheel
velocity loop itself:

    * the robot is lifted clear of the ground and the base is pinned (no falling, no contact)
    * all 12 leg joints are held at one fixed pose
    * NO actor, NO gyro compensation, NO action delay, NO action ramp, NO command gate
    * each wheel is given the same target sequence: 0 -> +0.5 -> 0 -> -0.5 rad/s, held long enough to settle

Recorded per sample, for each wheel: the final target, the measured velocity, and the torque actually
applied BEFORE and AFTER the effort limit, so a difference can be attributed to the loop rather than to
the limiter.  Also recorded: the physics step, the number of physics sub-steps per control sample, and the
control period - the three quantities the operator asked to align.

MuJoCo version.  The Isaac twin is r11_isaac_wheel_loop.py with the same schema and the same sequence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

DT = 0.000625
SUBSTEPS = 32
STEP_DT = DT * SUBSTEPS
WHEEL_KD = 6.0
KP_LEG = 55.0
LEG_KD = 2.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
SEQUENCE = [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0), (-0.5, 2.0), (0.0, 1.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lift-z", type=float, default=0.6, help="how far to raise the base so wheels are clear")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    m.opt.gravity[:] = 0.0          # fixed base: no gravity load on the loop under test
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, a.lift_z)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for k in range(16):
            d.qpos[qadr[k]] = DEF[k]
            d.qvel[vadr[k]] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    trace = []
    reset()
    for target_vel, hold_s in SEQUENCE:
        n = int(hold_s / STEP_DT)
        for _ in range(n):
            # raw demand computed exactly as the contract states, then limited
            demand_leg = KP_LEG * (DEF[:12] - d.qpos[qadr][:12]) - LEG_KD * d.qvel[vadr][:12]
            demand_wheel = WHEEL_KD * (target_vel - d.qvel[vadr][12:])
            demand = np.r_[demand_leg, demand_wheel]
            applied = np.clip(demand, -LIM, LIM)
            for s in range(SUBSTEPS):
                d.ctrl[aid] = applied
                mujoco.mj_step(m, d)
            trace.append({"target": target_vel,
                          "q_meas": [round(float(x), 6) for x in d.qpos[qadr][12:]],
                          "dq_meas": [round(float(x), 6) for x in d.qvel[vadr][12:]],
                          "tau_demand": [round(float(x), 6) for x in demand[12:]],
                          "tau_applied": [round(float(x), 6) for x in applied[12:]],
                          "tau_clipped": bool(np.any(np.abs(demand[12:]) > LIM[12:]))})

    tail = {}
    for target_vel in (0.5, -0.5):
        rows = [r for r in trace if r["target"] == target_vel]
        rows = rows[len(rows) // 2:]          # second half of each hold, i.e. settled
        tail[str(target_vel)] = {
            "dq_mean": [round(float(np.mean([r["dq_meas"][i] for r in rows])), 4) for i in range(4)],
            "tau_applied_mean": [round(float(np.mean([r["tau_applied"][i] for r in rows])), 4) for i in range(4)]}
    rep = {"sim": "mujoco", "physics_dt_s": DT, "substeps_per_sample": SUBSTEPS,
           "control_period_s": STEP_DT, "sequence": [[t, h] for t, h in SEQUENCE],
           "gravity_off": True, "base_lift_z_m": a.lift_z,
           "wheel_kd": WHEEL_KD, "wheel_effort_limit_Nm": LIMIT["foot"],
           "settled": tail, "trace": trace}
    (a.out / "wheel_loop.json").write_text(json.dumps(rep))
    print("physics_dt=%.6f substeps=%d control_period=%.4f" % (DT, SUBSTEPS, STEP_DT))
    print("settled wheel speed at target +0.5: %s" % tail["0.5"]["dq_mean"])
    print("settled wheel speed at target -0.5: %s" % tail["-0.5"]["dq_mean"])
    print("settled applied torque at +0.5: %s" % tail["0.5"]["tau_applied_mean"])
    print("MJ_WHEEL_LOOP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
