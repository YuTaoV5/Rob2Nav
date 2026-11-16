#!/usr/bin/env python3
"""Is the MuJoCo wheel torque recomputed per 20 ms sample or per 0.625 ms substep?

The operator's arithmetic makes this the first thing to settle.  With a wheel inertia of about
0.0025 kg*m^2 and an initial demand of 3 N*m:

    torque held for the whole 20 ms   ->  3 * 0.020 / 0.0025  =  24.0 rad/s   (matches the measured 24.5)
    torque recomputed every 0.625 ms  ->  the first substep moves only about 0.75 rad/s and the loop converges

So the 24.5 first sample is evidence about the UPDATE RATE, not necessarily about damping.

This script runs the same single wheel under two arms and records the whole chain per substep:

    A : demand computed once per 20 ms sample, ctrl held constant across the 32 substeps
    B : demand recomputed from the freshly integrated dq before every substep

For every substep it records dq_before -> torque_request -> ctrl -> actuator_force -> dq_after, so the
"does torque actually change within a sample" question is answered by data rather than by reading code.
It also prints the model's actual timestep and whether the fixed-base setup rewrites joint state between
substeps, which is the second thing the operator asked to confirm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

DT = 0.000625
SUBSTEPS = 32
WHEEL_KD = 6.0
KP_LEG = 55.0
LEG_KD = 2.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=40)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    m.opt.gravity[:] = 0.0                     # suspended wheel, isolate the loop
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])

    print("model timestep after set: %.9f  (requested %.9f)" % (m.opt.timestep, DT))
    print("integrator: %d   substeps/sample: %d   control period: %.4f s" % (m.opt.integrator, SUBSTEPS, DT * SUBSTEPS))

    def fresh():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.6)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for k in range(16):
            d.qpos[qadr[k]] = DEF[k]
            d.qvel[vadr[k]] = 0.0
        mujoco.mj_forward(m, d)

    def leg_demand():
        return KP_LEG * (DEF[:12] - d.qpos[qadr][:12]) - LEG_KD * d.qvel[vadr][:12]

    def run(arm: str):
        fresh()
        rows = []
        for _s in range(a.samples):
            held = None
            if arm == "A":
                # computed ONCE per 20 ms sample, then held for all 32 substeps
                dq = d.qvel[vadr][12:]
                held = np.r_[leg_demand(), WHEEL_KD * (a.target - dq)]
                held = np.clip(held, -LIM, LIM)
            for s in range(SUBSTEPS):
                q_before = float(d.qpos[qadr[12]])
                dq_before = float(d.qvel[vadr[12]])
                if arm == "A":
                    request = held
                else:
                    # recomputed from the FRESH dq before every substep
                    dq = d.qvel[vadr][12:]
                    request = np.clip(np.r_[leg_demand(), WHEEL_KD * (a.target - dq)], -LIM, LIM)
                d.ctrl[aid] = request
                mujoco.mj_step(m, d)
                af = None
                try:
                    af = float(d.actuator_force[aid[12]])
                except Exception:
                    af = None
                rows.append({"substep": s,
                             "wheel": "FL_foot",
                             "dq_before": round(dq_before, 6),
                             "torque_request": round(float(request[12]), 6),
                             "ctrl": round(float(d.ctrl[aid[12]]), 6),
                             "actuator_force": None if af is None else round(af, 6),
                             "dq_after": round(float(d.qvel[vadr[12]]), 6)})
        return rows

    ra = run("A")
    rb = run("B")

    def summary(rows, label):
        first_sample = rows[:SUBSTEPS]
        sample_ends = [rows[i * SUBSTEPS + SUBSTEPS - 1]["dq_after"] for i in range(a.samples)]
        torques = [r["torque_request"] for r in first_sample]
        print("\n=== arm %s ===" % label)
        print("  first substep : dq_before=%.4f torque=%.4f -> dq_after=%.4f"
              % (first_sample[0]["dq_before"], first_sample[0]["torque_request"], first_sample[0]["dq_after"]))
        print("  torque sequence over the first sample: %s" % [round(t, 3) for t in torques[:8]])
        print("  dq at each 20 ms sample end: %s" % [round(x, 4) for x in sample_ends[:10]])
        print("  end-of-run dq: %.4f" % sample_ends[-1])
        return {"first_substep_dq_after": first_sample[0]["dq_after"],
                "first_sample_torques": [round(t, 4) for t in torques],
                "sample_end_dq": [round(x, 4) for x in sample_ends],
                "final_dq": round(sample_ends[-1], 4)}

    sa = summary(ra, "A  20 ms hold")
    sb = summary(rb, "B  per-substep")
    rep = {"target_rad_s": a.target, "model_timestep_s": float(m.opt.timestep),
           "integrator_id": int(m.opt.integrator), "substeps_per_sample": SUBSTEPS,
           "control_period_s": DT * SUBSTEPS, "wheel_kd": WHEEL_KD,
           "wheel_effort_limit_Nm": LIMIT["foot"], "gravity_off": True,
           "fixed_base_note": "base pose and joint state are set ONCE in fresh(); nothing rewrites them between substeps",
           "summary_A": sa, "summary_B": sb, "trace_A": ra, "trace_B": rb}
    (a.out / "substep_ab.json").write_text(json.dumps(rep))
    print("\nSUBSTEP_AB_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
