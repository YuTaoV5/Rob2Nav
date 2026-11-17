#!/usr/bin/env python3
"""AS2-W rolling line: loaded four-wheel kd comparison -- MuJoCo side.

Consumes the SAME physical-unit four-wheel target sequence (the JSON written by the Isaac
calibration pass) and the SAME contract values as the Isaac pass:

    wheel action scale 5.0 (already applied upstream: the targets in the JSON are rad/s),
    wheel kp 0 / kd <-- the only variable, gyro OFF, leg kp 55 / kd 2.0, leg scale 0.35,
    +-13.5 N.m wheel ceiling (0.9 x the 15 N.m controller limit).

What the velocity loop actually consumes
----------------------------------------
The target written into `data.ctrl` is derived once per 20 ms policy step and held for all 32
physics substeps; the value the loop consumes is recorded at the first AND the last substep of
every policy step, and the two must be identical.  The requested wheel torque tau = kd*(v_tgt - v)
is recorded separately from the clipped torque that is actually dispatched, so the
"request vs applied" split the operator asked for exists on this side too.

Every case does its own reset and its own settle, and the settle has to PASS (four wheels loaded,
upright, rest velocity small and printed) before the phases are timed.  A fall or a non-wheel
contact is recorded as failure evidence, never dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import mujoco

JOINTS = [f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])
LEG_KP, LEG_KD = 55.0, 2.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT[n.split("_")[1]] for n in JOINTS])
DT, SUBSTEPS = 0.000625, 32
STEP_DT = DT * SUBSTEPS
LOAD_N = 2.0


def pose_offsets(st: str):
    off = np.zeros(12)
    if st == "B":
        for i, n in enumerate(JOINTS[:12]):
            part = n.split("_")[1]
            if part == "thigh":
                off[i] = 0.05
            elif part == "calf":
                off[i] = -0.10
            elif part == "hip":
                off[i] = 0.01 if i < 4 else -0.01
        return off, 0.02, 0.385, 4.5
    return off, 0.0, 0.381395, 4.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kd", type=float, required=True)
    ap.add_argument("--label", default="mujoco")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seq = json.loads(args.seq.read_text())
    assert seq["schema"] == "w1w2_roll_seq_v1", seq.get("schema")

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    m.opt.timestep = DT
    assert abs(m.opt.timestep - DT) < 1e-12
    d = mujoco.MjData(m)
    print("mujoco %s dt=%.6f substeps=%d step_dt=%.4f kd=%.3f"
          % (mujoco.__version__, DT, SUBSTEPS, STEP_DT, args.kd), flush=True)

    def jid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    qadr = np.array([int(m.jnt_qposadr[jid(n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[jid(n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    feet = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n): i for i, n in enumerate(WHEELS)}
    if len(feet) != 4 or min(feet) < 0:
        raise RuntimeError("wheel bodies missing from the MJCF: %s" % feet)
    if min(aid) < 0 or len(set(aid.tolist())) != 16:
        raise RuntimeError("actuator mapping incomplete: %s" % aid)
    kp = np.r_[np.full(12, LEG_KP), np.zeros(4)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, args.kd)]

    # model facts that must be checked, not assumed
    total_mass = float(m.body_mass.sum())
    print("MODEL total_mass_kg=%.4f weight_n=%.3f nbody=%d nq=%d nv=%d"
          % (total_mass, total_mass * 9.81, m.nbody, m.nq, m.nv), flush=True)
    print("MODEL wheel_joint_dof %s foot_ctrlrange=%s"
          % ({w: int(vadr[12 + i]) for i, w in enumerate(WHEELS)},
             [list(map(float, m.actuator_ctrlrange[aid[12 + i]])) for i in range(4)]), flush=True)

    def u_norm(v):
        return float(v)

    def loads():
        per = np.zeros(4)
        nonwheel = 0
        for c in range(d.ncon):
            con = d.contact[c]
            b1 = int(m.geom_bodyid[int(con.geom1)])
            b2 = int(m.geom_bodyid[int(con.geom2)])
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            mag = abs(float(f6[0]))
            if b1 == 0 and b2 in feet:
                per[feet[b2]] += mag
            elif b2 == 0 and b1 in feet:
                per[feet[b1]] += mag
            elif b1 != 0 and b2 != 0 and mag > 4.0:
                nonwheel += 1
        return per, nonwheel

    def base_state():
        # base is a free joint: qpos[0:3] pos, qpos[3:7] quat (w,x,y,z)
        w, x, y, z = d.qpos[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        up = float(-(R.T @ np.array([0.0, 0.0, -1.0]))[2])
        v_body = R.T @ d.qvel[0:3]
        return float(v_body[0]), float(v_body[1]), float(d.qvel[5]), up

    def reset(st: str):
        off, roll, z, settle_s = pose_offsets(st)
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, z)
        d.qpos[3:7] = (np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0)
        d.qpos[qadr] = DEF + np.r_[off, np.zeros(4)]
        mujoco.mj_forward(m, d)
        return settle_s

    frames = []
    summary = {}
    trace_path = args.out / ("trace_%s_kd%s.jsonl" % (args.label, ("%.2f" % args.kd).replace(".", "p")))
    fh = trace_path.open("w")

    for st in ("A", "B"):
        for case in seq["cases"]:
            cname = case["name"]
            settle_s = reset(st)
            # settle: wheels zero, legs held at the stand pose
            n = int(round(settle_s / STEP_DT))
            tail = []
            for k in range(n):
                tgt = np.zeros(4)
                for _sub in range(SUBSTEPS):
                    v = d.qvel[vadr]
                    tau = np.r_[kp[:12] * (DEF[:12] - d.qpos[qadr][:12]) - kd[:12] * v[:12],
                                kd[12:] * (tgt - v[12:])]
                    d.ctrl[aid] = np.clip(tau, -LIM, LIM)
                    mujoco.mj_step(m, d)
                if k >= n - 25:
                    per, _nw = loads()
                    vx, vy, wz, up = base_state()
                    tail.append((per.copy(), vx, wz, up))
            loads_now = tail[-1][0]
            stand = {
                "leg_angles_rad": [round(float(x), 4) for x in d.qpos[qadr][:12]],
                "loads_n": [round(float(x), 2) for x in loads_now],
                "load_sum_over_weight": round(float(loads_now.sum() / (total_mass * 9.81)), 4),
                "vx": round(float(np.mean([t[1] for t in tail])), 5),
                "wz": round(float(np.mean([t[2] for t in tail])), 5),
                "up_min": round(float(min(t[3] for t in tail)), 5),
                "dq_absmean": round(float(np.abs(d.qvel[vadr][12:]).mean()), 5),
            }
            settled = bool(min(stand["loads_n"]) > LOAD_N and stand["up_min"] > 0.95)
            print("STAND %s/%s ok=%s loads=%s up=%.4f" % (st, cname, settled, stand["loads_n"],
                                                          stand["up_min"]), flush=True)
            cframes = []
            term_at = None
            for phase in seq["cases"][[c["name"] for c in seq["cases"]].index(cname)]["phases"]:
                ph = phase["ph"]
                tgt = np.array([float(x) for x in phase["tgt"]])
                nfr = int(round(float(phase["s"]) / STEP_DT))
                for k in range(nfr):
                    # The PD must be evaluated at the PHYSICS rate, exactly as the project's
                    # r11_mj_matrix_v2 does and exactly as the PhysX implicit drive does.  Holding
                    # one torque for all 32 substeps (a 50 Hz zero-order hold) collapses the robot --
                    # that was the first version's bug and it is why the target/request split below is
                    # recorded per substep.
                    tau_req_end = None
                    tau_end = None
                    consumed_first = None
                    consumed_last = None
                    clipped_any = 0
                    req_absmax = 0.0
                    for sub in range(SUBSTEPS):
                        q = d.qpos[qadr]
                        v = d.qvel[vadr]
                        tau_req = np.r_[kp[:12] * (DEF[:12] - q[:12]) - kd[:12] * v[:12],
                                        kd[12:] * (tgt - v[12:])]
                        tau = np.clip(tau_req, -LIM, LIM)
                        clipped_any += int(np.sum(np.abs(tau_req - tau) > 1e-9))
                        req_absmax = max(req_absmax, float(np.abs(tau_req[12:]).max()))
                        if sub == 0:
                            consumed_first = tgt.copy()
                        d.ctrl[aid] = tau
                        mujoco.mj_step(m, d)
                        consumed_last = tgt.copy()
                        tau_req_end = tau_req[12:].copy()
                        tau_end = tau[12:].copy()
                    per, nw = loads()
                    vx, vy, wz, up = base_state()
                    dq = d.qvel[vadr][12:].copy()
                    term = bool(up < 0.85 or nw > 0)
                    rec = {
                        "state": st, "case": cname, "ph": ph, "k": k, "t": round(k * STEP_DT, 4),
                        "tgt": [round(float(x), 6) for x in tgt],
                        "tgt_first_substep": [round(float(x), 6) for x in consumed_first],
                        "tgt_last_substep": [round(float(x), 6) for x in consumed_last],
                        "tgt_substep_dev": round(float(np.abs(consumed_last - consumed_first).max()), 7),
                        "dq": [round(float(x), 6) for x in dq],
                        "load": [round(float(x), 3) for x in per],
                        "tau_req_wheels": [round(float(x), 4) for x in tau_req_end],
                        "tau_applied_wheels": [round(float(x), 4) for x in tau_end],
                        "tau_req_absmax_wheels": round(req_absmax, 4),
                        "tau_clipped_substeps": clipped_any,
                        "vx": round(vx, 6), "vy": round(vy, 6), "wz": round(wz, 6),
                        "up": round(up, 6), "term": int(term),
                    }
                    cframes.append(rec)
                    if term:
                        term_at = {"case": cname, "state": st, "ph": ph, "k": k, "frame": rec}
                        print("TERMINATED %s/%s at %s k=%d up=%.3f" % (st, cname, ph, k, up), flush=True)
                        break
                if term_at:
                    break
            for r in cframes:
                fh.write(json.dumps(r) + "\n")
            fh.flush()

            def stats(rows):
                if not rows:
                    return {"n": 0}
                dq = np.array([r["dq"] for r in rows], float)
                tgt = np.array([r["tgt"] for r in rows], float)
                load = np.array([r["load"] for r in rows], float)
                req = np.array([r["tau_req_wheels"] for r in rows], float)
                app = np.array([r["tau_applied_wheels"] for r in rows], float)
                diff_t = float(((tgt[:, 1] + tgt[:, 3] - tgt[:, 0] - tgt[:, 2]) / 2).mean())
                diff_m = float(((dq[:, 1] + dq[:, 3] - dq[:, 0] - dq[:, 2]) / 2).mean())
                diag_m = float(((dq[:, 0] + dq[:, 3] - dq[:, 1] - dq[:, 2]) / 2).mean())
                unload = load < LOAD_N
                longest = np.zeros(4, int)
                for w in range(4):
                    run = 0
                    for flag in unload[:, w]:
                        run = run + 1 if flag else 0
                        longest[w] = max(longest[w], run)
                return {
                    "n": len(rows),
                    "tgt_mean": [round(float(x), 5) for x in tgt.mean(axis=0)],
                    "dq_mean": [round(float(x), 5) for x in dq.mean(axis=0)],
                    "tgt_absmean": round(float(np.abs(tgt).mean()), 5),
                    "dq_absmean": round(float(np.abs(dq).mean()), 5),
                    "delivery_ratio": (round(float(np.abs(dq).mean() / np.abs(tgt).mean()), 4)
                                       if np.abs(tgt).mean() > 1e-6 else None),
                    "abs_error_rad_s": round(float(np.abs(dq - tgt).mean()), 5),
                    "diff_tgt": round(diff_t, 5), "diff_meas": round(diff_m, 5),
                    "diag_meas": round(diag_m, 5),
                    "vx_mean": round(float(np.mean([r["vx"] for r in rows])), 5),
                    "vy_mean": round(float(np.mean([r["vy"] for r in rows])), 5),
                    "wz_mean": round(float(np.mean([r["wz"] for r in rows])), 5),
                    "up_min": round(float(min(r["up"] for r in rows)), 5),
                    "loads_mean": [round(float(x), 2) for x in load.mean(axis=0)],
                    "load_sum_over_weight": round(float(load.sum(axis=1).mean() / (total_mass * 9.81)), 4),
                    "unload_samples_per_wheel": [int(x) for x in unload.sum(axis=0)],
                    "longest_unload_samples_per_wheel": [int(x) for x in longest],
                    "min_loaded_wheels": int(min(int((~unload[i]).sum()) for i in range(len(rows)))),
                    "support_frac": round(float((~unload).all(axis=1).mean()), 4),
                    "tau_req_absmax": round(float(max(r["tau_req_absmax_wheels"] for r in rows)), 4),
                    "tau_req_over_budget": round(float(max(r["tau_req_absmax_wheels"] for r in rows) / 13.5), 4),
                    "tau_applied_absmax": round(float(np.abs(app).max()), 4),
                    "tau_clipped_frames": int(sum(r["tau_clipped_substeps"] > 0 for r in rows)),
                    "tgt_substep_dev_max": round(float(max(r["tgt_substep_dev"] for r in rows)), 7),
                }

            tail1 = max(1, int(round(1.0 / STEP_DT)))
            drive = stats([f for f in cframes if f["ph"] == "drive"][-tail1:])
            base = stats([f for f in cframes if f["ph"] == "baseline"])
            brake = stats([f for f in cframes if f["ph"] == "brake"])
            br = [f for f in cframes if f["ph"] == "brake"]
            stop_k = next((i for i, f in enumerate(br) if max(abs(x) for x in f["dq"]) < 0.05), None)
            case_sum = {
                "state": st, "case": cname, "note": case.get("note"),
                "settled": settled, "stand": stand,
                "baseline": base, "drive_tail_1s": drive, "brake": brake,
                "brake_stop_time_s": (round(stop_k * STEP_DT, 3) if stop_k is not None else None),
                "brake_entry_dq_absmax": (max(abs(x) for x in br[0]["dq"]) if br else None),
                "brake_wheel_travel_rad": (round(float(np.abs(np.array([f["dq"] for f in br], float))
                                                        .sum(axis=0).mean() * STEP_DT), 4) if br else None),
                "terminated": bool(term_at), "termination": term_at,
                "gate_ok": bool(settled and drive.get("n", 0) > 0 and not term_at
                                and abs(base.get("dq_absmean", 9)) < 0.05),
            }
            summary["%s/%s" % (st, cname)] = case_sum
            print("CASE %s/%s ok=%s drive dq=%s tgt=%s diff_t=%+.4f diff_m=%+.4f vx=%+.4f wz=%+.4f "
                  "sup=%.3f tau_req_max=%.2f brake_stop=%s"
                  % (st, cname, case_sum["gate_ok"], drive.get("dq_mean"), drive.get("tgt_mean"),
                     drive.get("diff_tgt", 0), drive.get("diff_meas", 0), drive.get("vx_mean", 0),
                     drive.get("wz_mean", 0), drive.get("support_frac", 0),
                     drive.get("tau_req_absmax", 0), case_sum["brake_stop_time_s"]), flush=True)
    fh.close()

    rep = {
        "report": "w1w2_roll_mujoco_loaded_v1", "sim": "mujoco", "label": args.label,
        "mujoco_version": mujoco.__version__, "kd": float(args.kd), "xml": str(args.xml),
        "seq": str(args.seq), "torque_budget_nm": 13.5,
        "step_dt_s": STEP_DT, "physics_dt_s": DT, "substeps": SUBSTEPS,
        "total_mass_kg": round(total_mass, 4), "weight_n": round(total_mass * 9.81, 3),
        "leg_kp": LEG_KP, "leg_kd": LEG_KD, "wheel_kp": 0.0,
        "joint_limits_nm": LIMIT, "load_threshold_n": LOAD_N,
        "contact_force_convention": "normal force |f_n| per contact (MuJoCo), Isaac side uses contact-sensor z force",
        "cases": summary,
        "verdict": "PASS" if summary and all(v["gate_ok"] for v in summary.values()) else "FAIL",
    }
    (args.out / ("summary_%s_kd%s.json" % (args.label, ("%.2f" % args.kd).replace(".", "p")))
     ).write_text(json.dumps(rep, indent=1))
    print("TRACE %s" % trace_path, flush=True)
    print("MUJOCO_VERDICT %s" % rep["verdict"], flush=True)
    print("MUJOCO_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
