#!/usr/bin/env python3
"""AS2-W rolling: MuJoCo closed-loop evaluation over the SAME 11-case matrix.

The Isaac Lab side already has an 11-case frozen matrix (hold, turn +-0.05/+-0.10,
straight +-0.25, arc +-0.25, brake +-0.25).  The MuJoCo side so far only covered
the 5 turn/hold cases, so the dual-simulator acceptance requirement is incomplete.
This script closes that gap with the identical case list, identical gates and
identical protocol, on CPU only (no GPU contention with the Isaac runs).

Contract (r11 reference, asserted):
    MuJoCo 3.12.0, policy period 20 ms, physics 0.000625 s x 32 substeps,
    wheel kp=0 / kd=6 / +-13.5 Nm, leg kp/kd 55/2, leg scale 0.35,
    wheel scale 16, action clip 6.0, obs 65, command bias -0.4.
    The MJCF's own timestep (0.00125) is overridden to 0.000625 and asserted.

Frozen gates (identical to Isaac):
    hold     10 s, last 3 s |vx| < 0.02 and |wz| < 0.02
    turn     12 s, last 3 s signed yaw tracking in [0.70, 1.30]
    straight 12 s, last 3 s signed vx tracking in [0.70, 1.30]
    arc      12 s, last 3 s BOTH vx and yaw tracking in [0.70, 1.30]
    brake    10 s (5 s drive then release), last 3 s |vx| < 0.02
    all: four-wheel support >= 0.95 in the last 3 s, no termination (upright<0.85
         or non-wheel contact), upright (projected gravity z) >= 0.85

Deterministic stand states are produced by three settle durations.  MuJoCo's
snapshot path is bit-reproducible only because it carries `qacc_warmstart`, so to
stay comparable with the Isaac side each case here starts from a fresh
`mj_resetData` + settle rather than from a restored snapshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mujoco

JOINTS = [f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
          for leg in ("FL", "FR", "RL", "RR")]
WN_BODY = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])

LEG_KP, LEG_KD, WHEEL_KD = 55.0, 2.0, 6.0
LEG_SCALE, WHEEL_SCALE = 0.35, 16.0
CLIP = 6.0
DT, SUBSTEPS = 0.000625, 32
STEP_DT = DT * SUBSTEPS
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
_TRACE = []
LIM = np.array([LIMIT[n.split("_")[1]] for n in JOINTS])
LOAD_N = 5.0
LOAD_Z_N = 2.0
TAIL_S = 3.0
BRAKE_HOLD_S = 5.0
STRAIGHT_TIER = 0.25
ARC_TIER = (0.25, 0.05)
SETTLE_STATES = (2.0, 3.0, 5.0)


def build_actor(state: dict) -> nn.Sequential:
    idx = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.")})
    layers = []
    for j, i in enumerate(idx):
        w, b = state[f"mlp.{i}.weight"], state[f"mlp.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data, lin.bias.data = w.clone(), b.clone()
        layers.append(lin)
        if j != len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def d4(A):
    A = np.asarray(A, float)
    return A[:, 1] + A[:, 3] - A[:, 0] - A[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-z", type=float, default=0.381395)
    # Per-frame trace dump for the "which wheel unloads and when" diagnosis.  Deliberately
    # implemented INSIDE this script rather than in a second one: an earlier standalone probe
    # duplicated the observation assembly, got the joint ordering wrong, and produced a robot
    # lying on one wheel.  Reusing this code path keeps the dynamics identical to the matrix.
    ap.add_argument("--dump-trace", type=Path, default=None,
                    help="if set, write a per-frame trace JSON (loads, wheel targets, measured dq, yaw)")
    # gyro-1 wrapper, replicated EXACTLY as in the Isaac side so the two simulators can be
    # compared under the same packaging:
    #   err   = wz_cmd - wz_meas          (body-frame yaw rate, read ONCE per policy step)
    #   mag   = clip(yaw_kp * err, +-ff_cap)
    #   tau_i = kd*(target_i - v_i) + S_i*mag       S = (-1,+1,-1,+1) for FL,FR,RL,RR
    #   the SUM is then clipped at +-13.5 Nm, same as the Isaac actuator clamp
    # Bounded differential TARGET boost.  Distinct from both the open-loop torque feedforward and
    # the gyro feedback: this adds a bounded offset to the wheel VELOCITY TARGET proportional to the
    # commanded yaw rate, so the kd law keeps driving toward it even when the policy's own
    # differential decays.  The operator's item 1 asks for exactly this against the right-turn tail
    # collapse, keeping the actor's leg feedback untouched and changing one variable only.
    #   target_wheel_i += S_i * clip(diff_gain * wz_cmd, +-diff_cap)
    ap.add_argument("--diff-gain", type=float, default=0.0,
                    help="rad/s of wheel target per rad/s of commanded yaw rate (0 = off)")
    ap.add_argument("--diff-cap", type=float, default=0.5,
                    help="bound on the added differential magnitude, rad/s")
    ap.add_argument("--yaw-kp", type=float, default=0.0,
                    help="gyro-1 yaw feedback gain, Nm per rad/s of yaw-rate error; 0 = base wrapper")
    ap.add_argument("--ff-cap", type=float, default=4.0,
                    help="gyro-1 feedforward cap, Nm (applied before the +-13.5 Nm total clamp)")
    ap.add_argument("--ff-gain", type=float, default=0.0,
                    help="gyro-1 open-loop term, Nm per rad/s of commanded wz (0 = closed-loop only)")
    ap.add_argument("--states", default=",".join(str(s) for s in SETTLE_STATES))
    ap.add_argument("--label", default="mujoco")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    states = [float(s) for s in args.states.split(",")]

    m = mujoco.MjModel.from_xml_path(str(args.xml))
    m.opt.timestep = DT
    assert abs(m.opt.timestep - DT) < 1e-12, "timestep override failed"
    assert abs(DT * SUBSTEPS - 0.02) < 1e-12, "substeps do not give 20 ms"
    d = mujoco.MjData(m)
    print("mujoco %s  dt=%.6f  substeps=%d  step_dt=%.4f"
          % (mujoco.__version__, m.opt.timestep, SUBSTEPS, STEP_DT), flush=True)

    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
                     for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    feet_bid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n): i
                for i, n in enumerate(WN_BODY)}
    kp = np.r_[np.full(12, LEG_KP), np.full(4, 0.0)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, WHEEL_KD)]
    # gyro-1 state: matches the Isaac implementation (state-free, one wz sample per 20 ms)
    S_WHEEL = np.array([-1.0, 1.0, -1.0, 1.0])      # FL, FR, RL, RR -> left-vs-right differential
    GYRO = {"kp": float(args.yaw_kp), "cap": float(args.ff_cap),
            "ffg": float(args.ff_gain), "wz_cmd": 0.0, "on": bool(args.yaw_kp != 0.0 or args.ff_gain != 0.0),
            "dg": float(args.diff_gain), "dcap": float(args.diff_cap)}
    nonwheel_bodies = [i for i in range(m.nbody) if i != 0]

    ck = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))
    print("actor layers:", [list(l.weight.shape) for l in act if isinstance(l, nn.Linear)],
          flush=True)

    def obs_of(cmd, prev, ep, zr):
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        R = quat_to_mat(d.qpos[3:7])
        qrel = q - DEF
        qrel[12:] = (qrel[12:] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        o = np.zeros(65)
        o[0:3] = np.asarray(cmd, float) - np.array([0.4, 0.0, 0.0])
        o[3:6] = R.T @ d.qvel[3:6]
        o[6:9] = R.T @ np.array([0.0, 0.0, -1.0])
        o[9] = 0.4
        o[10:26] = qrel
        o[26:42] = dq
        o[42:58] = prev
        clk = 2 * np.pi * ep * STEP_DT / 0.62
        o[58], o[59] = np.sin(clk), np.cos(clk)
        zero = bool(np.all(np.asarray(cmd, float) == 0.0))
        if zero:
            if zr[0] is None:
                zr[0] = qrel[12:].copy()
            dd = (qrel[12:] - zr[0] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            o[60:64] = np.clip(dd / 0.20, -3, 3)
        else:
            zr[0] = None
        o[64] = 1.0 if zero else 0.0
        return o

    def step_ctrl(a):
        tgt = np.r_[a[:12] * LEG_SCALE + DEF[:12], a[12:] * WHEEL_SCALE]
        if GYRO["dg"] != 0.0:
            # bounded differential added to the TARGET, sign follows the command so it reverses
            # with wz_cmd; this is what keeps the tail differential from collapsing
            mag = float(np.clip(GYRO["dg"] * GYRO["wz_cmd"], -GYRO["dcap"], GYRO["dcap"]))
            tgt[12:] = tgt[12:] + S_WHEEL * mag
        ff = np.zeros(4)
        if GYRO["on"]:
            # Read the body-frame yaw rate ONCE per policy step (free-joint qvel[3:6] is in the
            # body frame), exactly like the Isaac wrapper which samples root_ang_vel_b once per
            # policy step.  The resulting torque is held for all 32 sub-steps.
            wz_meas = float(d.qvel[5])
            mag = (GYRO["ffg"] * GYRO["wz_cmd"]
                   + GYRO["kp"] * (GYRO["wz_cmd"] - wz_meas))
            mag = float(np.clip(mag, -GYRO["cap"], GYRO["cap"]))
            ff = S_WHEEL * mag
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:]) + ff]
            d.ctrl[aid] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)
        return tgt

    def loads():
        per = np.zeros(4); perz = np.zeros(4); nonwheel = 0
        for c in range(d.ncon):
            con = d.contact[c]
            b1 = int(m.geom_bodyid[int(con.geom1)])
            b2 = int(m.geom_bodyid[int(con.geom2)])
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, c, f6)
            f = abs(float(f6[0]))
            if b1 == 0 and b2 in feet_bid:
                per[feet_bid[b2]] += f; perz[feet_bid[b2]] += f
            elif b2 == 0 and b1 in feet_bid:
                per[feet_bid[b1]] += f; perz[feet_bid[b1]] += f
            elif b1 != 0 and b2 != 0 and f > 4.0:
                nonwheel += 1
        return per, perz, nonwheel

    def reset_pose(base_z):
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, base_z)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)

    def settle(settle_s):
        reset_pose(args.base_z)
        GYRO["wz_cmd"] = 0.0
        prev = np.zeros(16); zr = [None]; ep = 0
        for _ in range(int(settle_s / STEP_DT)):
            o = obs_of([0, 0, 0], prev, ep, zr)
            with torch.no_grad():
                prev = act(torch.tensor([o], dtype=torch.float32))[0].numpy()
            step_ctrl(np.clip(prev, -CLIP, CLIP))
            ep += 1
        return float(d.qpos[2])

    def run_case(settle_s, cmd_fn, seconds):
        bz = settle(settle_s)
        prev = np.zeros(16); zr = [None]; ep = int(settle_s / STEP_DT)
        n = int(seconds / STEP_DT)
        R = {k: [] for k in ("yaw", "vx", "vy", "load", "load_z", "app", "meas", "up")}
        term = 0; nonwheel_hits = 0
        for k in range(n):
            c = cmd_fn(k * STEP_DT)
            GYRO["wz_cmd"] = float(c[2])
            o = obs_of(c, prev, ep, zr)
            with torch.no_grad():
                raw = act(torch.tensor([o], dtype=torch.float32))[0].numpy()
            prev = raw
            a = np.clip(raw, -CLIP, CLIP)
            tgt = step_ctrl(a)
            ep += 1
            R["app"].append(tgt[12:].tolist())
            R["meas"].append(d.qvel[vadr][12:].tolist())
            R["yaw"].append(float(d.qvel[5]))
            R["vx"].append(float(d.qvel[0]))
            R["vy"].append(float(d.qvel[1]))
            per, perz, nw = loads()
            R["load"].append(per.tolist()); R["load_z"].append(perz.tolist())
            nonwheel_hits += nw
            g = quat_to_mat(d.qpos[3:7]).T @ np.array([0.0, 0.0, -1.0])
            up = float(-g[2])
            R["up"].append(up)
            if up < 0.85 or nw:
                term = 1
        R["term"] = term; R["nonwheel_hits"] = nonwheel_hits; R["base_z"] = bz
        if args.dump_trace is not None:
            _TRACE.append({
                "case": None,          # labelled by the writer below
                "settle_s": settle_s,
                "cmd": [float(x) for x in cmd_fn(0.0)],
                "step_dt": STEP_DT,
                "loads_N": [[float(v) for v in row] for row in R["load"]],
                "wheel_applied": [[float(v) for v in row] for row in R["app"]],
                "wheel_meas": [[float(v) for v in row] for row in R["meas"]],
                "yaw_rate": [float(v) for v in R["yaw"]],
            })
        return R

    def judge(R, cmd, brake=False, brake_at=0.0):
        tail = max(1, int(TAIL_S / STEP_DT))
        yaw = np.asarray(R["yaw"], float); vx = np.asarray(R["vx"], float)
        L = np.asarray(R["load"], float); Lz = np.asarray(R["load_z"], float)
        sup = (L > LOAD_N).all(axis=1)
        app_d = d4(R["app"]); mea_d = d4(R["meas"])
        wz_cmd, vx_cmd = float(cmd[2]), float(cmd[0])
        out = {"cmd": [float(x) for x in cmd], "settle_base_z": R["base_z"],
               "vx_tail": float(vx[-tail:].mean()),
               "wz_tail": float(yaw[-tail:].mean()),
               "ddq_target_tail": float(np.mean(d4(np.array(R["app"]))[-tail:])),
               "ddq_measured_tail": float(np.mean(d4(np.array(R["meas"]))[-tail:])),
               "support_tail": float(sup[-tail:].mean()),
               "support_all": float(sup.mean()),
               "upright_min": float(np.min(R["up"])),
               "terminated": int(R["term"]),
               "nonwheel_hits": int(R["nonwheel_hits"]),
               "exec_tail": (float(np.abs(mea_d[-tail:]).mean()
                                   / np.abs(app_d[-tail:]).mean())
                             if np.abs(app_d[-tail:]).mean() > 1e-9 else None),
               "app_ddq_tail": float(app_d[-tail:].mean()),
               "meas_ddq_tail": float(mea_d[-tail:].mean()),
               "unloaded_frac_z2n_tail": float((Lz[-tail:] < LOAD_Z_N).mean()),
               "unloaded_frac_norm5n_tail": float((L[-tail:] < LOAD_N).mean())}
        if abs(wz_cmd) > 1e-9:
            out["yaw_track"] = out["wz_tail"] / wz_cmd
        if abs(vx_cmd) > 1e-9:
            out["vx_track"] = out["vx_tail"] / vx_cmd
        if brake:
            i0 = int(brake_at / STEP_DT); seg = vx[i0:]
            out["brake_stop_dist_m"] = float(np.abs(seg).sum() * STEP_DT)
        return out

    def gates(case, j):
        s1 = (j["terminated"] == 0 and j["support_tail"] >= 0.95
              and j["upright_min"] >= 0.85 and j["nonwheel_hits"] == 0)
        if not s1:
            return False, "stage1"
        if case == "hold":
            ok = abs(j["vx_tail"]) < 0.02 and abs(j["wz_tail"]) < 0.02
            return True, ("ok" if ok else "hold_drift")
        if case.startswith("turn"):
            t = j.get("yaw_track")
            return True, ("ok" if (t is not None and 0.70 <= t <= 1.30) else "yaw")
        if case.startswith("straight"):
            t = j.get("vx_track")
            return True, ("ok" if (t is not None and 0.70 <= t <= 1.30) else "vx")
        if case.startswith("arc"):
            ty, tv = j.get("yaw_track"), j.get("vx_track")
            ok = (ty is not None and 0.70 <= ty <= 1.30
                  and tv is not None and 0.70 <= tv <= 1.30)
            return True, ("ok" if ok else "yaw_or_vx")
        if case.startswith("brake"):
            return True, ("ok" if abs(j["vx_tail"]) < 0.02 else "brake")
        return True, "ok"

    def const(vx, wz):
        return lambda t: [vx, 0.0, wz]
    def brake_fn(vx):
        return lambda t: [vx, 0.0, 0.0] if t < BRAKE_HOLD_S else [0.0, 0.0, 0.0]

    CASES = [("hold", 10.0, const(0.0, 0.0), False),
             ("turn_left_0.05", 12.0, const(0.0, +0.05), False),
             ("turn_right_0.05", 12.0, const(0.0, -0.05), False),
             ("turn_left_0.10", 12.0, const(0.0, +0.10), False),
             ("turn_right_0.10", 12.0, const(0.0, -0.10), False),
             ("straight_fwd_0.25", 12.0, const(+STRAIGHT_TIER, 0.0), False),
             ("straight_bwd_0.25", 12.0, const(-STRAIGHT_TIER, 0.0), False),
             ("arc_left_0.25", 12.0, const(ARC_TIER[0], +ARC_TIER[1]), False),
             ("arc_right_0.25", 12.0, const(ARC_TIER[0], -ARC_TIER[1]), False),
             ("brake_fwd_0.25", 10.0, brake_fn(+STRAIGHT_TIER), True),
             ("brake_bwd_0.25", 10.0, brake_fn(-STRAIGHT_TIER), True)]

    report = {"simulator": "mujoco", "version": mujoco.__version__,
              "checkpoint": str(args.ckpt),
              "contract": {"timestep": DT, "substeps": SUBSTEPS, "step_dt": STEP_DT,
                           "wrapper": {"diff_gain": float(args.diff_gain), "diff_cap": float(args.diff_cap),
                                       "yaw_kp": float(args.yaw_kp), "ff_cap": float(args.ff_cap),
                                       "ff_gain": float(args.ff_gain),
                                       "S_vector": [float(v) for v in S_WHEEL],
                                       "wz_sample": "body-frame qvel[5], once per 20 ms policy step",
                                       "torque_clamp_nm": float(LIMIT["foot"]),
                                       "note": "ff is added to the kd law and the SUM is clamped at +-13.5 Nm, same as the Isaac actuator"},
                           "wheel_kd": WHEEL_KD, "wheel_limit": LIMIT["foot"],
                           "leg_kp": LEG_KP, "leg_kd": LEG_KD,
                           "leg_scale": LEG_SCALE, "wheel_scale": WHEEL_SCALE,
                           "clip": CLIP, "base_z_init": args.base_z,
                           "mjcf_timestep_overridden": True},
              "gates": {"hold": [0.02, 0.02], "track_band": [0.70, 1.30],
                        "support": 0.95, "upright": 0.85, "brake_vx": 0.02,
                        "tail_s": TAIL_S, "load_n": LOAD_N, "load_z_n": LOAD_Z_N},
              "states": states, "cases": {}}

    for settle_s in states:
        print()
        print("-- stand state settle=%.1fs --" % settle_s, flush=True)
        for case, secs, fn, is_brake in CASES:
            R = run_case(settle_s, fn, secs)
            j = judge(R, fn(0.0), brake=is_brake, brake_at=BRAKE_HOLD_S)
            s1, reason = gates(case, j)
            j["verdict"] = ("INVALID_STAGE1" if not s1
                            else ("PASS" if reason == "ok" else "FAIL"))
            j["reason"] = reason
            j["settle_s"] = settle_s
            report["cases"].setdefault(case, []).append(j)
            print("   %-18s %-15s sup=%.3f up=%.3f term=%d nw=%d vx=%+.4f wz=%+.4f "
                  "trk=%s/%s exec=%s"
                  % (case, j["verdict"], j["support_tail"], j["upright_min"],
                     j["terminated"], j["nonwheel_hits"], j["vx_tail"], j["wz_tail"],
                     ("%+.2f" % j["vx_track"]) if "vx_track" in j else "-",
                     ("%+.2f" % j["yaw_track"]) if "yaw_track" in j else "-",
                     ("%.3f" % j["exec_tail"]) if j["exec_tail"] else "-"), flush=True)

    print()
    print("=== MuJoCo summary (pass on ALL %d states) ===" % len(states))
    npass = 0
    for case, js in report["cases"].items():
        p = all(j["verdict"] == "PASS" for j in js)
        npass += int(p)
        yt = [j["yaw_track"] for j in js if "yaw_track" in j]
        vt = [j["vx_track"] for j in js if "vx_track" in j]
        print("  %-18s %-6s yaw=%s vx=%s"
              % (case, "PASS" if p else "FAIL",
                 ("[%+.2f,%+.2f]" % (min(yt), max(yt))) if yt else "-",
                 ("[%+.2f,%+.2f]" % (min(vt), max(vt))) if vt else "-"))
    report["pass_total"] = npass
    report["n_cases"] = len(report["cases"])
    print("  ==> %d/%d cases pass on all states" % (npass, len(report["cases"])),
          flush=True)
    (args.out / "mujoco_matrix_v2.json").write_text(json.dumps(report, indent=1, default=str))
    if args.dump_trace is not None and _TRACE:
        # Label the traces in the order the cases and their states were run: out["cases"] is an
        # ordered dict and each entry has one row per settle state.
        idx = 0
        for name, rows in report["cases"].items():
            for _row in rows:
                if idx >= len(_TRACE):
                    break
                _TRACE[idx]["case"] = name
                idx += 1
        args.dump_trace.write_text(json.dumps({"checkpoint": str(args.ckpt),
                                               "n_traces": len(_TRACE),
                                               "traces": _TRACE}, indent=1, default=str))
        print("TRACE_DUMPED", len(_TRACE), flush=True)
    print()
    print("MJ_MATRIX_V2_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
