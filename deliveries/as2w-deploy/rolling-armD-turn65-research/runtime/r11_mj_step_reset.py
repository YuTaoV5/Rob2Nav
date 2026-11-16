#!/usr/bin/env python3
"""Does a net-zero step cycle break the post-turn standing-posture floor?

Previous measurement: after a turn, simply commanding the legs back to the fixed stand posture converges
the leg error to a plateau of 0.193-0.197 rad within half a second and then stops improving over 6 s.  That
makes the residual a loaded PD equilibrium, so the only way to go lower is to unload each leg in turn -
which is exactly the operator's step-to-reset idea.

This measures whether it works.  Sequence on one continuous trajectory:

    1. settle, then run a turn (wz = +-0.05) for 12 s with the policy
    2. hold fixed stand for 4 s and record the plateau (the baseline)
    3. step reset: drive the analytic stepping gait for N cycles with a net-zero stride, so the robot
       steps in place - each foot lifts and re-plants
    4. hold fixed stand again for 4 s and record the leg error

The gait is `as2w_stepping_control.calibrated_sequence`, whose stride is 2*(swing_forward - return_to_rear) - 1
and therefore integrates to zero displacement over one cycle; the command only has to exceed 0.015 to
activate the swing, so a small forward command gives in-place stepping rather than travel.

CPU-only, so it does not contend with training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/code")
from as2w_stepping_control import calibrated_sequence  # noqa: E402

DT = 0.000625
SUBSTEPS = 32
STEP_DT = DT * SUBSTEPS
LEG_KD, WHEEL_KD = 2.0, 6.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE = 0.35, 16.0
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--hold-s", type=float, default=4.0)
    ap.add_argument("--step-command", type=float, default=0.12, help="vx used to activate the gait")
    ap.add_argument("--gait-period-s", type=float, default=0.8)
    ap.add_argument("--step-cycles", type=float, default=5.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch.nn as nn
    import mujoco

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

    m = mujoco.MjModel.from_xml_path(str(a.xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, 55.0), np.full(4, 0.0)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, WHEEL_KD)]
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
        act16 = np.clip(act16, -6.0, 6.0)
        tgt = np.r_[act16[:12] * LEG_SCALE + DEF[:12], act16[12:] * WHEEL_SCALE]
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:])]
            d.ctrl[aid] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)

    def leg_err():
        return float(np.abs(d.qpos[qadr][:12] - DEF[:12]).max())

    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    prev = np.zeros(16); zr = [None]; ep = 0
    for _ in range(int(a.settle_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -6.0, 6.0)
        step_ctrl(prev); ep += 1

    cmd["v"] = np.array([0.0, 0.0, a.wz])
    for _ in range(int(a.turn_s / STEP_DT)):
        o = obs_of(prev, ep, zr)
        with torch.no_grad():
            prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -6.0, 6.0)
        step_ctrl(prev); ep += 1
    err_turn_end = leg_err()

    # baseline: hold fixed stand
    for _ in range(int(a.hold_s / STEP_DT)):
        step_ctrl(np.zeros(16))
    err_hold_baseline = leg_err()

    # step reset: net-zero stride gait, in place
    n_steps = int(a.step_cycles * a.gait_period_s / STEP_DT)
    cmdseq = torch.tensor([[a.step_command, 0.0, 0.0]], dtype=torch.float32)
    gait_acts = []
    for k in range(n_steps):
        clock = torch.tensor([(k * STEP_DT / a.gait_period_s) % 1.0], dtype=torch.float32)
        seq = calibrated_sequence(cmdseq, clock)[0].detach().numpy()
        gait_acts.append(seq)
        step_ctrl(np.r_[seq, np.zeros(4)])
    err_after_steps = leg_err()

    # hold again
    for _ in range(int(a.hold_s / STEP_DT)):
        step_ctrl(np.zeros(16))
    err_final = leg_err()

    rep = {"checkpoint": str(a.ckpt), "wz": a.wz,
           "gait": {"command": a.step_command, "period_s": a.gait_period_s, "cycles": a.step_cycles,
                    "step_frames": n_steps,
                    "act_absmax": float(np.abs(np.asarray(gait_acts)).max())},
           "leg_err_max_rad": {
               "after_turn": err_turn_end,
               "hold_baseline": err_hold_baseline,
               "after_step_cycle": err_after_steps,
               "after_final_hold": err_final},
           "delta_from_step": err_after_steps - err_hold_baseline,
           "delta_from_final": err_final - err_hold_baseline}
    (a.out / "step_reset.json").write_text(json.dumps(rep, indent=1))
    print("wz=%+.2f  turn_end=%.3f  hold_baseline=%.3f  after_steps=%.3f  final=%.3f  "
          "d_step=%+.3f  d_final=%+.3f  gait_act_absmax=%.3f"
          % (a.wz, err_turn_end, err_hold_baseline, err_after_steps, err_final,
             rep["delta_from_step"], rep["delta_from_final"], rep["gait"]["act_absmax"]))
    print("STEP_RESET_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
