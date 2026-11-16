#!/usr/bin/env python3
"""Right-side asymmetry: is the diagonal bias in the POLICY or in the WHEEL/contact path?

The operator's question (goal step 3): the policy's diagonal component is about 2.57x the left-right
differential, and turn_right / arc_right are clearly worse than the left side.  To decide which link in the
chain fails first, this compares the LEFT and RIGHT arcs of the SAME weight over the SAME protocol:

    commanded wheel target   (what the policy asked for, after the action wrapper)
        -> measured wheel speed (what the wheels actually did)
        -> body yaw rate        (what the robot actually turned into)

If the commanded differential is already unequal between the two arcs, the bias is in the POLICY.  If the
command is symmetric but the measured differential is not, the bias is in the WHEEL/CONTACT path.  If both are
symmetric but yaw is not, the bias is in how the yaw rate couples to the differential.

The per-leg decomposition is also reported, because a diagonal bias shows up as
    |left-diagonal| != |right-diagonal|  (FL+RL vs FR+RR)
rather than as a simple left/right asymmetry.

Reads the two state traces produced by r11_isaac_state_trace.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEG = ("FL", "FR", "RL", "RR")


def load(p: Path):
    d = json.loads((p / "state_trace.json").read_text())
    return d["settle"], d["turn"]


def act_target(row):
    """The wheel part of the PREVIOUS action is the closest thing to the commanded wheel target we
    recorded; the action wrapper scales it by 16 to reach rad/s."""
    return np.array(row["act"][12:], float) * 16.0


def measured(row):
    return np.array(row["dq"][12:], float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=float, default=3.0, help="seconds at the end of the turn to average")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    res = {}
    for tag, path in (("arc_left", a.left), ("arc_right", a.right)):
        settle, turn = load(path)
        n = int(a.window / 0.02)
        seg = turn[-n:]
        cmd_t = np.array([act_target(r) for r in seg])
        mea = np.array([measured(r) for r in seg])
        yaw = np.array([r["yaw"] for r in seg], float)
        loads = np.array([r.get("loads_n") or [np.nan] * 4 for r in seg], float)
        # left/right differential and the two diagonals
        cmd_lr = (cmd_t[:, 0] + cmd_t[:, 3]) - (cmd_t[:, 1] + cmd_t[:, 2])   # (FL+RR) - (FR+RL)
        mea_lr = (mea[:, 0] + mea[:, 3]) - (mea[:, 1] + mea[:, 2])
        cmd_diag = np.abs((cmd_t[:, 0] + cmd_t[:, 2]) - (cmd_t[:, 1] + cmd_t[:, 3]))
        mea_diag = np.abs((mea[:, 0] + mea[:, 2]) - (mea[:, 1] + mea[:, 3]))
        res[tag] = {
            "commanded_wheel_target_rad_s": [round(float(x), 4) for x in cmd_t.mean(axis=0)],
            "measured_wheel_speed_rad_s": [round(float(x), 4) for x in mea.mean(axis=0)],
            "mean_wheel_load_N": [None if np.isnan(x) else round(float(x), 1) for x in loads.mean(axis=0)],
            "commanded_differential_FL_RR_minus_FR_RL": round(float(cmd_lr.mean()), 4),
            "measured_differential_FL_RR_minus_FR_RL": round(float(mea_lr.mean()), 4),
            "commanded_diagonal_magnitude": round(float(cmd_diag.mean()), 4),
            "measured_diagonal_magnitude": round(float(mea_diag.mean()), 4),
            "diag_to_lr_ratio_commanded": round(float(cmd_diag.mean() / max(abs(cmd_lr.mean()), 1e-9)), 3),
            "diag_to_lr_ratio_measured": round(float(mea_diag.mean() / max(abs(mea_lr.mean()), 1e-9)), 3),
            "yaw_rate_mean": round(float(yaw.mean()), 4),
            "yaw_rate_tail_mean": round(float(yaw[-50:].mean()), 4),
            "exec_ratio_measured_over_commanded": [
                round(float(mea[:, i].mean() / cmd_t[:, i].mean()), 3) if abs(cmd_t[:, i].mean()) > 1e-6 else None
                for i in range(4)],
        }

    L, R = res["arc_left"], res["arc_right"]
    verdicts = {}
    # 1) is the commanded differential already asymmetric?
    c_l, c_r = L["commanded_differential_FL_RR_minus_FR_RL"], R["commanded_differential_FL_RR_minus_FR_RL"]
    verdicts["commanded_differential_is_asymmetric"] = {
        "left": c_l, "right": c_r,
        "ratio_right_over_left": round(float(c_r / c_l), 3) if abs(c_l) > 1e-9 else None,
        "finding": "POLICY" if (abs(c_l) > 1e-6 and abs(c_r) > 1e-6 and
                                abs(c_l + c_r) / max(abs(c_l), abs(c_r)) > 0.5) else "NOT primarily the policy"}
    # 2) is the measured differential asymmetric given the command?
    m_l, m_r = L["measured_differential_FL_RR_minus_FR_RL"], R["measured_differential_FL_RR_minus_FR_RL"]
    verdicts["measured_differential_is_asymmetric"] = {
        "left": m_l, "right": m_r,
        "ratio_right_over_left": round(float(m_r / m_l), 3) if abs(m_l) > 1e-9 else None}
    # 3) diagonal magnitude: the operator's "diagonal ~2.57x the differential" check
    verdicts["diagonal_check"] = {
        "commanded_diag_over_lr_left": L["diag_to_lr_ratio_commanded"],
        "commanded_diag_over_lr_right": R["diag_to_lr_ratio_commanded"],
        "measured_diag_over_lr_left": L["diag_to_lr_ratio_measured"],
        "measured_diag_over_lr_right": R["diag_to_lr_ratio_measured"],
        "operator_reference": 2.57,
        "note": "the ratio is only meaningful when the differential is non-trivial"}
    # 4) does yaw follow the differential?
    verdicts["yaw_follows_differential"] = {
        "left_yaw": L["yaw_rate_tail_mean"], "right_yaw": R["yaw_rate_tail_mean"],
        "left_measured_diff": m_l, "right_measured_diff": m_r,
        "left_yaw_per_measured_diff": round(float(L["yaw_rate_tail_mean"] / m_l), 4) if abs(m_l) > 1e-6 else None,
        "right_yaw_per_measured_diff": round(float(R["yaw_rate_tail_mean"] / m_r), 4) if abs(m_r) > 1e-6 else None}
    # 5) execution ratio per wheel: is a specific wheel not delivering?
    verdicts["per_wheel_execution"] = {
        "left": dict(zip(LEG, L["exec_ratio_measured_over_commanded"])),
        "right": dict(zip(LEG, R["exec_ratio_measured_over_commanded"]))}

    rep = {"window_s": a.window, "per_case": res, "verdicts": verdicts}
    (a.out / "asymmetry.json").write_text(json.dumps(rep, indent=1))

    print("=== commanded wheel target (rad/s) ===")
    for t in ("arc_left", "arc_right"):
        print("  %-10s %s   diff=%+.4f  diag=%.4f" % (
            t, res[t]["commanded_wheel_target_rad_s"],
            res[t]["commanded_differential_FL_RR_minus_FR_RL"], res[t]["commanded_diagonal_magnitude"]))
    print("\n=== measured wheel speed (rad/s) ===")
    for t in ("arc_left", "arc_right"):
        print("  %-10s %s   diff=%+.4f  diag=%.4f  yaw=%+.4f" % (
            t, res[t]["measured_wheel_speed_rad_s"],
            res[t]["measured_differential_FL_RR_minus_FR_RL"], res[t]["measured_diagonal_magnitude"],
            res[t]["yaw_rate_tail_mean"]))
    print("\n=== per-wheel execution ratio (measured/commanded) ===")
    for t in ("arc_left", "arc_right"):
        print("  %-10s %s" % (t, verdicts["per_wheel_execution"]["left" if t == "arc_left" else "right"]))
    print("\n=== commanded diff asymmetry ===", verdicts["commanded_differential_is_asymmetric"])
    print("=== diagonal check ===", verdicts["diagonal_check"])
    print("=== yaw vs measured diff ===", verdicts["yaw_follows_differential"])
    print("ASYMMETRY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
