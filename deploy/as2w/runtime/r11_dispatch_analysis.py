#!/usr/bin/env python3
"""Analyse the captured dispatch chain: commanded differential vs commanded diagonal, separately.

The operator's instruction for this round:
  * report the DIAGONAL component and the LEFT-RIGHT DIFFERENTIAL component separately, in rad/s
  * when the differential is close to zero, do NOT compute the ratio of the two - a near-zero denominator
    inflates the "diagonalisation" and makes it look like a finding when it is just noise

So every metric here is an absolute magnitude in rad/s, and the ratio is only printed when the
differential is meaningfully non-zero, with that condition stated.

Reads the wheel_dispatch.json files produced by r11_isaac_wheel_dispatch.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
DIFF_MIN = 0.05      # rad/s - below this the differential is treated as zero and no ratio is computed


def parts(v):
    """Return (differential, diagonal) in rad/s.
    differential = (FL + RR) - (FR + RL)   : left-right speed difference
    diagonal     = (FL + RL) - (FR + RR)   : the cross (diagonal) pairing
    """
    fl, fr, rl, rr = v
    diff = (fl + rr) - (fr + rl)
    diag = (fl + rl) - (fr + rr)
    return float(diff), float(diag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="directory holding <case>/wheel_dispatch.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cases = {}
    for d in sorted(a.dir.iterdir()):
        f = d / "wheel_dispatch.json"
        if not f.is_file():
            continue
        rep = json.loads(f.read_text())
        rows = rep.get("frames", [])
        if not rows:
            continue
        cmd = [r["dispatched_wheel_target"] for r in rows if r.get("dispatched_wheel_target") is not None]
        mea = [r["measured_wheel_speed"] for r in rows]
        yaw = [r["yaw_rate"] for r in rows]
        if not cmd:
            continue
        cmd = np.array(cmd); mea = np.array(mea); yaw = np.array(yaw)
        n = min(len(cmd), len(mea), len(yaw))
        cmd, mea, yaw = cmd[:n], mea[:n], yaw[:n]

        cd = np.array([parts(v)[0] for v in cmd])
        cg = np.array([parts(v)[1] for v in cmd])
        md = np.array([parts(v)[0] for v in mea])
        mg = np.array([parts(v)[1] for v in mea])

        # the settled tail (last 3 s of an 8 s turn)
        tail = max(1, n - 150)
        c_abs_diff = float(np.abs(cd[tail:]).mean())
        c_abs_diag = float(np.abs(cg[tail:]).mean())
        m_abs_diff = float(np.abs(md[tail:]).mean())
        m_abs_diag = float(np.abs(mg[tail:]).mean())

        ratio = None
        if c_abs_diff >= DIFF_MIN:
            ratio = round(c_abs_diag / c_abs_diff, 3)
        ratio_m = None
        if m_abs_diff >= DIFF_MIN:
            ratio_m = round(m_abs_diag / m_abs_diff, 3)

        cases[d.name] = {
            "frames": n,
            "commanded_mean_wheel_target_rad_s": [round(float(x), 4) for x in cmd[tail:].mean(axis=0)],
            "measured_mean_wheel_speed_rad_s": [round(float(x), 4) for x in mea[tail:].mean(axis=0)],
            "commanded_abs_differential_rad_s": round(c_abs_diff, 4),
            "commanded_abs_diagonal_rad_s": round(c_abs_diag, 4),
            "measured_abs_differential_rad_s": round(m_abs_diff, 4),
            "measured_abs_diagonal_rad_s": round(m_abs_diag, 4),
            "commanded_diag_over_diff": ratio,
            "measured_diag_over_diff": ratio_m,
            "ratio_omitted_because_diff_below_min": (ratio is None),
            "diff_min_threshold_rad_s": DIFF_MIN,
            "yaw_rate_tail_mean": round(float(yaw[tail:].mean()), 5),
            "delivery_ratio_per_wheel": [
                round(float(mea[tail:, i].mean() / cmd[tail:, i].mean()), 3) if abs(cmd[tail:, i].mean()) > 0.05 else None
                for i in range(4)],
        }

    rep = {"cases": cases, "conventions": {
        "differential": "(FL+RR) - (FR+RL), the left-right speed difference",
        "diagonal": "(FL+RL) - (FR+RR), the cross pairing",
        "units": "rad/s",
        "ratio_rule": "a ratio is reported only when the commanded differential is at least %.2f rad/s; "
                      "below that the denominator is noise and the ratio is omitted rather than printed" % DIFF_MIN}}
    (a.out / "dispatch_analysis.json").write_text(json.dumps(rep, indent=1))

    print("%-11s %9s %9s | %9s %9s | %9s %9s | %s" % (
        "case", "cmd_diff", "cmd_diag", "mea_diff", "mea_diag", "cmd_r", "mea_r", "delivery(FL,FR,RL,RR)"))
    for name, c in cases.items():
        cr = "omitted" if c["commanded_diag_over_diff"] is None else "%.2f" % c["commanded_diag_over_diff"]
        mr = "omitted" if c["measured_diag_over_diff"] is None else "%.2f" % c["measured_diag_over_diff"]
        print("%-11s %9.4f %9.4f | %9.4f %9.4f | %9s %9s | %s" % (
            name, c["commanded_abs_differential_rad_s"], c["commanded_abs_diagonal_rad_s"],
            c["measured_abs_differential_rad_s"], c["measured_abs_diagonal_rad_s"],
            cr, mr, c["delivery_ratio_per_wheel"]))
    print("\n=== 左右对比（命令侧差速）===")
    if "spin_left" in cases and "spin_right" in cases:
        L = cases["spin_left"]; R = cases["spin_right"]
        print("  spin_left  cmd_diff=%.4f  cmd_diag=%.4f  yaw=%+.5f" % (
            L["commanded_abs_differential_rad_s"], L["commanded_abs_diagonal_rad_s"], L["yaw_rate_tail_mean"]))
        print("  spin_right cmd_diff=%.4f  cmd_diag=%.4f  yaw=%+.5f" % (
            R["commanded_abs_differential_rad_s"], R["commanded_abs_diagonal_rad_s"], R["yaw_rate_tail_mean"]))
        for k in ("commanded_abs_differential_rad_s", "commanded_abs_diagonal_rad_s"):
            a, b = L[k], R[k]
            print("  %-36s right/left = %.2f" % (k, b / a if abs(a) > 1e-9 else float("nan")))
    if "arc_left" in cases and "arc_right" in cases:
        L = cases["arc_left"]; R = cases["arc_right"]
        print("  arc_left   cmd_diff=%.4f  cmd_diag=%.4f  yaw=%+.5f" % (
            L["commanded_abs_differential_rad_s"], L["commanded_abs_diagonal_rad_s"], L["yaw_rate_tail_mean"]))
        print("  arc_right  cmd_diff=%.4f  cmd_diag=%.4f  yaw=%+.5f" % (
            R["commanded_abs_differential_rad_s"], R["commanded_abs_diagonal_rad_s"], R["yaw_rate_tail_mean"]))
        for k in ("commanded_abs_differential_rad_s", "commanded_abs_diagonal_rad_s"):
            a, b = L[k], R[k]
            print("  %-36s right/left = %.2f" % (k, b / a if abs(a) > 1e-9 else float("nan")))
    print("\nDISPATCH_ANALYSIS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
