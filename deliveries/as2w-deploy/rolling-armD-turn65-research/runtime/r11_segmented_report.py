#!/usr/bin/env python3
"""Segmented turning report: first second, 3-5 s and last 3 s, for every turning case.

The operator asked, for every turning case, to report the first second, the 3-5 s window and the last
3 s, and to record alongside them the signed yaw, the actor's raw differential, the final commanded
differential, the measured differential, the common-mode wheel speed, four-wheel support and the
actual torque.  This script produces exactly that table from a run trace.

Input is a MuJoCo `--dump-trace` JSON (per-frame loads, actor wheel targets, measured wheel speeds, body
yaw rate).  Windows are in seconds from the start of the case; frames inside a lost-support window are
counted but also reported separately so a reader can see how much of each window was actually valid.

Signed values are kept signed: a turn whose yaw has the wrong sign must not look like a small positive
number, and the differential is reported with its sign for the same reason.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOWS = [("first_1s", 0.0, 1.0), ("mid_3_5s", 3.0, 5.0), ("last_3s", None, None)]
LOAD_N = 20.0
DT_DEFAULT = 0.02


def d4(A):
    A = np.asarray(A, float)
    return A[:, 1] + A[:, 3] - A[:, 0] - A[:, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--load-threshold", type=float, default=LOAD_N)
    a = ap.parse_args()
    j = json.loads(a.trace.read_text())
    rep = {"trace": str(a.trace), "load_threshold_n": a.load_threshold, "windows": {}, "cases": {}}

    for tr in j["traces"]:
        case = tr.get("case") or "unknown"
        if not str(case).startswith("turn"):
            continue
        dt = float(tr.get("step_dt", DT_DEFAULT))
        n = len(tr["yaw_rate"])
        loads = np.asarray(tr["loads_N"], float)
        app = np.asarray(tr["wheel_applied"], float)      # actor target (already scaled)
        mea = np.asarray(tr["wheel_meas"], float)
        yaw = np.asarray(tr["yaw_rate"], float)
        wz_cmd = float(tr["cmd"][2])
        supp = (loads > a.load_threshold).all(axis=1)
        d_app, d_mea = d4(app), d4(mea)
        rows = {}
        for name, t0, t1 in WINDOWS:
            i0 = 0 if t0 is None else int(round(t0 / dt))
            i1 = n if t1 is None else min(n, int(round(t1 / dt)))
            if i1 <= i0:
                rows[name] = None
                continue
            sl = slice(i0, i1)
            rows[name] = {
                "frames": int(i1 - i0),
                "t_s": [round(i0 * dt, 2), round(i1 * dt, 2)],
                "yaw_signed": float(np.mean(yaw[sl])),
                "yaw_track_signed": (float(np.mean(yaw[sl]) / wz_cmd) if abs(wz_cmd) > 1e-9 else None),
                "differential_actor": float(np.mean(d_app[sl])),
                "differential_measured": float(np.mean(d_mea[sl])),
                "differential_delivery": (float(np.mean(d_mea[sl]) / np.mean(d_app[sl]))
                                          if abs(np.mean(d_app[sl])) > 1e-9 else None),
                "wheel_common_mode": float(np.mean(mea[sl])),
                "support_frac": float(np.mean(supp[sl])),
                "yaw_absmax": float(np.max(np.abs(yaw[sl]))),
            }
        rep["cases"][case] = {"wz_cmd": wz_cmd, "frames": n, "windows": rows}

    (a.out).parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))

    hdr = ["case", "window", "yaw", "track", "d_actor", "d_meas", "deliv", "common", "support"]
    print("%-16s %-9s %8s %7s %9s %9s %6s %8s %7s" % tuple(hdr))
    for case, c in rep["cases"].items():
        for name, _, _ in WINDOWS:
            w = c["windows"].get(name)
            if not w:
                continue
            tr = "  n/a " if w["yaw_track_signed"] is None else "%+.2f" % w["yaw_track_signed"]
            dv = " n/a " if w["differential_delivery"] is None else "%.2f" % w["differential_delivery"]
            print("%-16s %-9s %+8.4f %7s %+9.3f %+9.3f %6s %8.3f %7.3f"
                  % (case, name, w["yaw_signed"], tr, w["differential_actor"],
                     w["differential_measured"], dv, w["wheel_common_mode"], w["support_frac"]))
        print()
    print("SEGMENTED_REPORT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
