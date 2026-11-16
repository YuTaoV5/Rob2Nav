#!/usr/bin/env python3
"""Compare the MuJoCo and Isaac closed-loop state traces and report WHICH FIELD diverges first.

Both traces were recorded by r11_mj_state_trace.py and r11_isaac_state_trace.py with the same schema and
the same protocol: settle 3 s with a zero command, then 12 s of a pure-turn command, both starting from
the same fixed stand.  So the two runs are aligned at the settle/turn boundary by construction.

For each field the divergence is the per-step absolute difference, normalised by a scale that reflects a
meaningful magnitude for that field (so wheel speeds and quaternions are comparable).  The report gives,
per field, the first frame at which it exceeds the threshold and the magnitude reached.  The EARLIEST
field is the mechanism: it is what the two simulators disagree about before anything else.

Prints a ranked order plus the exact numbers, and writes the whole per-frame divergence table so the
claim can be re-checked without re-running either simulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = [
    ("obs_cmd", slice(0, 3), 0.30, "command term of the observation"),
    ("obs_angvel_b", slice(3, 6), 0.30, "body angular velocity in the observation"),
    ("obs_gravity_b", slice(6, 9), 0.10, "projected gravity in the observation"),
    ("obs_qrel_leg", slice(10, 22), 0.20, "leg joint positions relative to default"),
    ("obs_qrel_wheel", slice(22, 26), 0.50, "wheel joint positions relative to default"),
    ("obs_dq_leg", slice(26, 38), 0.50, "leg joint velocities"),
    ("obs_dq_wheel", slice(38, 42), 0.50, "wheel joint velocities"),
    ("obs_prevact_leg", slice(42, 54), 0.30, "previous leg action"),
    ("obs_prevact_wheel", slice(54, 58), 0.50, "previous wheel action"),
    ("obs_phase", slice(58, 60), 0.10, "gait clock"),
    ("obs_zerodelta", slice(60, 65), 0.50, "zero-command wheel delta and flag"),
    ("act_leg", None, 0.30, "leg action output"),
    ("act_wheel", None, 0.50, "wheel action output"),
    ("q_leg", None, 0.15, "measured leg joint position"),
    ("dq_wheel_meas", None, 0.50, "measured wheel velocity"),
    ("angvel_world", None, 0.20, "root angular velocity, world frame"),
    ("yaw", None, 0.10, "yaw rate, the scoring quantity"),
]


def vec(row, name):
    if name == "act_leg":
        return np.array(row["act"][:12])
    if name == "act_wheel":
        return np.array(row["act"][12:])
    if name == "q_leg":
        return np.array(row["q"][:12])
    if name == "dq_wheel_meas":
        return np.array(row["dq"][12:])
    if name in ("angvel", "angvel_world"):
        return np.array(row["angvel"])
    if name == "yaw":
        return np.array([row["yaw"]])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mj", type=Path, required=True)
    ap.add_argument("--isaac", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold-frac", type=float, default=0.5,
                    help="a field counts as diverged once its normalised difference exceeds this")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    mj = json.loads((a.mj / "state_trace.json").read_text())
    is_ = json.loads((a.isaac / "state_trace.json").read_text())
    n_settle = min(len(mj["settle"]), len(is_["settle"]))
    n_turn = min(len(mj["turn"]), len(is_["turn"]))
    dt = mj["step_dt"]
    print("settle frames compared %d, turn frames compared %d (dt %.4f s)" % (n_settle, n_turn, dt))

    rows = []
    for name, sl, scale, human in FIELDS:
        first = None
        peak = 0.0
        peak_at = None
        series = []
        allrows = list(zip(mj["turn"][:n_turn], is_["turn"][:n_turn]))
        for k, (rm, ri) in enumerate(allrows):
            if sl is not None:
                d = np.abs(np.array(rm["obs"][sl]) - np.array(ri["obs"][sl]))
                div = float(np.max(d)) / scale
            else:
                vm, vi = vec(rm, name), vec(ri, name)
                div = float(np.max(np.abs(vm - vi))) / scale
            series.append(round(div, 4))
            if peak_at is None or div > peak:
                peak, peak_at = div, k
            if first is None and div > a.threshold_frac:
                first = k
        rows.append({"field": name, "meaning": human, "first_diverged_frame": first,
                      "first_diverged_s": None if first is None else round(first * dt, 3),
                      "peak_normalised_divergence": round(peak, 3),
                      "peak_at_frame": peak_at, "scale": scale,
                      "series": series})

    ranked = [r for r in rows if r["first_diverged_frame"] is not None]
    ranked.sort(key=lambda r: r["first_diverged_frame"])
    never = [r for r in rows if r["first_diverged_frame"] is None]

    print("\n%-22s %-10s %-8s %s" % ("FIELD", "FIRST (s)", "PEAK", "MEANING"))
    for r in ranked:
        print("%-22s %-10.3f %-8.2f %s" % (r["field"], r["first_diverged_s"], r["peak_normalised_divergence"], r["meaning"]))
    for r in never:
        print("%-22s %-10s %-8.2f %s" % (r["field"], "never", r["peak_normalised_divergence"], r["meaning"]))

    rep = {"mj": str(a.mj), "isaac": str(a.isaac), "dt": dt,
           "settle_frames": n_settle, "turn_frames": n_turn,
           "threshold_frac": a.threshold_frac,
           "ranked_earliest_first": [{"field": r["field"], "first_s": r["first_diverged_s"],
                                       "peak": r["peak_normalised_divergence"], "meaning": r["meaning"]} for r in ranked],
           "never_diverged": [{"field": r["field"], "peak": r["peak_normalised_divergence"]} for r in never],
           "by_field": rows,
           "interpretation_rule": "the earliest field is the mechanism; fields that diverge later are its consequences"}
    (a.out / "divergence_order.json").write_text(json.dumps(rep, indent=1))

    # tail comparison of the decisive quantities
    tail = list(zip(mj["turn"][-150:], is_["turn"][-150:]))
    def tail_mean(fn):
        return (float(np.mean([fn(x[0]) for x in tail])), float(np.mean([fn(x[1]) for x in tail])))
    wy_m, wy_i = tail_mean(lambda r: r["yaw"])
    dq_m, dq_i = tail_mean(lambda r: (r["dq"][13] + r["dq"][15]) - (r["dq"][12] + r["dq"][14]))
    print("\ntail-3s  yaw rate: mujoco %+.4f  isaac %+.4f" % (wy_m, wy_i))
    print("tail-3s  wheel differential: mujoco %+.4f  isaac %+.4f" % (dq_m, dq_i))
    print("DIVERGENCE_ORDER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
