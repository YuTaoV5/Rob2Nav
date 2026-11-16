#!/usr/bin/env python3
"""Stage-2 comparison: turn the sanctioned evaluation output of start/A/B into one auditable table.

Design rule
-----------
The comparison must use the R4000 run's OWN evaluation entry (snapshot ``train_as2w_isaac_robust.py
--evaluate``), because that entry is what defines the control contract and the scoring.  So this script
never builds an environment and never re-scores anything: it only reads ``speed-evaluation.json`` files
written by that entry and aggregates them.  Reusing the sanctioned scorer is what makes "same control
contract, same scoring" true rather than claimed.

What the sanctioned output can and cannot answer
    available per row : passed, alive, direction_ok, translation_ok, mean velocity, mean signed yaw,
                        wheel target/measured common and differential, unloaded_s, nonwheel_frames,
                        min knee-surface clearance, brake time/distance, hold displacement/heading,
                        last-2s leg error and leg target offset, termination terms
    NOT available     : the first-second / 3-5 s / last-3 s split and the longest CONTINUOUS unload.
                        Those need a per-frame trace recorder on top of the same entry; this script marks
                        them as ``requires_trace_recorder`` instead of silently inventing them.

Worst-side rule (from the operator's protocol)
    a label's turning result is scored on its WORST side, and a run that only improves one side, only one
    seed, or that improves after unload/termination is not an improvement.  This script therefore reports
    left and right separately and reduces to the worst side rather than averaging them away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENARIOS = ("turn_left", "turn_right", "arc_left", "arc_right")
BASE_KEYS = ("track", "stable", "track_and_brake", "hold", "track_brake_hold", "total", "mean_along_m_s")


def find_block(obj, need):
    """Depth-first search for the dict that holds the evaluation sections."""
    if isinstance(obj, dict):
        if all(k in obj for k in need):
            return obj
        for v in obj.values():
            r = find_block(v, need)
            if r is not None:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find_block(v, need)
            if r is not None:
                return r
    return None


def load_eval(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    speed = find_block(raw, ("forward", "back"))
    turn = find_block(raw, ("turn_left", "arc_right"))
    return speed or {}, turn or {}


def summarise(path: Path):
    speed, turn = load_eval(path)
    out = {"source": str(path), "speed": {}, "scenarios": {}}
    for d in ("forward", "back"):
        if d in speed and isinstance(speed[d], dict):
            out["speed"][d] = {k: speed[d].get(k) for k in BASE_KEYS}
    for name in SCENARIOS:
        blk = turn.get(name)
        if not isinstance(blk, dict):
            continue
        rows = [r for r in blk.get("rows", []) if isinstance(r, dict)]
        alive = [r for r in rows if r.get("alive", True)]
        yaw = [float(r.get("mean_velocity", [0, 0, 0])[2]) for r in alive]
        diff_t, diff_m, unloaded, knee = [], [], [], []
        brake_ok = hold_ok = 0
        for r in alive:
            wd = r.get("wheel_target_common_measured_common_target_diff_measured_diff_rad_s") or []
            if len(wd) >= 4:
                diff_t.append(float(wd[2]))
                diff_m.append(float(wd[3]))
            unloaded.append(float(r.get("unloaded_s") or 0.0))
            if r.get("min_knee_surface_m") is not None:
                knee.append(float(r["min_knee_surface_m"]))
            pr = r.get("post_release") or {}
            brake_ok += int(bool(pr.get("brake_ok")))
            hold_ok += int(bool(pr.get("hold_ok")))
        n = max(1, len(alive))
        out["scenarios"][name] = {
            "passed": blk.get("passed"), "total": blk.get("total"),
            "alive": len(alive), "terminated": len(rows) - len(alive),
            "stable": blk.get("stable"), "wrong_direction": blk.get("wrong_direction"),
            "mean_yaw_signed": round(sum(yaw) / n, 5) if yaw else None,
            "mean_yaw_abs": round(sum(abs(v) for v in yaw) / n, 5) if yaw else None,
            "wheel_diff_target_mean": round(sum(diff_t) / max(1, len(diff_t)), 4) if diff_t else None,
            "wheel_diff_measured_mean": round(sum(diff_m) / max(1, len(diff_m)), 4) if diff_m else None,
            "wheel_diff_delivery": (round(abs(sum(diff_m) / len(diff_m)) / abs(sum(diff_t) / len(diff_t)), 4)
                                    if diff_t and abs(sum(diff_t) / len(diff_t)) > 1e-6 else None),
            "unloaded_s_max": round(max(unloaded), 3) if unloaded else None,
            "unloaded_s_mean": round(sum(unloaded) / n, 3) if unloaded else None,
            "min_knee_surface_m": round(min(knee), 4) if knee else None,
            "brake_ok_rate": round(brake_ok / n, 3), "hold_ok_rate": round(hold_ok / n, 3),
        }
    return out


def worst_side(scen):
    pairs = (("turn", "turn_left", "turn_right"), ("arc", "arc_left", "arc_right"))
    out = {}
    for label, left, right in pairs:
        vals = []
        for side in (left, right):
            s = scen.get(side)
            if s and s["total"]:
                vals.append({"side": side, "pass_rate": (s["passed"] or 0) / s["total"],
                             "yaw_abs": s["mean_yaw_abs"], "alive": s["alive"], "total": s["total"]})
        if vals:
            out[label] = sorted(vals, key=lambda v: (v["pass_rate"], -(v["yaw_abs"] or 0)))[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", action="append", required=True,
                    help="label=seed=path/to/speed-evaluation.json (repeatable)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = {}
    for spec in args.entry:
        label, seed, path = spec.split("=", 2)
        runs.setdefault(label, []).append((int(seed), Path(path)))
    print("== stage-2 comparison (sanctioned R4000 eval output only) ==")
    payload = {"labels": {}, "note": {
        "available": "passed/alive/direction/translation, signed yaw, wheel target vs measured differential, "
                     "unloaded_s, knee clearance, brake/hold, termination terms",
        "requires_trace_recorder": ["first-second / 3-5 s / last-3 s signed yaw split",
                                    "longest CONTINUOUS unload duration",
                                    "worst per-axis leg target step per 20 ms",
                                    "leg/wheel velocity RMS"]}}
    for label, items in runs.items():
        per_seed = []
        for seed, path in sorted(items):
            if not path.is_file():
                per_seed.append({"seed": seed, "error": "missing %s" % path})
                continue
            s = summarise(path)
            s["seed"] = seed
            s["worst_side"] = worst_side(s["scenarios"])
            per_seed.append(s)
        payload["labels"][label] = per_seed
        ok = [p for p in per_seed if "scenarios" in p]
        if not ok:
            print("  %-8s no readable runs" % label)
            continue
        print("  %-8s seeds=%s" % (label, [p["seed"] for p in ok]))
        for d in ("forward", "back"):
            parts = []
            for p in ok:
                b = p["speed"].get(d, {})
                if b:
                    parts.append("s%d %s/%s tbH %s mean %.3f" % (p["seed"], b.get("track"), b.get("total"),
                                                                 b.get("track_brake_hold"), b.get("mean_along_m_s") or 0))
            print("    %-8s %s" % (d, " | ".join(parts)))
        for name in SCENARIOS:
            parts = []
            for p in ok:
                s = p["scenarios"].get(name)
                if s:
                    parts.append("s%d %d/%d yaw%+.3f deliv %s term %d" %
                                 (p["seed"], s["passed"] or 0, s["total"], s["mean_yaw_signed"] or 0,
                                  s["wheel_diff_delivery"], s["terminated"]))
            print("    %-11s %s" % (name, " | ".join(parts)))
        for grp, w in ok[0]["worst_side"].items():
            print("    worst %-5s %s" % (grp, w))
    out = args.out or Path(__file__).resolve().parent.parent / "evidence" / "stage2_compare.latest.json"
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("  evidence: %s" % out)
    print("STAGE2_COMPARE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
