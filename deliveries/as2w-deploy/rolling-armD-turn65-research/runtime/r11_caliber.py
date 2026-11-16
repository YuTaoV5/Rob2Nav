#!/usr/bin/env python3
"""Report MuJoCo 11-case matrix results under BOTH pass-count calibers.

Why
  The baseline m3998 itself fails stage-1 (four-wheel support < 0.95) on three of the
  four pure-turn cases:
      turn_left_0.05   support 0.747 / 0.687 / 0.647
      turn_left_0.10   support 0.493 / 0.507 / 0.500
      turn_right_0.10  support 0.767 / 0.773 / 0.793
  (verified from matrix-v2-20261116-025950/mujoco_matrix_v2.json)
  A raw "x/11" therefore under-states nothing but OVER-states difficulty: no candidate
  can pass those three without first beating the baseline's own support behaviour, and
  the pure-turn rows then measure support rather than yaw tracking.

Calibers
  raw11   : all 11 cases, kept for historical comparability
  ach8    : the 8 cases whose stage-1 is reachable at baseline (everything except the
            three listed above).  This is the caliber for judging real capability change.

Usage
  python r11_caliber.py <dir-or-json> [<more>...]
  Accepts either a mujoco_matrix_v2.json path, or a directory that contains one
  (searched at the top level and one level down).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CASES = ["hold", "turn_left_0.05", "turn_right_0.05", "turn_left_0.10",
         "turn_right_0.10", "straight_fwd_0.25", "straight_bwd_0.25",
         "arc_left_0.25", "arc_right_0.25", "brake_fwd_0.25", "brake_bwd_0.25"]
# Cases that the BASELINE m3998 fails at stage-1 (four-wheel support).  This is a
# baseline property only, NOT a proof that the case is unreachable: later checkpoints
# DO reach stage-1 on turn_right_0.10 (support 1.000, armD3 m4100) and on
# turn_left_0.05 (support 0.987, armE m4500).  The caliber below is therefore
# recomputed from the data (union of cases any provided run passed) so it cannot
# silently under-count later.
BASELINE_STAGE1_FAILING = {"turn_left_0.05", "turn_left_0.10", "turn_right_0.10"}
ACHIEVABLE = [c for c in CASES if c not in BASELINE_STAGE1_FAILING]  # default/single-run view


def find_json(p: Path):
    if p.is_file() and p.suffix == ".json":
        return p
    for name in ("mujoco_matrix_v2.json", "mujoco_matrix_level.json",
                 "mujoco_matrix_level_lambda.json", "mujoco_matrix_lp.json"):
        f = p / name
        if f.exists():
            return f
    hits = sorted(p.glob("*/mujoco_matrix_v2.json"))
    return hits[0] if hits else None


def judge_states(rows):
    """A case passes only if EVERY stand state passed."""
    if not rows:
        return None, None
    passed = all(r.get("verdict") == "PASS" for r in rows)
    sup = [r.get("support_tail", float("nan")) for r in rows]
    return passed, (min(sup) if sup else None)


def report(path: Path):
    js = find_json(path)
    if js is None:
        print("  [%s] no matrix json found" % path)
        return None
    d = json.load(open(js))
    cases = d.get("cases", {})
    res = {}
    for c in CASES:
        ok, sup = judge_states(cases.get(c))
        res[c] = (ok, sup)
    raw = sum(1 for c in CASES if res[c][0])
    ach = sum(1 for c in ACHIEVABLE if res[c][0])
    print("  %s" % js.parent.name)
    print("    raw11 = %d/11    ach8 = %d/8" % (raw, ach))
    for c in CASES:
        ok, sup = res[c]
        if ok is None:
            continue
        mark = "PASS" if ok else "fail"
        tag = "  (baseline stage1-failing)" if c in BASELINE_STAGE1_FAILING else ""
        print("      %-18s %-4s support_min=%.3f%s"
              % (c, mark, sup if sup is not None else float("nan"), tag))
    return {"name": js.parent.name, "raw11": raw, "ach8": ach,
            "cases": {c: bool(res[c][0]) for c in CASES}}


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = []
    for a in sys.argv[1:]:
        r = report(Path(a))
        if r:
            out.append(r)
            print()
    if len(out) > 1:
        # data-driven reachable set: every case that at least one run passed
        reached = set()
        for r in out:
            for c, ok in r.get("cases", {}).items():
                if ok:
                    reached.add(c)
        never = [c for c in CASES if c not in reached]
        n_ach = len(CASES) - len(never)
        print("=== SUMMARY ===")
        print("  baseline m3998: raw11=4")
        print("  cases NEVER passed by any evaluated run: %s" % (", ".join(never) if never else "none"))
        print("  data-driven reachable set = %d cases" % n_ach)
        print()
        print("  %-38s %7s %8s" % ("run", "raw11", "reach%d" % n_ach))
        for r in out:
            got = sum(1 for c in CASES if c in reached and r.get("cases", {}).get(c))
            print("  %-38s %5d/11 %6d/%d" % (r["name"][:38], r["raw11"], got, n_ach))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
