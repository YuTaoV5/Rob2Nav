#!/usr/bin/env python3
"""Report the acceptance matrix under BOTH basis: all-seed (min basis) and mean basis.

The operator requires the matrix to be reported with mean and min bases side by side, but the acceptance
script's console output only prints a pass count, and its JSON carries the mean and the worst-case numbers
in separate fields without a verdict for each basis.  This reads that JSON and prints both.

Two bases:

  min basis   a case passes only if EVERY seed passes - the strict reading, and the one that decides whether
              the candidate may advance.  This is the script's own pass_all.
  mean basis  a case passes if the mean tracking across seeds is inside the band, the mean support is above
              the support floor, no seed terminated, and every seed's upright is above the floor.  Reported
              for comparison and for spotting cases that are close rather than hopeless.

A case that passes mean but fails min is a case with seed scatter - worth knowing, because the goal selects
on worst-case behaviour, so those cases must NOT be promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

YAW_BAND = (0.70, 1.30)
VX_BAND = (0.70, 1.30)
SUPPORT_MIN = 0.95
UPRIGHT_MIN = 0.85


def verdict(j, basis: str, case: str = None) -> str:
    if j.get("term_any", 0):
        return "FAIL"
    if j.get("pass_all"):
        # already all-seed pass
        pass
    if basis == "min":
        return "PASS" if j.get("pass_all") else "FAIL"
    # mean basis
    ok = True
    if j.get("support_min", 0.0) < SUPPORT_MIN:
        ok = False                                    # worst seed must still be supported
    if j.get("upright_min", 0.0) < UPRIGHT_MIN:
        ok = False
    # The tracking band only applies to the cases that actually ask for tracking.  A brake case's tail
    # vx is near zero BY DESIGN, so applying the 0.70-1.30 vx band to it would wrongly fail every brake.
    # This was caught by running the tool on a real report: brake_fwd/bwd came out mean-FAIL while the
    # acceptance script's own verdict had them PASS.
    name = (case or "").lower()
    if ("turn" in name or "arc" in name) and j.get("yaw_track_mean") is not None:
        if not (YAW_BAND[0] <= abs(j["yaw_track_mean"]) <= YAW_BAND[1]):
            ok = False
    if ("straight" in name or "arc" in name) and j.get("vx_track_mean") is not None:
        if not (VX_BAND[0] <= abs(j["vx_track_mean"]) <= VX_BAND[1]):
            ok = False
    return "PASS" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rep = json.loads(a.json.read_text())
    rows = []
    print("%-22s %-6s %-6s %8s %8s %8s %8s %7s %s"
          % ("case", "min", "mean", "yaw_mean", "sup_min", "up_min", "exec_mean", "tau_max", "seeds"))
    for cand in rep.get("candidates", []):
        label = cand.get("label", "?")
        for case, j in cand.get("cases", {}).items():
            vm = verdict(j, "min", case)
            vb = verdict(j, "mean", case)
            rows.append({"candidate": label, "case": case, "min_basis": vm, "mean_basis": vb, **{
                k: j.get(k) for k in ("yaw_track_mean", "vx_track_mean", "support_min",
                                       "upright_min", "exec_mean", "tau_abs_max", "ff_abs_max",
                                       "cap_hits", "verdicts")}})
            print("%-22s %-6s %-6s %8s %8.3f %8.3f %8s %7.2f %s"
                  % (case, vm, vb,
                     "-" if j.get("yaw_track_mean") is None else "%+.2f" % j["yaw_track_mean"],
                     j.get("support_min", 0.0), j.get("upright_min", 0.0),
                     "-" if j.get("exec_mean") is None else "%.3f" % j["exec_mean"],
                     j.get("tau_abs_max", 0.0),
                     ",".join(x[0] for x in j.get("verdicts", []))))
        n_min = sum(1 for r in rows if r["candidate"] == label and r["min_basis"] == "PASS")
        n_mean = sum(1 for r in rows if r["candidate"] == label and r["mean_basis"] == "PASS")
        scatter = [r["case"] for r in rows if r["candidate"] == label
                   and r["mean_basis"] == "PASS" and r["min_basis"] == "FAIL"]
        print("  ==> %s  min basis %d/%d   mean basis %d/%d   mean-only(scatter) %d %s"
              % (label, n_min, len(cand.get("cases", {})), n_mean, len(cand.get("cases", {})),
                 len(scatter), scatter if scatter else ""))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"source": str(a.json), "rows": rows}, indent=1))
        print("  wrote", a.out)
    print("DUAL_BASIS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
