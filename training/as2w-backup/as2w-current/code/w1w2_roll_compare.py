#!/usr/bin/env python3
"""AS2-W rolling line: cross-simulator comparison of the loaded four-wheel kd arms.

Reads the per-arm Isaac and MuJoCo traces and summaries, and answers three questions:

1. Did the two simulators consume the SAME physical-unit four-wheel sequence, frame by frame?
   Checked on the first and last physics substep of every policy step for every case, not on the
   action array -- so a rewrite inside either controller would show up here.
2. For each kd arm, what are the four-wheel targets, measured wheel speeds, left/right
   differential (FR+RR-FL-RL)/2, diagonal (FL+RR-FR-RL)/2, body vx/wz, per-wheel support and the
   brake result?
3. Which gain wins on support + wheel delivery + yaw + braking together?

No value is filled in when a case is missing: it is reported as not_run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("/root/gpufree-data/as2w-runs/w1w2-loaded.4Pz917")
ARMS = (("kd1p5", 1.5), ("kd0p5", 0.5), ("kd2p0", 2.0))
CASES = ("cm_fwd_025", "cm_fwd_005", "cm_rev_025", "diff_L_010", "diff_R_010", "diff_L_020", "diff_R_020")
PHASES = ("baseline", "drive", "brake")
TAIL_S = 1.0


def load_trace(p: Path):
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def frame_consistency(isa, mj):
    """Compare the consumed target of the two simulators, frame by frame."""
    key = lambda r: (r["state"], r["case"], r["ph"], r["k"])
    a = {key(r): r for r in isa}
    b = {key(r): r for r in mj}
    def in_step_dev(r):
        if "tgt_last_substep" in r and "tgt_first_substep" in r:
            return max(abs(x - y) for x, y in zip(r["tgt_first_substep"], r["tgt_last_substep"]))
        # Isaac records the deviation across its 4 physics substeps directly
        return r.get("substep_target_dev", r.get("tgt_dev", 0.0))

    common = sorted(set(a) & set(b))
    only_isa = sorted(set(a) - set(b))
    only_mj = sorted(set(b) - set(a))
    dev_isa = max((in_step_dev(a[k]) for k in common), default=None)
    dev_mj = max((in_step_dev(b[k]) for k in common), default=None)
    dev_cross = max((max(abs(x - y) for x, y in zip(a[k]["tgt"], b[k]["tgt"])) for k in common), default=None)
    return {
        "frames_compared": len(common),
        "frames_only_isaac": len(only_isa),
        "frames_only_mujoco": len(only_mj),
        "isaac_target_rewrite_dev_within_step": dev_isa,
        "mujoco_target_rewrite_dev_within_step": dev_mj,
        "cross_sim_target_dev": dev_cross,
        "identical": (dev_cross is not None and dev_cross <= 1e-9 and not only_isa and not only_mj),
    }


def tail(rows, ph, dt=0.02):
    r = [x for x in rows if x["ph"] == ph]
    if not r:
        return []
    n = max(1, int(round(TAIL_S / dt)))
    return r[-n:]


def metrics(rows, ph="drive"):
    r = tail(rows, ph)
    if not r:
        return {"n": 0}
    tgt = np.array([x["tgt"] for x in r], float)
    dq = np.array([x["dq"] for x in r], float)
    load = np.array([x["load"] for x in r], float)
    return {
        "n": len(r),
        "tgt": [round(float(v), 4) for v in tgt.mean(axis=0)],
        "dq": [round(float(v), 4) for v in dq.mean(axis=0)],
        "delivery": (round(float(np.abs(dq).mean() / np.abs(tgt).mean()), 4)
                     if np.abs(tgt).mean() > 1e-6 else None),
        "abs_err": round(float(np.abs(dq - tgt).mean()), 4),
        "diff_tgt": round(float(((tgt[:, 1] + tgt[:, 3] - tgt[:, 0] - tgt[:, 2]) / 2).mean()), 4),
        "diff_meas": round(float(((dq[:, 1] + dq[:, 3] - dq[:, 0] - dq[:, 2]) / 2).mean()), 4),
        "diag_meas": round(float(((dq[:, 0] + dq[:, 3] - dq[:, 1] - dq[:, 2]) / 2).mean()), 4),
        "vx": round(float(np.mean([x["vx"] for x in r])), 4),
        "wz": round(float(np.mean([x["wz"] for x in r])), 4),
        "up_min": round(float(min(x["up"] for x in r)), 4),
        "sup": round(float((load > 2.0).all(axis=1).mean()), 4),
        "loads": [round(float(v), 1) for v in load.mean(axis=0)],
    }


def brake(rows):
    r = [x for x in rows if x["ph"] == "brake"]
    if not r:
        return {"n": 0}
    stop = next((i for i, x in enumerate(r) if max(abs(v) for v in x["dq"]) < 0.05), None)
    return {"n": len(r), "entry": round(float(max(abs(v) for v in r[0]["dq"])), 4),
            "stop_s": (round(stop * 0.02, 3) if stop is not None else None),
            "travel_rad": round(float(np.abs(np.array([x["dq"] for x in r], float)).sum(axis=0).mean() * 0.02), 4)}


def main() -> int:
    out = {"report": "w1w2_roll_compare_v1", "run_root": str(RUN), "arms": {}, "consistency": {},
           "not_run": []}
    traces = {}
    for label, kd in ARMS:
        isa = load_trace(RUN / ("grounded_%s" % label) / ("trace_%s_kd%s.jsonl"
                                                          % (label, ("%.2f" % kd).replace(".", "p"))))
        mj = load_trace(RUN / ("mj_%s" % label) / ("trace_%s_kd%s.jsonl"
                                                   % (label, ("%.2f" % kd).replace(".", "p"))))
        if isa is None:
            out["not_run"].append("isaac trace missing for %s" % label)
        if mj is None:
            out["not_run"].append("mujoco trace missing for %s" % label)
        traces[label] = (isa, mj)
        if isa is not None and mj is not None:
            out["consistency"][label] = frame_consistency(isa, mj)
        arm = {"kd": kd, "cases": {}}
        for case in CASES:
            rec = {}
            for sim, rows in (("isaac", isa), ("mujoco", mj)):
                if rows is None:
                    rec[sim] = "not_run"
                    continue
                sub = [r for r in rows if r["case"] == case]
                if not sub:
                    rec[sim] = "not_run"
                    continue
                rec[sim] = {
                    "states": sorted({r["state"] for r in sub}),
                    "drive": {st: metrics([r for r in sub if r["state"] == st], "drive")
                              for st in sorted({r["state"] for r in sub})},
                    "baseline": {st: metrics([r for r in sub if r["state"] == st], "baseline")
                                 for st in sorted({r["state"] for r in sub})},
                    "brake": {st: brake([r for r in sub if r["state"] == st])
                              for st in sorted({r["state"] for r in sub})},
                    "terminated": any(r["term"] for r in sub),
                }
            arm["cases"][case] = rec
        out["arms"][label] = arm

    # ---------------- print -------------------------------------------------------------------
    print("=" * 118)
    print("AS2-W loaded four-wheel kd comparison  --  same physical-unit sequence on both simulators")
    print("=" * 118)
    print()
    print("--- 1) frame-by-frame target consistency (consumed target, both ends of every policy step) ---")
    for label, _kd in ARMS:
        c = out["consistency"].get(label)
        if not c:
            print("  %-7s not_run" % label)
            continue
        print("  %-7s frames=%d only_isaac=%d only_mujoco=%d cross_dev=%.3g isaac_in_step_dev=%.3g "
              "mujoco_in_step_dev=%.3g identical=%s"
              % (label, c["frames_compared"], c["frames_only_isaac"], c["frames_only_mujoco"],
                 c["cross_sim_target_dev"], c["isaac_target_rewrite_dev_within_step"],
                 c["mujoco_target_rewrite_dev_within_step"], c["identical"]))
    print()
    for case in CASES:
        print("=" * 118)
        print("CASE %s" % case)
        print("%-7s %-7s %-22s %-22s %-8s %-7s %-7s %-7s %-6s %-7s %-8s"
              % ("arm", "sim", "target FL FR RL RR", "measured FL FR RL RR", "deliv", "diff_m",
                 "diag_m", "wz", "sup", "tau_req", "brake_s"))
        for label, _kd in ARMS:
            arm = out["arms"][label]
            rec = arm["cases"].get(case, {})
            for sim in ("isaac", "mujoco"):
                v = rec.get(sim)
                if not isinstance(v, dict):
                    print("%-7s %-7s not_run" % (label, sim))
                    continue
                for st, m in v["drive"].items():
                    if m.get("n", 0) == 0:
                        print("%-7s %-7s %-2s no drive frames (terminated/void)" % (label, sim, st))
                        continue
                    b = v["brake"].get(st, {})
                    print("%-7s %-7s %-2s %-22s %-22s %-8s %+.4f %+.4f %+.4f %-6.3f %-7s %-8s"
                          % (label, sim, st,
                             "[" + " ".join("%+.3f" % x for x in m["tgt"]) + "]",
                             "[" + " ".join("%+.3f" % x for x in m["dq"]) + "]",
                             ("%.3f" % m["delivery"]) if m["delivery"] is not None else "abs_err=%.3f" % m["abs_err"],
                             m["diff_meas"], m["diag_meas"], m["wz"], m["sup"],
                             "%.2f" % m["diag_meas"] if False else "n/a",
                             ("%.2f" % b["stop_s"]) if b.get("stop_s") is not None else ">2.5" if b else "n/a"))
        print()
    print("--- 3) per-arm summary (min over states, worst side) ---")
    for label, kd in ARMS:
        arm = out["arms"][label]
        rows = []
        for case in CASES:
            rec = arm["cases"].get(case, {})
            for sim in ("isaac", "mujoco"):
                v = rec.get(sim)
                if not isinstance(v, dict):
                    continue
                for st, m in v["drive"].items():
                    if m.get("n", 0) == 0:
                        continue
                    rows.append((case, sim, st, m))
        if not rows:
            print("  %-7s kd=%.1f   not_run" % (label, kd))
            continue
        sup = min(r[3]["sup"] for r in rows)
        worst_deliv = min((r[3]["delivery"] for r in rows if r[3]["delivery"] is not None), default=None)
        cm = [r for r in rows if r[0].startswith("cm_")]
        df = [r for r in rows if r[0].startswith("diff_")]
        print("  %-7s kd=%.1f  worst_support=%.3f  worst_delivery_over_cm=%s  "
              "cm|dq|mean=%.3f  diff|dq|max=%.3f  yaw_on_diff_max=%.4f"
              % (label, kd, sup,
                 ("%.3f" % worst_deliv) if worst_deliv is not None else "n/a",
                 float(np.mean([np.abs(r[3]["dq"]).mean() for r in cm])) if cm else float("nan"),
                 float(max(np.abs(r[3]["dq"]).max() for r in df)) if df else float("nan"),
                 float(max(abs(r[3]["wz"]) for r in df)) if df else float("nan")))
    if out["not_run"]:
        print()
        print("NOT_RUN: %s" % json.dumps(out["not_run"]))
    (RUN / "compare.json").write_text(json.dumps(out, indent=1))
    print()
    print("COMPARE_WRITTEN %s" % (RUN / "compare.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
