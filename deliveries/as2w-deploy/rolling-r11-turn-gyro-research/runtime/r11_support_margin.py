#!/usr/bin/env python3
"""Support-margin geometry, written to be SELF-VALIDATED before any number from it is cited.

Why this file exists: two earlier probes reported opposite signs for the three-foot margin on the same
left-spin case (+0.0139 m with RR removed, -0.0173 m with FR removed), and the same code returned a NEGATIVE
margin for FOUR-wheel support - which is impossible for a stance that is standing.  The geometry had never
been validated, so neither number could be used.

This module defines one convex-polygon margin that is correct for both a triangle and a quadrilateral:

    margin = min over edges of the SIGNED distance from the point to the edge line,
             with the sign positive when the point is on the interior side.

It is then validated against cases whose answers are known independently:

  unit square, centre point              -> positive, and equal to 0.5
  unit square, point outside by 0.2      -> -0.2
  equilateral-ish triangle, centroid     -> positive
  triangle, point outside one edge       -> negative, matching the analytic distance
  a real settled four-wheel stance        -> MUST be positive (this is the check the earlier code failed)
  a real settled stance with one wheel removed -> reported for all four removals, sign discussed

Run with --selftest to execute only the analytic checks; run with --xml/--ckpt to also do the stance checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convex_hull(points):
    """Monotone chain; returns the hull in counter-clockwise order."""
    pts = sorted(set(map(tuple, np.round(points, 9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def support_margin(point, supports):
    """Minimum signed distance from `point` to the convex hull edges of `supports`.

    Positive inside, negative outside.  Correct for 3 or 4 (or more) supports because it uses the hull.
    """
    hull = convex_hull(np.asarray(supports, float))
    if len(hull) < 3:
        return None
    p = np.asarray(point, float)
    # hull is CCW, so the interior lies to the LEFT of each directed edge
    best = None
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        ab = b - a
        n = np.array([-ab[1], ab[0]])          # left normal
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        d = float(np.dot(p - a, n) / nn)
        best = d if best is None else min(best, d)
    return best


def selftest():
    ok = True
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cases = [
        ("square centre", [0.5, 0.5], sq, 0.5),
        ("square near edge", [0.5, 0.05], sq, 0.05),
        ("square outside", [1.2, 0.5], sq, -0.2),
        ("square corner outside", [-0.1, -0.1], sq, -0.1),
    ]
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cen = tri.mean(axis=0)
    # the centroid sits 1/3 from each leg and (1 - 2/3)/sqrt(2) from the hypotenuse; the hypotenuse
    # is the nearest, and that is what the margin must return
    cases.append(("triangle centroid", cen, tri, (1.0 - cen.sum()) / np.sqrt(2)))
    cases.append(("triangle outside hypotenuse", [1.0, 1.0], tri, -1.0 / np.sqrt(2)))
    print("%-30s %10s %10s %s" % ("case", "computed", "expected", "ok"))
    for name, p, s, expect in cases:
        got = support_margin(p, s)
        good = got is not None and abs(got - expect) < 1e-6
        ok = ok and good
        print("%-30s %10.6f %10.6f %s" % (name, got, expect, "OK" if good else "MISMATCH"))
    # order independence: shuffling the supports must not change the answer
    rng = np.random.default_rng(0)
    order_ok = True
    for _ in range(20):
        perm = rng.permutation(4)
        a = support_margin([0.13, 0.37], sq)
        b = support_margin([0.13, 0.37], sq[perm])
        if abs(a - b) > 1e-12:
            order_ok = False
            print("ORDER MISMATCH", a, b, perm)
    ok = ok and order_ok
    print("order independence:", "OK" if order_ok else "FAILED")
    print("SELFTEST_" + ("PASS" if ok else "FAIL"))
    return ok


def stance_check(xml: Path, out: Path | None):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    LEGS = ("FL", "FR", "RL", "RR")
    JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in LEGS]
    DEF = np.array([{"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}[n.split("_")[1]] for n in JOINTS])
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    wb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{L}_foot") for L in LEGS]
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = (0.0, 0.0, 0.381395); d.qpos[3:7] = (1.0, 0, 0, 0); d.qpos[qadr] = DEF
    mujoco.mj_forward(m, d)
    for _ in range(6000):
        mujoco.mj_step(m, d)
    pts = np.array([d.xpos[b][:2] for b in wb])
    tot = 0.0; acc = np.zeros(2)
    for i in range(m.nbody):
        tot += m.body_mass[i]; acc += m.body_mass[i] * d.xpos[i][:2]
    com = acc / max(tot, 1e-9)
    rep = {"com_xy": com.tolist(), "wheels_xy": pts.tolist(), "margins": {}}
    m4 = support_margin(com, pts)
    rep["margins"]["all_four"] = m4
    print("four-wheel stance margin = %+.4f m  (%s)" % (m4, "POSITIVE as required" if m4 > 0 else "NEGATIVE - WRONG"))
    ok = m4 is not None and m4 > 0
    for drop in range(4):
        idx = [i for i in range(4) if i != drop]
        m3 = support_margin(com, pts[idx])
        rep["margins"]["drop_" + LEGS[drop]] = m3
        print("  drop %-2s -> margin %+.4f m" % (LEGS[drop], m3))
    rep["four_wheel_positive"] = bool(ok)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1))
        print("wrote", out)
    print("STANCE_CHECK_" + ("PASS" if ok else "FAIL"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--xml", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ok = True
    if a.selftest or a.xml is None:
        ok = selftest() and ok
    if a.xml is not None:
        ok = stance_check(a.xml, a.out) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
