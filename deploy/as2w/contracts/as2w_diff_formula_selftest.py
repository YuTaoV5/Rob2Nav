"""Self-test for the wheel differential and diagonal formulas, in FL/FR/RL/RR order.

The operator caught an error in my reported formula: I wrote (FR+RR-FL-FR)/2, which repeats FR and
drops RL.  The agreed definitions are:

    left-right differential = (FR + RR - FL - RL) / 2
    diagonal component      = (FL + RR - FR - RL) / 2

Both are in rad/s when the wheel speeds are in rad/s, and every wheel is positive when rolling FORWARD.

The operator also supplied the expected self-check: feeding [-1, +1, -1, +1] must give a differential of
+2 rad/s with zero diagonal, and the reversed sequence must give -2 rad/s with zero diagonal.  This file
reproduces that check and adds a few more so a future edit cannot quietly reintroduce the typo.

Run: python as2w_diff_formula_selftest.py
"""
from __future__ import annotations

ORDER = ("FL", "FR", "RL", "RR")


def differential(w):
    """(-1, +1, -1, +1) weighting, divided by 2. All wheels positive = rolling forward."""
    fl, fr, rl, rr = w
    return (fr + rr - fl - rl) / 2.0


def diagonal(w):
    """(FL, +RR, -FR, -RL) weighting, divided by 2."""
    fl, fr, rl, rr = w
    return (fl + rr - fr - rl) / 2.0


def main() -> int:
    ok = True
    print("order:", " ".join(ORDER))
    print()

    cases = [
        ("operator check: [-1,+1,-1,+1]", (-1.0, 1.0, -1.0, 1.0), 2.0, 0.0),
        ("operator check: reverse",       (1.0, -1.0, 1.0, -1.0), -2.0, 0.0),
        ("pure forward (all +1)",         (1.0, 1.0, 1.0, 1.0), 0.0, 0.0),
        ("pure backward (all -1)",        (-1.0, -1.0, -1.0, -1.0), 0.0, 0.0),
        # diagonal = (FL + RR - FR - RL)/2, so it needs FL and RR on one side of FR and RL.
        # NOTE: [+1,+1,-1,-1] is NOT a diagonal case - front and rear oppose each other and BOTH
        # quantities come out zero. I got this wrong on the first attempt; kept here as a guard.
        ("diagonal only [+1,-1,-1,+1]",   (1.0, -1.0, -1.0, 1.0), 0.0, 2.0),
        ("diagonal only [-1,+1,+1,-1]",   (-1.0, 1.0, 1.0, -1.0), 0.0, -2.0),
        ("front/rear opposed (both zero)", (1.0, 1.0, -1.0, -1.0), 0.0, 0.0),
        ("left faster [1,0,1,0]",         (1.0, 0.0, 1.0, 0.0), -1.0, 0.0),
        ("right faster [0,1,0,1]",        (0.0, 1.0, 0.0, 1.0), 1.0, 0.0),
    ]
    print("%-32s %-22s %10s %10s %8s %8s" % ("case", "wheels(FL,FR,RL,RR)", "diff", "diag", "ok", "expected"))
    for name, w, exp_d, exp_g in cases:
        d, g = differential(w), diagonal(w)
        good = abs(d - exp_d) < 1e-9 and abs(g - exp_g) < 1e-9
        ok &= good
        print("%-32s %-22s %10.2f %10.2f %8s %s" % (
            name, str(tuple(w)), d, g, "PASS" if good else "FAIL",
            "(%.1f, %.1f)" % (exp_d, exp_g)))

    print()
    print("=== 错误公式会被本自检抓住（回归保护）===")
    def wrong_reported(w):
        fl, fr, rl, rr = w
        return (fr + rr - fl - fr) / 2.0          # the typo I actually published
    w = (-1.0, 1.0, -1.0, 1.0)
    print("  correct  = %+.3f" % differential(w))
    print("  my typo  = %+.3f" % wrong_reported(w))
    if abs(differential(w) - wrong_reported(w)) < 1e-9:
        print("  FAIL the typo is indistinguishable, the test is useless")
        ok = False
    else:
        print("  PASS they differ, so the self-test catches the typo")

    print()
    print("DIFF_FORMULA_SELFTEST_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
