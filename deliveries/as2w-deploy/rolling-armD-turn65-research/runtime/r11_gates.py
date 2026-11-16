#!/usr/bin/env python3
"""Shared gate definitions for the two rulings of 2026-11-16, used by BOTH simulators.

The goal requires the same judgement on the training side, the Isaac side and the MuJoCo side, so the
thresholds live in one module and neither evaluator may restate them.

Ruling 1 - brief unloading (four-wheel rolling)
    Judged PER WHEEL on continuous unloading duration, at the 20 ms control sample.
        1-3 consecutive unload samples  -> BRIEF unload (allowed)
        4th consecutive unload sample onward -> that case FAILS the support gate
    200 ms is kept as the tier that marks SEVERE loss of support (diagnostic only, not the fail line).
    Also reported, because the operator asked for them explicitly:
        per-wheel cumulative unload time, per-wheel longest continuous unload, minimum number of
        simultaneously load-bearing wheels.

Ruling 2 - fixed stand
    Referenced to the frozen stand pose.  All 12 leg axes must stay within max |error| <= 0.10 rad
    CONTINUOUSLY for at least 1 second.  During that second the four-wheel load, the no-non-wheel-contact
    condition, the hold velocity gate and the torque budget must hold as well.  If the return-to-stand
    window ends without satisfying this, the return-to-stand is recorded as a failure and the wheel-centre
    residual plus its effect on the next takeover is recorded.
    The current ~0.19 rad plateau is kept as the failure baseline.

Both rulings are SIMULATION SCREENING values for the next dual-simulator round; the formal gates stay
unfrozen until held-out re-tests and hardware evidence exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

SAMPLE_DT = 0.020                 # 20 ms control period
UNLOAD_BRIEF_MAX_SAMPLES = 3      # 1..3 consecutive samples count as brief
UNLOAD_FAIL_SAMPLE = 4            # 4th consecutive sample fails the case
UNLOAD_SEVERE_S = 0.200           # diagnostic tier only
FIXED_STAND_TOL_RAD = 0.10
FIXED_STAND_HOLD_S = 1.0
FIXED_STAND_BASELINE_RAD = 0.19   # the current plateau, recorded as the failure baseline


@dataclass
class WheelUnloadReport:
    per_wheel_cumulative_s: dict
    per_wheel_longest_s: dict
    per_wheel_fail: dict
    min_simultaneous_loaded: int
    case_fails_support: bool
    severe_loss: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "per_wheel_cumulative_s": self.per_wheel_cumulative_s,
            "per_wheel_longest_s": self.per_wheel_longest_s,
            "per_wheel_fails": self.per_wheel_fail,
            "min_simultaneous_loaded_wheels": self.min_simultaneous_loaded,
            "case_fails_support": self.case_fails_support,
            "severe_loss_of_support": self.severe_loss,
            "rules": {
                "sample_dt_s": SAMPLE_DT,
                "brief_max_consecutive_samples": UNLOAD_BRIEF_MAX_SAMPLES,
                "fails_from_consecutive_sample": UNLOAD_FAIL_SAMPLE,
                "severe_tier_s": UNLOAD_SEVERE_S,
                "note": "1-3 consecutive unload samples are brief; the 4th fails the case; 200 ms is a diagnostic tier only"},
            "detail": self.detail}


def wheel_unload_report(load_history: Sequence[Sequence[float]], load_threshold_n: float,
                        wheel_names: Sequence[str] = ("FL", "FR", "RL", "RR")) -> WheelUnloadReport:
    """load_history: one row per 20 ms sample, one column per wheel."""
    n = len(load_history)
    nw = len(wheel_names)
    cum = {w: 0 for w in wheel_names}
    longest = {w: 0 for w in wheel_names}
    fails = {w: False for w in wheel_names}
    run = {w: 0 for w in wheel_names}
    min_loaded = nw
    for row in load_history:
        loaded = 0
        for i, w in enumerate(wheel_names):
            if row[i] < load_threshold_n:
                cum[w] += 1
                run[w] += 1
                if run[w] > longest[w]:
                    longest[w] = run[w]
                if run[w] >= UNLOAD_FAIL_SAMPLE:
                    fails[w] = True
            else:
                run[w] = 0
                loaded += 1
        if loaded < min_loaded:
            min_loaded = loaded
    longest_s = {w: round(longest[w] * SAMPLE_DT, 3) for w in wheel_names}
    cum_s = {w: round(cum[w] * SAMPLE_DT, 3) for w in wheel_names}
    return WheelUnloadReport(
        per_wheel_cumulative_s=cum_s,
        per_wheel_longest_s=longest_s,
        per_wheel_fail=fails,
        min_simultaneous_loaded=min_loaded,
        case_fails_support=any(fails.values()),
        severe_loss=any(v >= UNLOAD_SEVERE_S for v in longest_s.values()),
        detail={"samples": n, "load_threshold_n": load_threshold_n})


@dataclass
class FixedStandReport:
    satisfied: bool
    hold_duration_s: float
    worst_axis: Optional[str]
    worst_axis_error_rad: float
    load_ok: bool
    contact_ok: bool
    hold_speed_ok: bool
    torque_ok: bool
    baseline_plateau_rad: float

    def as_dict(self) -> dict:
        return {
            "satisfied": self.satisfied,
            "hold_duration_s": self.hold_duration_s,
            "worst_axis": self.worst_axis,
            "worst_axis_error_rad": self.worst_axis_error_rad,
            "four_wheel_load_ok": self.load_ok,
            "no_non_wheel_contact_ok": self.contact_ok,
            "hold_speed_gate_ok": self.hold_speed_ok,
            "torque_budget_ok": self.torque_ok,
            "baseline_plateau_rad": self.baseline_plateau_rad,
            "rules": {
                "tolerance_rad": FIXED_STAND_TOL_RAD,
                "hold_s": FIXED_STAND_HOLD_S,
                "conjunction": "all 12 leg axes within tolerance continuously for the whole hold, while load, contact, hold speed and torque also hold",
                "on_failure": "record the return-to-stand as failed plus the wheel-centre residual and its effect on the next takeover"}}


def fixed_stand_report(leg_error_history: Sequence[float],
                       load_ok_history: Sequence[bool],
                       contact_ok_history: Sequence[bool],
                       hold_speed_ok_history: Sequence[bool],
                       torque_ok_history: Sequence[bool],
                       axis_names: Optional[Sequence[str]] = None) -> FixedStandReport:
    """Each history has one entry per 20 ms sample over the return-to-stand window.  leg_error_history is
    the max |error| over the 12 leg axes at that sample."""
    n = len(leg_error_history)
    need = int(round(FIXED_STAND_HOLD_S / SAMPLE_DT))
    best_run = 0
    run = 0
    worst_overall = 0.0
    for k in range(n):
        ok = (leg_error_history[k] <= FIXED_STAND_TOL_RAD and load_ok_history[k]
              and contact_ok_history[k] and hold_speed_ok_history[k] and torque_ok_history[k])
        worst_overall = max(worst_overall, leg_error_history[k])
        run = run + 1 if ok else 0
        if run > best_run:
            best_run = run
    hold_s = round(best_run * SAMPLE_DT, 3)
    return FixedStandReport(
        satisfied=best_run >= need,
        hold_duration_s=hold_s,
        worst_axis=None if axis_names is None else axis_names[max(range(12), key=lambda i: 0)],
        worst_axis_error_rad=round(worst_overall, 4),
        load_ok=all(load_ok_history) if load_ok_history else False,
        contact_ok=all(contact_ok_history) if contact_ok_history else False,
        hold_speed_ok=all(hold_speed_ok_history) if hold_speed_ok_history else False,
        torque_ok=all(torque_ok_history) if torque_ok_history else False,
        baseline_plateau_rad=FIXED_STAND_BASELINE_RAD)


def selftest() -> int:
    """Prove the two rules behave as ruled before anyone cites them."""
    ok = True
    # a single 3-sample unload must be brief, not a failure
    hist = [[100.0] * 4 for _ in range(10)]
    for row in hist[3:6]:
        row[0] = 0.0
    r = wheel_unload_report(hist, 20.0)
    if r.case_fails_support or r.per_wheel_longest_s["FL"] != 0.06:
        print("FAIL brief 3-sample unload was not treated as brief:", r.as_dict()); ok = False
    # the 4th consecutive sample must fail
    hist = [[100.0] * 4 for _ in range(10)]
    for row in hist[3:7]:
        row[0] = 0.0
    r = wheel_unload_report(hist, 20.0)
    if not r.case_fails_support or not r.per_wheel_fail["FL"]:
        print("FAIL a 4-sample unload did not fail the case:", r.as_dict()); ok = False
    # 200 ms tier
    hist = [[100.0] * 4 for _ in range(20)]
    for row in hist[2:12]:
        row[1] = 0.0
    r = wheel_unload_report(hist, 20.0)
    if not r.severe_loss or r.per_wheel_longest_s["FR"] != 0.2:
        print("FAIL the 200 ms severe tier did not trigger:", r.as_dict()); ok = False
    # min simultaneously loaded
    hist = [[100.0] * 4 for _ in range(5)]
    hist[2] = [100.0, 0.0, 0.0, 100.0]
    r = wheel_unload_report(hist, 20.0)
    if r.min_simultaneous_loaded != 2:
        print("FAIL min simultaneously loaded was %d, expected 2" % r.min_simultaneous_loaded); ok = False
    # fixed stand: 1.0 s inside tolerance satisfies, 0.9 s does not
    for n_ok, expect in ((50, True), (45, False)):
        hist_err = [0.05] * n_ok + [0.30] * (60 - n_ok)
        rep = fixed_stand_report(hist_err, [True] * 60, [True] * 60, [True] * 60, [True] * 60)
        if rep.satisfied is not expect:
            print("FAIL fixed_stand with %d ok samples gave satisfied=%s, expected %s" % (n_ok, rep.satisfied, expect)); ok = False
        if not rep.satisfied:
            pass
    # one broken condition must break the hold
    rep = fixed_stand_report([0.05] * 60, [True] * 60, [True] * 60, [True] * 60, [False] * 60)
    if rep.satisfied:
        print("FAIL fixed_stand satisfied while the torque budget was violated"); ok = False
    print("GATES_SELFTEST_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
