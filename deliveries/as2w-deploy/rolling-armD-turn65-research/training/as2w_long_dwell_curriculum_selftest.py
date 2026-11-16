#!/usr/bin/env python3
"""Self-test for the long-dwell curriculum arithmetic, without Isaac.

The operator requires that the log PROVE the command actually held for 8-12 s after the ramp.  Before
trusting any training run, this exercises the same arithmetic on a CPU mock and asserts:

  1. only the intended fraction of turn/arc samples is marked long
  2. a long sample gets a planned dwell inside [8, 12] s and a SHORT one keeps the parent cadence
  3. `time_left` really receives the planned dwell (this is what makes the hold happen at all)
  4. the elapsed clock reaches the plan and the hold is closed as COMPLETED
  5. a hold cut short by a new resample is closed as TRUNCATED, not silently dropped
  6. the summary counts and the mean actual dwell match what happened

Deliberately mirrors the class logic: this is the "verify it is actually non-zero" step the project keeps
needing, applied to the dwell instead of to a sensor.

Run: python as2w_long_dwell_curriculum_selftest.py
"""
from __future__ import annotations

import torch

DWELL_MIN, DWELL_MAX = 8.0, 12.0
PARENT_LO, PARENT_HI = 1.8, 3.0
STEP_DT = 0.02
LONG_TURN_FRAC = 0.375
LONG_ARC_FRAC = 0.40


class Mock:
    """A minimal stand-in that replicates the dwell bookkeeping."""

    def __init__(self, n=4096, seed=0):
        torch.manual_seed(seed)
        self.num_envs = n
        self.step_dt = STEP_DT
        self._frac = torch.tensor([0.40, 0.25, 0.10, 0.10, 0.15])
        self._frac = self._frac / self._frac.sum()
        self.cat = torch.full((n,), -1, dtype=torch.long)
        self.is_long = torch.zeros(n, dtype=torch.bool)
        self.planned = torch.zeros(n)
        self.elapsed = torch.zeros(n)
        self.active = torch.zeros(n, dtype=torch.bool)
        self.time_left = torch.full((n,), 2.4)          # parent cadence 1.8-3 s
        self.started_n = self.completed_n = self.truncated_n = 0
        self.dwell_sum = 0.0
        self.dwell_n = 0
        self.events = []

    def resample(self, ids):
        n = ids.shape[0]
        cat = torch.multinomial(self._frac, n, replacement=True)
        r = torch.rand(n)
        is_long = ((cat == 0) & (r < LONG_TURN_FRAC)) | ((cat == 1) & (r < LONG_ARC_FRAC))
        self.cat[ids] = cat
        self.is_long[ids] = is_long
        self._close(ids, "resampled")
        plan = DWELL_MIN + torch.rand(n) * (DWELL_MAX - DWELL_MIN)
        plan = torch.where(is_long, plan, torch.zeros_like(plan))
        self.planned[ids] = plan
        self.elapsed[ids] = 0.0
        self.active[ids] = is_long
        self.started_n += int(is_long.sum())
        for e in ids[is_long].tolist():
            self.events.append({"event": "LONG_TURN_START", "env": int(e),
                                 "planned": float(self.planned[int(e)])})
        # a SHORT sample must be reset to a FRESH parent-cadence value, never keep an old long one
        fresh = PARENT_LO + torch.rand(n) * (PARENT_HI - PARENT_LO)
        self.time_left[ids] = torch.where(is_long, plan, fresh)
        return cat, is_long, plan

    def _close(self, ids, reason):
        still = ids[self.active[ids]]
        for j in still.tolist():
            j = int(j)
            el = float(self.elapsed[j]); pl = float(self.planned[j])
            done = el >= pl * 0.95
            self.events.append({"event": "LONG_TURN_END", "env": j, "actual": el,
                                 "planned": pl, "completed": done, "reason": reason})
            self.dwell_sum += el; self.dwell_n += 1
            if done:
                self.completed_n += 1
            else:
                self.truncated_n += 1
            self.active[j] = False

    def tick(self, steps):
        """Advance `steps` control steps, closing holds whose plan elapsed."""
        for _ in range(steps):
            self.elapsed[self.active] += self.step_dt
            done = self.active & (self.elapsed >= self.planned)
            if bool(done.any()):
                self._close(torch.arange(self.num_envs)[done], "completed")


def main() -> int:
    ok = True
    m = Mock(n=4096, seed=1)
    ids = torch.arange(m.num_envs)
    # first resample is meaningless (everything closes immediately); resample twice, test the second
    m.resample(ids)
    m.cat.zero_(); m.is_long.zero_(); m.active.zero_()
    m.started_n = m.completed_n = m.truncated_n = 0; m.dwell_sum = 0.0; m.dwell_n = 0
    m.events.clear()

    cat, is_long, plan = m.resample(ids)
    n_turn = int((cat == 0).sum()); n_arc = int((cat == 1).sum())
    n_long_turn = int(((cat == 0) & is_long).sum())
    n_long_arc = int(((cat == 1) & is_long).sum())

    print("=== 1) long sub-fraction ===")
    ft = n_long_turn / max(1, n_turn); fa = n_long_arc / max(1, n_arc)
    print("  pure_turn n=%d long=%d -> %.3f (target %.3f)" % (n_turn, n_long_turn, ft, LONG_TURN_FRAC))
    print("  arc       n=%d long=%d -> %.3f (target %.3f)" % (n_arc, n_long_arc, fa, LONG_ARC_FRAC))
    if abs(ft - LONG_TURN_FRAC) > 0.06 or abs(fa - LONG_ARC_FRAC) > 0.07:
        print("  FAIL sub-fraction out of tolerance"); ok = False
    else:
        print("  PASS")

    print("\n=== 2) planned dwell inside [8,12] for long, zero for short ===")
    lp = plan[is_long]; sp = plan[~is_long]
    print("  long : min=%.3f max=%.3f mean=%.3f n=%d" % (lp.min(), lp.max(), lp.mean(), len(lp)))
    print("  short: max=%.3f (must be 0)" % (sp.max() if len(sp) else 0.0))
    if lp.min() < DWELL_MIN - 1e-6 or lp.max() > DWELL_MAX + 1e-6:
        print("  FAIL planned dwell outside range"); ok = False
    elif len(sp) and float(sp.max()) != 0.0:
        print("  FAIL a short sample got a planned dwell"); ok = False
    else:
        print("  PASS")

    print("\n=== 3) time_left receives the planned dwell (this is what makes the hold happen) ===")
    tl_long = m.time_left[is_long]; tl_short = m.time_left[~is_long]
    print("  long  time_left min=%.3f max=%.3f" % (tl_long.min(), tl_long.max()))
    print("  short time_left max=%.3f (parent cadence 1.8-3.0)" % (tl_short.max() if len(tl_short) else 0.0))
    if not torch.allclose(tl_long, lp):
        print("  FAIL time_left was not set to the plan for long samples"); ok = False
    elif len(tl_short) and float(tl_short.max()) > PARENT_HI + 0.01:
        print("  FAIL a short sample's time_left was not reset to the parent cadence"); ok = False
    elif len(tl_short) and float(tl_short.min()) < PARENT_LO - 0.01:
        print("  FAIL a short sample's time_left fell below the parent range"); ok = False
    else:
        print("  PASS")

    print("\n=== 4) a long hold completes at its planned dwell ===")
    longest = float(lp.max())
    m.tick(int(longest / STEP_DT) + 2)
    n_completed = m.completed_n
    print("  after %.2f s of ticking: completed=%d truncated=%d" % (longest + 0.04, m.completed_n, m.truncated_n))
    if n_completed < len(lp) * 0.99:
        print("  FAIL not all long holds closed as completed"); ok = False
    else:
        print("  PASS")

    print("\n=== 5) a hold cut short is TRUNCATED, not dropped ===")
    m2 = Mock(n=256, seed=7)
    m2.resample(torch.arange(m2.num_envs))
    m2.cat.zero_(); m2.is_long.zero_(); m2.active.zero_()
    m2.started_n = m2.completed_n = m2.truncated_n = 0; m2.dwell_sum = 0.0; m2.dwell_n = 0
    m2.events.clear()
    cat2, is_long2, plan2 = m2.resample(torch.arange(m2.num_envs))
    m2.tick(5)                                  # 0.1 s only
    m2.resample(torch.arange(m2.num_envs))      # interrupts
    trunc = [e for e in m2.events if e["event"] == "LONG_TURN_END" and not e["completed"]]
    print("  truncated events=%d (must be > 0)" % len(trunc))
    if len(trunc) == 0:
        print("  FAIL a cut-short hold produced no TRUNCATED event"); ok = False
    else:
        e = trunc[0]
        print("    sample: actual=%.2f planned=%.2f reason=%s" % (e["actual"], e["planned"], e["reason"]))
        print("  PASS")

    print("\n=== 6) summary arithmetic ===")
    mean_dwell = m.dwell_sum / max(1, m.dwell_n)
    print("  started=%d completed=%d truncated=%d mean_actual=%.2f s (must be inside [8,12])"
          % (m.started_n, m.completed_n, m.truncated_n, mean_dwell))
    if not (DWELL_MIN - 0.2 <= mean_dwell <= DWELL_MAX + 0.2):
        print("  FAIL mean actual dwell outside the planned range"); ok = False
    else:
        print("  PASS")

    print("\nLONG_DWELL_SELFTEST_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
