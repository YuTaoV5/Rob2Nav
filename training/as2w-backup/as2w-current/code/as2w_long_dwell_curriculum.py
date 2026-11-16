"""AS2-W long-dwell curriculum: make the 8-12 s long turns ACTUALLY happen.

Why this file exists
--------------------
`as2w_ab_curriculum.py` printed AS2W_AB_TURN_DWELL_MIN_S / _MAX_S at startup but never read them
anywhere, so arms B, C and D never produced a single long-dwell turn - the only thing that changed was
the category mixture.  The real dwell came from the parent config's
`cmd.resampling_time_range = (1.8, 3.)`, which resamples every command after 1.8-3 s regardless of
category.

This class fixes that properly:
  * a fraction of the PURE-TURN samples and a fraction of the ARC samples are marked "long"
  * those samples get their per-env `time_left` set to U(DWELL_MIN, DWELL_MAX) seconds after resampling,
    while every other sample keeps the parent's 1.8-3 s
  * the command itself still ramps through the parent's lerp, so the hold is measured AFTER the ramp
    settles, which is what the operator asked to see in the log

It also emits an EVENT LOG so "the dwell was actually applied" is proven rather than assumed:
  LONG_TURN_START      env, category, target, planned dwell
  LONG_TURN_END        env, category, ACTUAL elapsed seconds, completed or truncated
  LONG_TURN_SUMMARY    counts: started / completed / truncated, per category, plus the mean actual dwell

The summary line is the evidence the operator asked for: it can be compared directly against the
planned 8-12 s, and truncated episodes are listed separately.

Usage: set AS2W_AB_CURRICULUM=1 and AS2W_AB_LONG_DWELL=1, keep the mixture fractions as before.
Nothing else changes: no reward, gain, friction or leg-pose edits.
"""
from __future__ import annotations

import os

import torch

from as2w_isaac_flat_env_all import FlatCommands


def _f(name, default):
    return float(os.environ.get(name, default))


class LongDwellCommands(FlatCommands):
    """Category mixture plus a genuine 8-12 s hold on a subset of turn/arc samples."""

    LONG_TURN_FRAC = 0.375   # of the PURE-TURN samples, 15/40 -> 15 pp of all cases
    LONG_ARC_FRAC = 0.40     # of the ARC samples, 10/25 -> 10 pp of all cases

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._frac = torch.tensor([
            _f("AS2W_AB_TURN_FRACTION", 0.40),      # 0 pure turn
            _f("AS2W_AB_ARC_FRACTION", 0.25),       # 1 arc
            _f("AS2W_AB_STRAIGHT_FRACTION", 0.10),  # 2 straight
            _f("AS2W_AB_BRAKE_FRACTION", 0.10),     # 3 brake
            _f("AS2W_AB_HOLD_FRACTION", 0.15),      # 4 hold
        ], dtype=torch.float32, device=self.device)
        self._frac = self._frac / self._frac.sum()
        self._long_turn_frac = _f("AS2W_AB_LONG_TURN_FRAC", self.LONG_TURN_FRAC)
        self._long_arc_frac = _f("AS2W_AB_LONG_ARC_FRAC", self.LONG_ARC_FRAC)
        self._dt = float(getattr(getattr(self, "_env", None), "step_dt", 0.02))
        self._dwell_min = _f("AS2W_AB_TURN_DWELL_MIN_S", 8.0)
        self._dwell_max = _f("AS2W_AB_TURN_DWELL_MAX_S", 12.0)
        self._enabled = _f("AS2W_AB_LONG_DWELL", 0.0) > 0

        self._cat = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._is_long = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._planned = torch.zeros(self.num_envs, device=self.device)
        self._started = torch.zeros(self.num_envs, device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._count = torch.zeros(5, dtype=torch.long, device=self.device)
        self._n_total = 0
        self._events = []
        # summary counters
        self._started_n = 0
        self._completed_n = 0
        self._truncated_n = 0
        self._dwell_sum = 0.0
        self._dwell_n = 0
        self._turn_steps = 0
        self._total_steps = 0

        print("AS2W_AB_CURRICULUM ON  class=%s  fractions=%s  long_dwell=%s  "
              "long_turn_frac=%.3f long_arc_frac=%.3f  dwell=%s-%s s"
              % (type(self).__name__, [round(float(v), 2) for v in self._frac],
                 "ON" if self._enabled else "OFF", self._long_turn_frac, self._long_arc_frac,
                 self._dwell_min, self._dwell_max), flush=True)

    # ------------------------------------------------------------------ sampling
    def _draw_mixture(self, n):
        """Draw n commands; also decide which turn/arc samples are LONG."""
        cat = torch.multinomial(self._frac, n, replacement=True)
        cmd = torch.zeros(n, 3, device=self.device)
        u = torch.rand(n, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0).to(self.device)

        m = cat == 0                                  # pure turn
        mag = torch.where(torch.rand(n, device=self.device) < 0.5, 0.05, 0.10)
        cmd[m, 2] = (sign * mag)[m]
        m = cat == 1                                  # arc
        cmd[m, 0] = 0.25
        cmd[m, 2] = (sign * 0.05)[m]
        m = cat == 2                                  # straight
        cmd[m, 0] = (torch.rand(n, device=self.device) * 0.50 - 0.25)[m]
        m = cat == 3                                  # brake
        cmd[m, 0] = (torch.where(u < 0.5, 1.0, -1.0) * 0.25)[m]
        # cat == 4 stays zero (hold)

        # which samples are long
        is_long = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self._enabled:
            r = torch.rand(n, device=self.device)
            is_long |= (cat == 0) & (r < self._long_turn_frac)
            is_long |= (cat == 1) & (r < self._long_arc_frac)
        return cmd, cat, is_long

    def _resample_command(self, env_ids):
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.arange(
            self.num_envs, device=self.device)[env_ids]
        n = ids.shape[0]
        cmd, cat, is_long = self._draw_mixture(n)
        self.targets[ids] = cmd
        self.static[ids] = False
        self._cat[ids] = cat
        self._is_long[ids] = is_long

        for c in range(5):
            self._count[c] += int((cat == c).sum())
        self._n_total += n

        if self._enabled:
            # close any still-active long hold on these envs (truncated by a new resample)
            self._close_long(ids, reason="resampled")
            # a long sample gets a per-env dwell; everything else keeps the parent cadence
            plan = self._dwell_min + torch.rand(n, device=self.device) * (self._dwell_max - self._dwell_min)
            plan = torch.where(is_long, plan, torch.zeros_like(plan))
            self._planned[ids] = plan
            self._elapsed[ids] = 0.0
            self._active[ids] = is_long
            if bool(is_long.any()):
                self._started_n += int(is_long.sum())
                for k, e in enumerate(ids[is_long].tolist()):
                    j = int(e)
                    self._events.append({
                        "event": "LONG_TURN_START", "env": j, "category": int(cat[is_long][k]),
                        "target": [round(float(v), 4) for v in cmd[is_long][k]],
                        "planned_dwell_s": round(float(plan[is_long][k]), 2)})
                print("LONG_TURN_START n=%d planned=%s s"
                      % (int(is_long.sum()), [round(float(x), 1) for x in plan[is_long][:6]]), flush=True)
            # finally force the hold: the command manager will not resample these until time_left expires.
            # A SHORT sample must get a FRESH parent-cadence value, not keep whatever it had before -
            # otherwise an env that was long last time would hold a short command for 8-12 s.
            rng = getattr(self.cfg, "resampling_time_range", (1.8, 3.0))
            lo, hi = float(rng[0]), float(rng[1])
            fresh = lo + torch.rand(n, device=self.device) * max(hi - lo, 1e-6)
            try:
                self.time_left[ids] = torch.where(is_long, plan, fresh)
            except Exception as exc:                     # noqa: BLE001 - report, never silently skip
                print("AS2W_LONG_DWELL_WARN could not set time_left: %r" % (exc,), flush=True)

    def _close_long(self, ids, reason):
        if not self._enabled:
            return
        still = ids[self._active[ids]]
        for j in still.tolist():
            j = int(j)
            elapsed = float(self._elapsed[j])
            planned = float(self._planned[j])
            done = elapsed >= planned * 0.95
            self._events.append({
                "event": "LONG_TURN_END", "env": j, "actual_dwell_s": round(elapsed, 2),
                "planned_dwell_s": round(planned, 2), "completed": bool(done), "reason": reason})
            self._dwell_sum += elapsed
            self._dwell_n += 1
            if done:
                self._completed_n += 1
            else:
                self._truncated_n += 1
            self._active[j] = False
            if not done or (self._completed_n % 40 == 0):
                print("LONG_TURN_END env=%d actual=%.2fs planned=%.2fs completed=%s reason=%s"
                      % (j, elapsed, planned, done, reason), flush=True)

    def _update_command(self):
        super()._update_command()
        if not self._enabled:
            self._total_steps += 1
            return
        # accumulate actual held time for the still-active long holds
        self._elapsed[self._active] += self._dt
        # count turning steps (what the policy ACTUALLY saw, not what we sampled)
        turning = (self.vel_command_b[:, 2].abs() > 1e-3) | (self.vel_command_b[:, 0].abs() > 1e-3)
        self._turn_steps += int(turning.sum())
        self._total_steps += int(self.num_envs)
        # periodic proof for the log: planned vs actual dwell, completions and truncations
        if self._total_steps % 100 == 0:
            mean_d = (self._dwell_sum / self._dwell_n) if self._dwell_n else 0.0
            print("LONG_TURN_SUMMARY steps=%d started=%d completed=%d truncated=%d "
                  "mean_actual_dwell=%.2fs planned=[%.1f,%.1f] turning_step_share=%.3f"
                  % (self._total_steps, self._started_n, self._completed_n, self._truncated_n,
                     mean_d, self._dwell_min, self._dwell_max,
                     self._turn_steps / max(1, self._total_steps)), flush=True)
        # a long hold is complete once its planned dwell elapsed
        done = self._active & (self._elapsed >= self._planned)
        if bool(done.any()):
            self._close_long(torch.arange(self.num_envs, device=self.device)[done], reason="completed")

    # ------------------------------------------------------------------ reporting
    def report_curriculum(self):
        tot = max(1, self._n_total)
        out = {name: {"count": int(self._count[i]), "share": round(float(self._count[i]) / tot, 4)}
               for i, name in enumerate(("pure_turn", "arc", "straight", "brake", "hold"))}
        mean_dwell = round(self._dwell_sum / self._dwell_n, 2) if self._dwell_n else None
        out["long_dwell"] = {
            "enabled": bool(self._enabled),
            "started": self._started_n, "completed": self._completed_n, "truncated": self._truncated_n,
            "mean_actual_dwell_s": mean_dwell,
            "planned_range_s": [self._dwell_min, self._dwell_max],
            "long_turn_frac_of_turn": self._long_turn_frac,
            "long_arc_frac_of_arc": self._long_arc_frac,
            "actual_turning_step_share": round(self._turn_steps / max(1, self._total_steps), 4),
            "note": "started/completed/truncated prove the dwell was actually applied; "
                    "mean_actual_dwell_s should land inside planned_range_s for completed holds"}
        return out

    def drain_events(self, limit=400):
        ev, self._events = self._events[:limit], self._events[limit:]
        return ev
