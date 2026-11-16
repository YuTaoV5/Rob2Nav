#!/usr/bin/env python3
"""Thin environment adapter that installs the B turn curriculum on the frozen R4000 command class.

Design constraint: the adapter must not subclass the R4000 command class at import time, because that
class lives in the run snapshot which imports Isaac.  So it is built by ``make_b_class`` at run time, from
the snapshot's own ``RollingSpeedCommands``; the snapshot files themselves are never rewritten, which is
what lets the self-check prove the action wrapper and observation are byte-identical to the R4000 run.

All sampling and dwell decisions are delegated to ``turn_schedule.TurnSchedule`` (pure Python, exercised
offline).  This class only maps those decisions onto per-environment tensors.

Mechanism
---------
``_resample_command``  keeps the R4000 behaviour for the base and park branches, then overwrites the
                       command, the dwell (``time_left``, which the base command term already consumes when
                       it holds a segment) and the release bookkeeping for the mixed branch only.
``_update_command``    forces the command to zero while a release segment is active, so the release is an
                       explicit zero-command window rather than an emergent side effect of the brake
                       reward, and then lets the R4000 command filter run unchanged.
"""
from __future__ import annotations

import os

import torch

from turn_schedule import CATEGORIES, TurnSchedule

MIXED_LO = 0.40          # TeleopCommands: kind in [.40, .80) is the mixed/curriculum branch
MIXED_HI = 0.80


def make_b_class(base_cls):
    """Return the B command class derived from the snapshot's RollingSpeedCommands."""

    class RollingSpeedCommandsB(base_cls):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            seed = int(os.environ.get("AS2W_TURN_B_SEED", "311"))
            self._ts = TurnSchedule(seed)
            n = self.num_envs
            self._pending_release = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._release_left = torch.zeros(n, device=self.device)
            self._b_counts = {c: 0 for c in CATEGORIES + ("release",)}
            self._b_dwell_min = {}
            self._b_dwell_max = {}
            print("AS2W_TURN_CURRICULUM_B ON seed=%d shares=%s arc_vx=%.3f arc_wz=%s "
                  "turn_long_dwell=%s release=%s"
                  % (seed, [round(float(self._ts.p["share_" + c]), 3) for c in CATEGORIES],
                     float(self._ts.p["arc_vx_m_s"]),
                     list(self._ts.p["arc_wz_choices"]),
                     list(self._ts.p["turn_long_dwell_s"]), list(self._ts.p["release_s"])),
                  flush=True)

        def _ids(self, env_ids):
            if env_ids is None:
                return torch.arange(self.num_envs, device=self.device)
            if isinstance(env_ids, torch.Tensor):
                return env_ids
            return torch.arange(self.num_envs, device=self.device)[env_ids]

        def _record(self, seg):
            key = "release" if seg["is_release"] else seg["category"]
            self._b_counts[key] = self._b_counts.get(key, 0) + 1
            d = float(seg["dwell_s"])
            self._b_dwell_min[key] = min(self._b_dwell_min.get(key, d), d)
            self._b_dwell_max[key] = max(self._b_dwell_max.get(key, d), d)

        def _resample_command(self, env_ids):
            # R4000 semantics first: base/park targets, ramp bookkeeping, reversal and zero draws.
            super()._resample_command(env_ids)
            ids = self._ids(env_ids)
            mixed = (self.kind[ids] >= MIXED_LO) & (self.kind[ids] < MIXED_HI)
            if not bool(mixed.any()):
                return
            for i in ids[mixed].tolist():
                seg = self._ts.draw_segment(bool(self._pending_release[i].item()))
                self.targets[i, 0] = float(seg["vx"])
                self.targets[i, 1] = 0.0
                self.targets[i, 2] = float(seg["wz"])
                self.time_left[i] = float(seg["dwell_s"])
                self._pending_release[i] = bool(seg["next_pending_release"])
                self._release_left[i] = float(seg["dwell_s"]) if seg["is_release"] else 0.0
                self._record(seg)

        def _update_command(self):
            rel = getattr(self, "_release_left", None)
            if rel is not None:
                active = rel > 0
                if bool(active.any()):
                    self.targets[active] = 0.0
                    self.ramp[active] = 0.0
                    rel[active] -= self._env.step_dt
            super()._update_command()

        def report_turn_curriculum(self):
            """Actual installed segment counts and dwell extremes, for the launcher to print on exit."""
            tot = max(1, sum(self._b_counts.values()))
            return {
                "installed_segments": dict(self._b_counts),
                "share": {k: round(v / tot, 4) for k, v in self._b_counts.items()},
                "dwell_min_s": {k: round(v, 3) for k, v in self._b_dwell_min.items()},
                "dwell_max_s": {k: round(v, 3) for k, v in self._b_dwell_max.items()},
                "pending_release_envs": int(self._pending_release.sum().item()),
            }

    RollingSpeedCommandsB.__name__ = "RollingSpeedCommandsB"
    return RollingSpeedCommandsB
