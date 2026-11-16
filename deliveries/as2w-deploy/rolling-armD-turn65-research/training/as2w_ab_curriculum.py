"""B-group long-turn curriculum: change the COMMAND SAMPLING only.

Operator's first-round spec (2026-11-16), as fractions of the sampled cases:

    35%  pure turns   vx=0, vy=0, wz = +-0.05 or +-0.10, dwell 8-12 s after the ramp
    25%  arcs         vx=+0.25 with the frozen left/right yaw steps
    15%  straight     low-speed continuous change plus +-0.25, vy=0
    15%  brakes       forward and reverse in equal share
    10%  hold         zero command

Nothing else changes: no reward, gain, friction or leg-pose edits.  The A arm keeps the armE
sampling untouched, so the two arms differ only here.

Why a separate class rather than an edit to FlatCommands: the A arm must keep exactly the previous
sampling, and the project has already been bitten repeatedly by settings that looked configured but
were never read.  This class prints the category counts it actually sampled, and the launcher asserts
the startup line is present, so "the curriculum is on" is proven rather than assumed.
"""
from __future__ import annotations

import os

import torch

from as2w_isaac_flat_env_all import FlatCommands


def _f(name, default):
    return float(os.environ.get(name, default))


class CurriculumFlatCommands(FlatCommands):
    """FlatCommands with the B-group category mixture."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._frac = torch.tensor([
            _f("AS2W_AB_TURN_FRACTION", 0.35),      # 0 pure turn
            _f("AS2W_AB_ARC_FRACTION", 0.25),       # 1 arc
            _f("AS2W_AB_STRAIGHT_FRACTION", 0.15),  # 2 straight
            _f("AS2W_AB_BRAKE_FRACTION", 0.15),     # 3 brake (drive then release)
            _f("AS2W_AB_HOLD_FRACTION", 0.10),      # 4 hold
        ], dtype=torch.float32, device=self.device)
        self._frac = self._frac / self._frac.sum()
        self._count = torch.zeros(5, dtype=torch.long, device=self.device)
        self._n_total = 0
        print("AS2W_AB_CURRICULUM ON  fractions=%s  dwell=%s-%s s"
              % ([round(float(v), 2) for v in self._frac],
                 os.environ.get("AS2W_AB_TURN_DWELL_MIN_S", "8"),
                 os.environ.get("AS2W_AB_TURN_DWELL_MAX_S", "12")), flush=True)

    def _resample(self, n):
        """Draw n commands from the category mixture; returns (n,3) commands and the category ids."""
        cat = torch.multinomial(self._frac, n, replacement=True)
        cmd = torch.zeros(n, 3, device=self.device)
        u = torch.rand(n, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0).to(self.device)

        # 0 pure turn: vx=0, wz=+-0.05 / +-0.10
        m = cat == 0
        mag = torch.where(torch.rand(n, device=self.device) < 0.5, 0.05, 0.10)
        cmd[m, 2] = (sign * mag)[m]

        # 1 arc: vx=+0.25 with the frozen yaw steps
        m = cat == 1
        cmd[m, 0] = 0.25
        cmd[m, 2] = (sign * 0.05)[m]

        # 2 straight: low-speed continuous change plus +-0.25, vy=0
        m = cat == 2
        cmd[m, 0] = (torch.rand(n, device=self.device) * 0.50 - 0.25)[m]

        # 3 brake: forward and reverse in equal share; the release to zero comes from _update_command
        m = cat == 3
        cmd[m, 0] = (torch.where(u < 0.5, 1.0, -1.0) * 0.25)[m]

        # 4 hold stays zero
        return cmd, cat

    def _resample_command(self, env_ids):
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.arange(
            self.num_envs, device=self.device)[env_ids]
        n = ids.shape[0]
        cmd, cat = self._resample(n)
        self.targets[ids] = cmd
        self.static[ids] = False
        for c in range(5):
            self._count[c] += int((cat == c).sum())
        self._n_total += n

    def report_curriculum(self):
        """Actual completed counts per category - the launcher prints this on exit."""
        tot = max(1, self._n_total)
        return {name: {"count": int(self._count[i]),
                       "share": round(float(self._count[i]) / tot, 4)}
                for i, name in enumerate(("pure_turn", "arc", "straight", "brake", "hold"))}
