#!/usr/bin/env python3
"""B-group turn curriculum: the pure, offline-testable decision core.

Why this module is separate from the environment
------------------------------------------------
The operator requires the offline self-check to prove *real* long-turn events, the release segment that
follows them, and left/right symmetry.  A check that re-implements the sampler proves nothing about the
code that trains.  So every sampling and dwell decision lives here, in a module that imports nothing from
Isaac, and the environment adapter (b_adapter.py) is a thin id-indexed shim that calls
``TurnSchedule.draw_segment``.  The self-check exercises this module directly, and separately verifies by
AST inspection that the adapter's resample and update paths call it.

What B changes relative to the frozen R4000 stage (see configs/ab_curriculum.json)
----------------------------------------------------------------------------------
R4000's own sampler (as2w_isaac_speed_range.py, ``RollingSpeedCommands.sample_motion`` under
``--rolling-low-speed-smooth``) draws, inside the 40 percent "mixed" branch of segments:
    pure_turn 0.40 : vx = 0,        wz = sign * (0.035 + 0.085*u)
    arc       0.15 : vx = 0.05,     wz = sign * (0.035 + 0.085*u)
    straight  0.45 : vx = sign*(0.03 + 0.17*u), wz = 0
and every segment, of every category, lasts U(1.1, 3.4) s because the dwell is the command term's global
``resampling_time_range``.  There is no sustained turning and no explicit release.

B keeps the same envelope and the same base/park branches and changes only this mixed branch:
    pure_turn 0.30 : as R4000, U(1.1, 3.4) s
    turn_long 0.15 : as R4000 in command, but U(8, 12) s, always followed by a release
    arc       0.25 : vx = 0.20 (in-range stage maximum), wz = sign * {0.02, 0.04, 0.06}, bidirectional
    straight  0.30 : as R4000
    release         : zero command for U(3.0, 3.5) s, inserted after every turn_long

Invariants enforced here so the self-check can assert them
    * every command stays inside the frozen stage envelope |vx| <= 0.20 m/s, |wz| <= 0.12 rad/s
    * sign is balanced 50/50 for every directional category (left/right mirror symmetry)
    * a turn_long is ALWAYS followed by a release, and a release never carries a command

Nothing in this module touches the action wrapper, observation, rewards, PPO or the control contract.
"""
from __future__ import annotations

import random

# Frozen R4000 stage envelope.  These are the bounds the continuation must not widen.
VX_CAP_M_S = 0.20
WZ_CAP_RAD_S = 0.12
VX_FLOOR_M_S = 0.03
WZ_FLOOR_RAD_S = 0.035

# Dwell ranges.  The R4000 stage's global resampling_time_range is (1.1, 3.4); B keeps it for every
# category except the new sustained turn.
DWELL_SHORT_S = (1.1, 3.4)
DWELL_LONG_S = (8.0, 12.0)
RELEASE_S = (3.0, 3.5)

CATEGORIES = ("pure_turn", "turn_long", "arc", "straight")

DEFAULTS = {
    # shares inside the 40 percent mixed branch; must sum to 1
    "share_pure_turn": 0.30,
    "share_turn_long": 0.15,
    "share_arc": 0.25,
    "share_straight": 0.30,
    # arc command set: small bidirectional yaw magnitudes, deliberately inside the frozen envelope
    "arc_vx_m_s": 0.20,
    "arc_wz_choices": (0.02, 0.04, 0.06),
    # sustained-turn dwell and the release that must follow it
    "turn_long_dwell_s": DWELL_LONG_S,
    "release_s": RELEASE_S,
}


def _u(rng, lo, hi):
    return lo + (hi - lo) * rng.random()


def _sign(rng):
    return -1.0 if rng.random() < 0.5 else 1.0


class TurnSchedule:
    """Draws B-machine segments.  One instance owns one RNG stream."""

    def __init__(self, seed: int, params: dict | None = None):
        self.rng = random.Random(int(seed))
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)

    # ---- category ----------------------------------------------------------------
    def draw_category(self) -> str:
        p = self.p
        shares = [(c, float(p["share_" + c])) for c in CATEGORIES]
        total = sum(w for _, w in shares)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("B category shares must sum to 1, got %r" % (total,))
        r = self.rng.random() * total
        acc = 0.0
        for name, w in shares:
            acc += w
            if r <= acc:
                return name
        return shares[-1][0]

    # ---- command -----------------------------------------------------------------
    def draw_command(self, category: str) -> tuple[float, float]:
        """Return (vx_m_s, wz_rad_s) for a category, drawn inside the frozen envelope."""
        rng = self.rng
        if category in ("pure_turn", "turn_long"):
            return 0.0, _sign(rng) * _u(rng, WZ_FLOOR_RAD_S, WZ_CAP_RAD_S)
        if category == "arc":
            vx = float(self.p["arc_vx_m_s"])
            if vx > VX_CAP_M_S + 1e-9:
                raise ValueError("arc vx %.3f exceeds the frozen R4000 stage cap %.3f"
                                 % (vx, VX_CAP_M_S))
            magnitude = float(self.rng.choice(list(self.p["arc_wz_choices"])))
            if magnitude > WZ_CAP_RAD_S + 1e-9:
                raise ValueError("arc wz %.3f exceeds the frozen cap %.3f" % (magnitude, WZ_CAP_RAD_S))
            return vx, _sign(rng) * magnitude
        if category == "straight":
            return _sign(rng) * _u(rng, VX_FLOOR_M_S, VX_CAP_M_S), 0.0
        raise ValueError("unknown category %r" % (category,))

    # ---- segment -----------------------------------------------------------------
    def draw_segment(self, pending_release: bool = False, remaining_s: float | None = None) -> dict:
        """Draw one segment.

        ``pending_release`` is the caller's per-environment state: True means the previous segment was a
        sustained turn and this one must therefore be the release.  The returned dict carries everything
        the adapter needs, including the dwell the environment should actually use.
        """
        if pending_release:
            dwell = _u(self.rng, *self.p["release_s"])
            ready = remaining_s is None or dwell <= remaining_s
            if not ready:
                dwelling = max(remaining_s, 0.02)
            else:
                dwelling = dwell
            return {
                "category": "release",
                "is_release": True,
                "is_turn_long": False,
                "vx": 0.0,
                "wz": 0.0,
                "dwell_s": dwelling,
                "requested_dwell_s": dwell,
                "truncated_by_episode_end": not ready,
                "clipped_at_episode_end": False,
                "downgraded_from": None,
                "next_pending_release": False,
            }
        category = self.draw_category()
        vx, wz = self.draw_command(category)
        is_long = category == "turn_long"
        dwell = _u(self.rng, *self.p["turn_long_dwell_s"]) if is_long else _u(self.rng, *DWELL_SHORT_S)
        downgraded_from = None
        # A sustained turn that cannot also fit its release inside the remaining episode time would be
        # silently truncated, so it is downgraded to a short turn instead.  This keeps requested dwell
        # equal to installed dwell and is reported, not hidden.
        if is_long and remaining_s is not None and dwell + self.p["release_s"][1] > remaining_s:
            downgraded_from = "turn_long"
            category, is_long = "pure_turn", False
            vx, wz = 0.0, _sign(self.rng) * _u(self.rng, WZ_FLOOR_RAD_S, WZ_CAP_RAD_S)
            dwell = min(_u(self.rng, *DWELL_SHORT_S), max(0.02, remaining_s))
        clipped = False
        if remaining_s is not None and dwell > remaining_s:
            clipped = True
            dwell = max(0.02, remaining_s)
        return {
            "category": category,
            "is_release": False,
            "is_turn_long": is_long,
            "vx": vx,
            "wz": wz,
            "dwell_s": dwell,
            "requested_dwell_s": dwell,
            "truncated_by_episode_end": False,
            "clipped_at_episode_end": clipped,
            "downgraded_from": downgraded_from,
            "next_pending_release": is_long,
        }

    # ---- convenience for the offline self-check ----------------------------------
    def simulate(self, n_steps: int, dt: float = 0.02, step_to_segment_s: float = 0.0) -> dict:
        """Walk the segment machine and return segment/step level statistics.

        The environment consumes one segment per resample and holds it for ``dwell_s`` policy steps.  This
        replica advances the same state machine, so the statistics are about the same decisions that the
        adapter will install.
        """
        pending = False
        segs = []
        release_after_long = []
        prev = None
        for _ in range(int(n_steps)):
            seg = self.draw_segment(pending)
            if prev is not None and prev["is_turn_long"]:
                release_after_long.append(seg["is_release"])
            segs.append(seg)
            pending = seg["next_pending_release"]
            prev = seg
        cats = {}
        dwells = {}
        for s in segs:
            cats[s["category"]] = cats.get(s["category"], 0) + 1
            dwells.setdefault(s["category"], []).append(s["dwell_s"])
        steps = {}
        for s in segs:
            n = max(1, int(round(s["dwell_s"] / dt)))
            steps[s["category"]] = steps.get(s["category"], 0) + n
        total_steps = sum(steps.values())
        return {
            "segments": len(segs),
            "category_segment_share": {k: v / len(segs) for k, v in cats.items()},
            "category_step_share": {k: v / total_steps for k, v in steps.items()},
            "dwell_range": {k: [min(v), max(v)] for k, v in dwells.items()},
            "long_turns": cats.get("turn_long", 0),
            "releases": cats.get("release", 0),
            "long_turn_followed_by_release_rate": (
                sum(1 for x in release_after_long if x) / len(release_after_long)
                if release_after_long else None),
            "sign_balance": {
                c: self._sign_balance([s["wz"] for s in segs if s["category"] == c])
                for c in ("pure_turn", "turn_long", "arc")},
            "envelope_ok": all(abs(s["vx"]) <= VX_CAP_M_S + 1e-9 and abs(s["wz"]) <= WZ_CAP_RAD_S + 1e-9
                               for s in segs),
            "release_has_zero_command": all(s["vx"] == 0.0 and s["wz"] == 0.0
                                           for s in segs if s["is_release"]),
        }

    @staticmethod
    def _sign_balance(values):
        if not values:
            return None
        pos = sum(1 for v in values if v > 0)
        neg = sum(1 for v in values if v < 0)
        return (pos - neg) / float(len(values))


def r4000_reference_sampler(seed: int, n: int) -> dict:
    """Reproduce the frozen R4000 mixed-branch sampler for side-by-side reporting only.

    This mirrors ``RollingSpeedCommands.sample_motion`` exactly as written in the run snapshot; it is used
    to report the A distribution next to B's and is never installed into any environment.
    """
    rng = random.Random(int(seed))
    cats = {"pure_turn": 0, "arc": 0, "straight": 0}
    wzs, vxs = [], []
    for _ in range(int(n)):
        draw = [rng.random(), rng.random(), rng.random()]
        sign = -1.0 if draw[0] < .5 else 1.0
        turn = draw[2] < .40
        arc = (.40 <= draw[2] < .55)
        if turn:
            cat, vx, wz = "pure_turn", 0.0, sign * (.035 + .085 * draw[1])
        elif arc:
            cat, vx, wz = "arc", .05, sign * (.035 + .085 * draw[1])
        else:
            cat, vx, wz = "straight", sign * (.03 + .17 * draw[1]), 0.0
        cats[cat] += 1
        vxs.append(vx)
        if wz:
            wzs.append(wz)
    n_cmd = max(1, len(wzs))
    return {
        "category_segment_share": {k: v / n for k, v in cats.items()},
        "wz_magnitude_range": [min(abs(w) for w in wzs), max(abs(w) for w in wzs)] if wzs else None,
        "wz_sign_balance": (sum(1 for w in wzs if w > 0) - sum(1 for w in wzs if w < 0)) / n_cmd,
        "vx_range": [min(vxs), max(vxs)],
        "dwell_s": "U(1.1, 3.4) for every category (global resampling_time_range)",
    }
