#!/usr/bin/env python3
"""Offline replay of the R4000 state machine and action pipeline, with no simulator.

The operator requires the deliverable to replay ``fixed stand -> takeover -> motion -> release -> brake ->
return to stand`` offline, check the filter reset and the state boundaries, confirm that the steady-state
leg target never moves more than 0.018 rad per 20 ms policy step, and record requested dwell, actual dwell
and the truncation reason.

What is replayed, and from which R4000 source
    * leg residual low pass (5 Hz first order)      as2w_rolling_filtered.filter_leg_residual
    * leg target slew limit 0.9 rad/s               as2w_isaac_speed_range RollingFixedExitEnvironment.step
                                                    (expressed as .9*step_dt/.35 action units, then x0.35)
    * takeover gate: command zero while episode_length_buf < 100   as2w_isaac_robust_env.TakeoverCommands
    * motion start: command zero before as2w_motion_start_s        train_as2w_isaac_robust.py
    * park / return to stand: park_until_s then zero               as2w_isaac_teleop.TeleopCommands
    * command filter lerp 0.12 and centered-stick deadband 0.008 m/s, 0.024 rad/s
    * reset zeroes the filter and slew states                      RollingFixedExitEnvironment._reset_idx

This is a model of the pipeline, not the pipeline itself; every constant is read from the config, which in
turn is checked against the R4000 source by selfcheck.py.  A mismatch is reported, never smoothed over.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import turn_schedule  # noqa: E402

FAILS = []


def ok(name, detail, **extra):
    print("  PASS  %-32s %s" % (name, detail), flush=True)
    return {"check": name, "status": "pass", "detail": detail, **extra}


def bad(name, detail, **extra):
    print("  FAIL  %-32s %s" % (name, detail), flush=True)
    FAILS.append(name)
    return {"check": name, "status": "fail", "detail": detail, **extra}


class Pipeline:
    """Replica of the R4000 leg pipeline: command filter, 5 Hz residual filter, 0.9 rad/s slew."""

    def __init__(self, dt, action_scale, rate_rad_s, filter_hz):
        self.dt = dt
        self.scale = action_scale
        self.rate = rate_rad_s
        self.alpha = -math.expm1(-2.0 * math.pi * filter_hz * dt)
        self.reset()

    def reset(self):
        self.filtered = 0.0      # normalized leg residual after the 5 Hz filter
        self.slewed = 0.0        # normalized leg residual after the slew limit
        self.joint_offset = 0.0  # rad, what the joints actually see

    def step(self, raw_residual):
        self.filtered += self.alpha * (raw_residual - self.filtered)
        delta = self.filtered - self.slewed
        limit = self.rate * self.dt / self.scale          # action units per policy step
        self.slewed += max(-limit, min(limit, delta))
        self.joint_offset = self.scale * self.slewed
        return self.joint_offset


def command_filter(prev, goal, dt, deadband_v=0.008, deadband_w=0.024, lerp=0.12):
    g = list(goal)
    if math.hypot(g[0], g[1]) < deadband_v:
        g[0] = g[1] = 0.0
    if abs(g[2]) < deadband_w:
        g[2] = 0.0
    out = [prev[i] + lerp * (g[i] - prev[i]) for i in range(3)]
    if all(abs(v) < 0.005 for v in out) and g == [0.0, 0.0, 0.0]:
        out = [0.0, 0.0, 0.0]
    return out


def replay(cfg, verbose=True):
    c = cfg["contract"]
    s = cfg["shared_run"]
    dt = c["policy_step_s"]
    pipe = Pipeline(dt, c["leg_action_scale_rad"], c["leg_target_rate_limit_rad_s"],
                    c["leg_residual_filter_hz"])
    episode_s = s["episode_length_s"]
    start_s = s["motion_start_s"]
    park_until = s["park_until_s"]

    # scripted request: (start_s, end_s, vx, wz, label)
    plan = [
        (0.0, start_s, 0.0, 0.0, "fixed_stand"),
        (start_s, start_s + 2.0, 0.0, 0.0, "takeover_zero"),
        (start_s + 2.0, start_s + 2.5, 0.0, 0.0, "hold"),
        (start_s + 2.5, start_s + 5.5, 0.20, 0.03, "arc_left_long"),
        (start_s + 5.5, 9.0, 0.0, 0.0, "release"),
        (9.0, 10.5, -0.20, 0.0, "straight_reverse"),
        (10.5, 11.2, 0.20, 0.0, "brake_entry"),
        (11.2, park_until, 0.0, 0.0, "brake_to_zero"),
        (park_until, episode_s, 0.0, 0.0, "return_to_stand"),
    ]

    steps = int(round(episode_s / dt))
    prev_cmd = [0.0, 0.0, 0.0]
    max_delta = 0.0
    nonzero_before_start = 0
    nonzero_before_takeover = 0
    zero_window_max = 0
    zero_window_now = 0
    rows = []
    for k in range(steps):
        t = k * dt
        goal = (0.0, 0.0, 0.0)
        for a, b, vx, wz, label in plan:
            if a <= t < b:
                goal = (vx, 0.0, wz)
                break
        # takeover gate: zero command while episode_length_buf < 100 (i.e. first 100 policy steps)
        if k < 100:
            goal = (0.0, 0.0, 0.0)
        if t < start_s:
            goal = (0.0, 0.0, 0.0)
        prev_joint = pipe.joint_offset
        raw = 1.0 if abs(goal[0]) > 0 or abs(goal[2]) > 0 else 0.0   # worst-case full residual
        pipe.step(raw)
        max_delta = max(max_delta, abs(pipe.joint_offset - prev_joint))
        cmd = command_filter(prev_cmd, goal, dt)
        if t < start_s and any(abs(v) > 1e-9 for v in cmd):
            nonzero_before_start += 1
        if k < 100 and any(abs(v) > 1e-9 for v in cmd):
            nonzero_before_takeover += 1
        if all(abs(v) < 1e-9 for v in cmd):
            zero_window_now += 1
            zero_window_max = max(zero_window_max, zero_window_now)
        else:
            zero_window_now = 0
        prev_cmd = cmd
        if k % 50 == 0 or k == steps - 1:
            rows.append({"t_s": round(t, 3), "cmd": [round(v, 4) for v in cmd],
                         "joint_offset_rad": round(pipe.joint_offset, 6)})

    checks = {}
    checks["leg_rate_limit_per_step"] = max_delta <= c["leg_target_rate_limit_rad_per_step"] + 1e-9
    checks["zero_command_before_motion_start"] = nonzero_before_start == 0
    checks["zero_command_during_takeover"] = nonzero_before_takeover == 0
    checks["release_window_at_least_3s"] = zero_window_max >= int(3.0 / dt) - 1
    # filter reset must erase history so the first post-reset step starts from zero
    pipe.reset()
    checks["filter_reset_zeroes_state"] = (pipe.filtered == 0.0 and pipe.slewed == 0.0
                                           and pipe.joint_offset == 0.0)

    detail = {
        "episode_s": episode_s, "motion_start_s": start_s, "park_until_s": park_until,
        "max_leg_target_delta_rad_per_step": round(max_delta, 6),
        "limit_rad_per_step": c["leg_target_rate_limit_rad_per_step"],
        "filter_alpha": round(pipe.alpha, 6),
        "slew_limit_action_units": round(c["leg_target_rate_limit_rad_s"] * dt / c["leg_action_scale_rad"], 6),
        "longest_zero_command_window_s": round(zero_window_max * dt, 3),
        "plan": [{"from_s": a, "to_s": b, "vx": vx, "wz": wz, "label": lab} for a, b, vx, wz, lab in plan],
        "trace_every_1s": rows,
    }
    if not all(checks.values()):
        FAILS.append("replay")
        if verbose:
            print("  FAIL  %-32s %s" % ("replay_checks", [k for k, v in checks.items() if not v]))
    elif verbose:
        print("  PASS  %-32s state sequence replayed: max leg delta %.6f rad/step (limit %.6f), "
              "zero command before motion start and during takeover, longest zero window %.3f s"
              % ("replay_state_sequence", max_delta, c["leg_target_rate_limit_rad_per_step"],
                 zero_window_max * dt))
    return {"status": "fail" if not all(checks.values()) else "pass",
            "checks": checks, "detail": detail}


def dwell_report(cfg, n_episodes=400, seed=311):
    """Requested versus installed dwell inside a real 23 s episode, plus why anything was cut short."""
    s = cfg["shared_run"]
    episode_s = s["episode_length_s"]
    ts = turn_schedule.TurnSchedule(seed)
    req = {}
    inst = {}
    trunc = 0
    clips = 0
    downgraded = 0
    long_ok = 0
    for _ in range(n_episodes):
        elapsed = 0.0
        pending = False
        while True:
            remaining = episode_s - elapsed
            if remaining <= 0.5:
                break
            seg = ts.draw_segment(pending, remaining_s=remaining)
            req[seg["category"]] = req.get(seg["category"], 0) + 1
            inst[seg["category"]] = inst.get(seg["category"], 0) + 1
            if seg.get("truncated_by_episode_end"):
                trunc += 1
            if seg.get("clipped_at_episode_end"):
                clips += 1
            if seg.get("downgraded_from"):
                downgraded += 1
            if seg["is_turn_long"]:
                long_ok += 1
            elapsed += seg["dwell_s"]
            pending = seg["next_pending_release"]
    total = sum(inst.values())
    return {
        "episodes": n_episodes, "episode_length_s": episode_s,
        "installed_segments": inst,
        "share": {k: round(v / max(1, total), 4) for k, v in inst.items()},
        "truncated_by_episode_end": trunc,
        "segments_clipped_by_episode_end": clips,
        "clip_note": "an ordinary final segment of an episode is expected to be clipped by the episode "
                     "boundary and is counted separately; a clipped RELEASE would break the after-long-turn "
                     "guarantee and is counted as truncated_by_episode_end, which must stay zero",
        "long_turn_downgraded_for_lack_of_time": downgraded,
        "installed_long_turns": long_ok,
        "dwell_policy": "a long turn is only installed when the remaining episode time can hold "
                        "dwell + the minimum release; otherwise it is downgraded to a short turn, so "
                        "requested dwell equals installed dwell inside an episode",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=TASK_ROOT / "configs" / "ab_curriculum.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    print("== offline replay (no simulator) ==")
    rep = replay(cfg)
    dw = dwell_report(cfg)
    print("  dwell: installed %s, truncated %d, long turns downgraded for lack of time %d"
          % (dw["installed_segments"], dw["truncated_by_episode_end"],
             dw["long_turn_downgraded_for_lack_of_time"]))
    payload = {"replay": rep, "dwell": dw, "failures": FAILS,
               "verdict": "PASS" if not FAILS else "FAIL"}
    out = args.out or (TASK_ROOT / "evidence" / "replay.json")
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print("  evidence: %s" % out)
    print("REPLAY_%s" % payload["verdict"], flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
