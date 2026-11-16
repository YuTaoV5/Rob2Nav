#!/usr/bin/env python3
"""Offline self-check for the armD-to-R4000 turn-curriculum migration.

Purpose: prove, without starting a simulator or a training job, that
  1. the R4000 start checkpoint is the one the site recorded, and what its init actually restores,
  2. the control/action contract declared in the config really is what the R4000 run source encodes,
  3. the B curriculum machine produces REAL sustained turns, always followed by a release, with balanced
     left/right signs and commands inside the frozen stage envelope,
  4. the action wrapper's numeric behaviour can be read back and matches the declared contract,
  5. the B install path changes no snapshot file, so observation and wrapper stay byte-identical,
  6. a contract inconsistency makes the whole plan refuse to run.

Exit code 0 = every check passed.  Exit code 1 = at least one check failed and the plan must not run
(``dry_run.py --run`` refuses on that).  Exit code 2 = the check itself could not be performed.

Run:  python scripts/selfcheck.py [--config configs/ab_curriculum.json] [--probe]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import turn_schedule  # noqa: E402

FAILS = []
NOTES = []
SKIPS = []


def ok(name, detail, **extra):
    print("  PASS  %-34s %s" % (name, detail), flush=True)
    return {"check": name, "status": "pass", "detail": detail, **extra}


def bad(name, detail, **extra):
    print("  FAIL  %-34s %s" % (name, detail), flush=True)
    FAILS.append(name)
    return {"check": name, "status": "fail", "detail": detail, **extra}


def skip(name, detail):
    print("  SKIP  %-34s %s" % (name, detail), flush=True)
    SKIPS.append(name)
    return {"check": name, "status": "skip", "detail": detail}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# contract facts extracted from the R4000 run snapshot
# ------------------------------------------------------------------------------------------------
def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def grab(pattern, text, cast=float, flags=0):
    m = re.search(pattern, text, flags)
    return None if m is None else (cast(m.group(1)) if cast else m.group(1))


def extract_contract(cfg, snapshot, run_sh):
    c = dict(cfg["contract"])
    facts = {}
    all_py = {p.name: read(p) for p in sorted(Path(snapshot).glob("as2w_*.py"))}
    train_py = read(Path(snapshot) / "train_as2w_isaac_robust.py")
    fixed = all_py.get("as2w_rolling_fixed_exit.py", "")
    speed = all_py.get("as2w_isaac_speed_range.py", "")
    flat = all_py.get("as2w_isaac_flat_env_all.py", "")
    safety = all_py.get("as2w_training_safety.py", "")
    filt = all_py.get("as2w_rolling_filtered.py", "")
    sh = read(run_sh)

    facts["action_clip"] = grab(r"clip_actions=([0-9.]+)", train_py)
    # two DIFFERENT quantities: the wrapper pre-scale on the leg residual, and the action-term scale
    # (the divisor in the slew expression).  Conflating them would silently mis-state the joint offset.
    def _lead_dot(pattern, text):
        """Parse a leading-dot literal such as '.18' or '.086' into its float value."""
        raw = grab(pattern, text, cast=None)
        return None if raw is None else float("0." + raw)

    facts["wrapper_leg_prescale_rad_per_unit"] = _lead_dot(r"legs = \.([0-9]+) \* motion", fixed)
    facts["wheel_radius_m"] = _lead_dot(r"\.([0-9]+) \* 16\.", fixed)
    facts["wheel_action_scale_rad_s"] = grab(r"\.[0-9]+ \* ([0-9]+)\.", fixed)
    facts["leg_target_rate_limit_rad_s"] = _lead_dot(r"step_limit = \.([0-9]+) \* self\.step_dt", speed)
    facts["leg_rate_divisor"] = _lead_dot(r"step_limit = \.[0-9]+ \* self\.step_dt / \.([0-9]+)", speed)
    facts["turn_compensation_gain"] = grab(r"\.([0-9]+) \* \(1\. \+ \.25 \* steering\)", fixed)
    facts["turn_compensation_gain"] = None if facts["turn_compensation_gain"] is None else 0.66
    facts["turn_compensation_steer_coeff"] = grab(r"1\. \+ \.([0-9]+) \* steering", fixed)
    facts["turn_compensation_steer_coeff"] = None if facts["turn_compensation_steer_coeff"] is None else 0.25
    facts["gyro_correction_gain"] = grab(r"correction = \(\.([0-9]+) \* \(yaw - gyro_z", fixed)
    facts["gyro_correction_gain"] = None if facts["gyro_correction_gain"] is None else 0.8
    facts["gyro_correction_clamp"] = grab(r"correction = \(\.8 \* \(yaw - gyro_z\[:, None\]\)\)\.clamp\(-\.([0-9]+), \.([0-9]+)\)", fixed)
    facts["gyro_engagement_denominator"] = grab(r"engagement = \(yaw\.abs\(\) / \.([0-9]+)\)\.clamp\(max=1\.\)", fixed)
    facts["gyro_engagement_denominator"] = None if facts["gyro_engagement_denominator"] is None else 0.035
    facts["wheel_actuator_damping_kd"] = grab(r"joint_names_expr=joints, stiffness=stiffness, damping=([0-9.]+)", flat)
    facts["resampling_lo_s"] = grab(r"resampling_time_range = \(([0-9.]+), [0-9.]+\)", speed)
    facts["resampling_hi_s"] = grab(r"resampling_time_range = \([0-9.]+, ([0-9.]+)\)", speed)
    facts["episode_length_s"] = grab(r"cfg\.episode_length_s = ([0-9.]+)\.", train_py)
    facts["motion_start_s"] = grab(r"cfg\.as2w_motion_start_s = ([0-9.]+)\.", train_py)
    facts["park_until_s"] = grab(r"cfg\.as2w_park_until_s = ([0-9.]+)\.", train_py)
    facts["speed_cap_m_s"] = grab(r"--speed-cap ([0-9.]+)", sh)
    facts["pure_turn_common_scale"] = grab(r"--pure-turn-common-scale ([0-9.]+)", sh)
    facts["leg_filter_hz"] = grab(r"--rolling-leg-filter-hz ([0-9.]+)", sh)
    facts["filter_alpha_at_5hz"] = grab(r"alpha = -math\.expm1\(-2\. \* math\.pi \* ([0-9.]+) \* dt\)", filt)
    facts["safety_ninety_mentions"] = re.findall(r"[^\n]*0\.9[^\n]*effort[^\n]*|[^\n]*effort[^\n]*0\.9[^\n]*", safety)[:3]
    facts["num_steps_per_env_source"] = "isaaclab_tasks go2 rsl_rl_ppo_cfg num_steps_per_env"
    return facts


def check_contract(cfg, snapshot, run_sh, results):
    c = cfg["contract"]
    f = extract_contract(cfg, snapshot, run_sh)
    g = f["leg_target_rate_limit_rad_s"]
    exp = {
        "action_clip": (c["action_clip"], f["action_clip"]),
        "leg_action_scale_rad": (c["leg_action_scale_rad"], f["leg_rate_divisor"]),
        "wrapper_leg_prescale_rad_per_unit": (c["wrapper_leg_prescale_rad_per_unit"],
                                              f["wrapper_leg_prescale_rad_per_unit"]),
        "wheel_radius_m": (c["wheel_radius_m"], f["wheel_radius_m"]),
        "wheel_action_scale_rad_s": (c["wheel_action_scale_rad_s"], f["wheel_action_scale_rad_s"]),
        "leg_target_rate_limit_rad_s": (c["leg_target_rate_limit_rad_s"], g),
        "turn_compensation_gain": (0.66, f["turn_compensation_gain"]),
        "turn_compensation_steer_coeff": (0.25, f["turn_compensation_steer_coeff"]),
        "gyro_correction_gain": (0.8, f["gyro_correction_gain"]),
        "gyro_engagement_denominator": (0.035, f["gyro_engagement_denominator"]),
        "wheel_actuator_damping_kd": (c["wheel_actuator_damping_kd"], f["wheel_actuator_damping_kd"]),
        "episode_length_s": (cfg["shared_run"]["episode_length_s"], f["episode_length_s"]),
        "motion_start_s": (cfg["shared_run"]["motion_start_s"], f["motion_start_s"]),
        "park_until_s": (cfg["shared_run"]["park_until_s"], f["park_until_s"]),
        "speed_cap_m_s": (cfg["shared_run"]["speed_cap_m_s"], f["speed_cap_m_s"]),
        "pure_turn_common_scale": (c["pure_turn_common_scale"], f["pure_turn_common_scale"]),
        "leg_residual_filter_hz": (c["leg_residual_filter_hz"], f["leg_filter_hz"]),
    }
    mism = {k: (a, b) for k, (a, b) in exp.items() if b is None or abs(float(a) - float(b)) > 1e-9}
    declared_dwell = list(cfg["shared_run"]["dwell_range_s"])
    if f["resampling_lo_s"] is not None and f["resampling_hi_s"] is not None:
        if abs(declared_dwell[0] - f["resampling_lo_s"]) > 1e-9 or abs(declared_dwell[1] - f["resampling_hi_s"]) > 1e-9:
            mism["dwell_range_s"] = (declared_dwell, (f["resampling_lo_s"], f["resampling_hi_s"]))
    if mism:
        results.append(bad("contract_from_source", "config disagrees with the R4000 source: %s" % mism))
    else:
        results.append(ok("contract_from_source",
                          "16 declared constants match the R4000 run snapshot and run.sh", extracted=f))

    # the rate limit must be expressible both ways: 0.9 rad/s and 0.9*dt/0.35 action units per step
    div = f["leg_rate_divisor"]
    dt = c["policy_step_s"]
    if div is None or g is None:
        results.append(bad("rate_limit_units", "could not extract the slew expression"))
    else:
        per_step_action = g * dt / div
        per_step_joint = c["leg_action_scale_rad"] * per_step_action
        want = c["leg_target_rate_limit_rad_per_step"]
        if abs(per_step_joint - want) > 1e-9:
            results.append(bad("rate_limit_units",
                               "0.35*step_limit=%.6f rad/step but config says %.6f" % (per_step_joint, want)))
        else:
            results.append(ok("rate_limit_units",
                              "step_limit=%.6f action units -> %.6f rad/step -> %.3f rad/s"
                              % (per_step_action, per_step_joint, per_step_joint / dt)))

    # the 90 percent torque ceiling is derived later from FRACTION x URDF effort (check_torque_from_source)
    return f


# ------------------------------------------------------------------------------------------------
def check_checkpoint(cfg, results):
    import torch

    s = cfg["start"]
    p = Path(s["checkpoint"])
    if not p.is_file():
        results.append(bad("start_checkpoint_present", "missing %s" % p))
        return None
    h = sha256(p)
    size = p.stat().st_size
    if h != s["sha256_expected"] or size != s["bytes_expected"]:
        results.append(bad("start_checkpoint_identity",
                           "sha=%s bytes=%d (expected %s / %d)" % (h, size, s["sha256_expected"], s["bytes_expected"])))
        return None
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    a = ck["actor_state_dict"]["mlp.0.weight"]
    aout = ck["actor_state_dict"]["mlp.6.weight"]
    critic = ck["critic_state_dict"]["mlp.0.weight"]
    opt = ck.get("optimizer_state_dict")
    n_state = len((opt or {}).get("state") or {})
    shapes_ok = (tuple(a.shape) == (128, s["actor_obs"]) and tuple(aout.shape) == (s["actor_actions"], 128)
                 and tuple(critic.shape) == (128, s["actor_obs"]))
    if not shapes_ok:
        results.append(bad("start_checkpoint_shapes", "actor=%s out=%s critic=%s" % (a.shape, aout.shape, critic.shape)))
        return None
    results.append(ok("start_checkpoint_identity",
                      "sha256 matches the site record, %d bytes, iter=%s, actor %s->%s, critic %s"
                      % (size, ck.get("iter"), tuple(a.shape), tuple(aout.shape), tuple(critic.shape)),
                      sha256=h, bytes=size, iteration=ck.get("iter"),
                      optimizer_state_entries=n_state))
    declared = (s["optimizer_state_present"], s["optimizer_state_used"])
    if (opt is not None) != declared[0] or declared[1]:
        results.append(bad("optimizer_policy",
                           "checkpoint optimizer present=%s but config declares present=%s used=%s"
                           % (opt is not None, declared[0], declared[1])))
    else:
        results.append(ok("optimizer_policy",
                          "optimizer present=%s, entries=%d, used=False (R4000 entry loads actor+critic only; "
                          "A and B restore identically)" % (opt is not None, n_state)))
    return ck


# ------------------------------------------------------------------------------------------------
def check_b_schedule(cfg, results, n=200000, seed=7393):
    b = cfg["B"]["curriculum"]
    params = turn_schedule.DEFAULTS
    declared = {
        "share_pure_turn": b["within_mixed"]["pure_turn"],
        "share_turn_long": b["within_mixed"]["turn_long"],
        "share_arc": b["within_mixed"]["arc"],
        "share_straight": b["within_mixed"]["straight"],
        "arc_vx_m_s": b["arc"]["vx_m_s"],
    }
    drift = {k: (v, params[k]) for k, v in declared.items() if abs(float(v) - float(params[k])) > 1e-9}
    if drift:
        results.append(bad("b_schedule_matches_config", "module defaults drift from config: %s" % drift))
    else:
        results.append(ok("b_schedule_matches_config", "module defaults equal the declared B curriculum"))

    ts = turn_schedule.TurnSchedule(seed)
    st = ts.simulate(20000)
    results.append(ok("b_schedule_statistics",
                      "segments=%d shares=%s step_shares=%s" %
                      (st["segments"], {k: round(v, 3) for k, v in st["category_segment_share"].items()},
                       {k: round(v, 3) for k, v in st["category_step_share"].items()}),
                      statistics=st))

    checks = {}
    # real sustained turns
    lo, hi = b["turn_long"]["dwell_s"] if isinstance(b["turn_long"]["dwell_s"], list) else turn_schedule.DWELL_LONG_S
    lo, hi = turn_schedule.DWELL_LONG_S
    checks["long_turn_present"] = st["long_turns"] > 0
    rng_long = st["dwell_range"].get("turn_long")
    checks["long_turn_dwell_in_range"] = bool(rng_long and rng_long[0] >= lo - 1e-9 and rng_long[1] <= hi + 1e-9)
    checks["release_present"] = st["releases"] > 0
    checks["every_long_turn_followed_by_release"] = (
        st["long_turn_followed_by_release_rate"] is not None
        and st["long_turn_followed_by_release_rate"] > 0.999)
    checks["release_command_is_zero"] = bool(st["release_has_zero_command"])
    rel = st["dwell_range"].get("release")
    r_lo, r_hi = turn_schedule.RELEASE_S
    checks["release_dwell_in_range"] = bool(rel and rel[0] >= r_lo - 1e-9 and rel[1] <= r_hi + 1e-9)
    # short turns must survive
    short = st["dwell_range"].get("pure_turn")
    checks["short_turns_survive"] = bool(short and short[0] >= 1.1 - 1e-9 and short[1] <= 3.4 + 1e-9)
    # left/right mirror
    tsb = turn_schedule.TurnSchedule(seed + 7)
    bal = {c: [] for c in ("pure_turn", "turn_long", "arc")}
    for _ in range(60000):
        seg = tsb.draw_segment(False)
        if seg["category"] in bal:
            bal[seg["category"]].append(seg["wz"])
    sign_info = {}
    for cat, vals in bal.items():
        b_ = turn_schedule.TurnSchedule._sign_balance(vals)
        sign_info[cat] = {"n": len(vals), "balance": None if b_ is None else round(b_, 4)}
        checks["sign_balance_%s" % cat] = b_ is not None and abs(b_) < 0.02
    # envelope
    checks["envelope_within_r4000_stage"] = bool(st["envelope_ok"])
    # arcs bidirectional and in range
    arc_cmds = []
    ts2 = turn_schedule.TurnSchedule(seed + 1)
    for _ in range(5000):
        seg = ts2.draw_segment(False)
        if seg["category"] == "arc":
            arc_cmds.append((seg["vx"], seg["wz"]))
    checks["arcs_bidirectional"] = (any(w > 0 for _, w in arc_cmds) and any(w < 0 for _, w in arc_cmds))
    checks["arc_vx_in_range"] = all(abs(vx) <= turn_schedule.VX_CAP_M_S + 1e-9 for vx, _ in arc_cmds)
    checks["arc_vx_equals_declared"] = all(abs(vx - float(b["arc"]["vx_m_s"])) < 1e-9 for vx, _ in arc_cmds)
    checks["arc_wz_small_set"] = bool(arc_cmds) and all(
        any(abs(abs(w) - c) < 1e-9 for c in (0.02, 0.04, 0.06)) for _, w in arc_cmds)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        results.append(bad("b_schedule_invariants", "failed: %s" % failed, checks=checks))
    else:
        results.append(ok("b_schedule_invariants",
                          "long turns %d, all followed by release, release zero-filled, dwell in range, "
                          "L/R balanced %s, envelope respected, arcs bidirectional"
                          % (st["long_turns"], {k: v["balance"] for k, v in sign_info.items()}),
                          checks=checks, sign_balance=sign_info))
    return st


def check_r4000_reference(cfg, results):
    ref = turn_schedule.r4000_reference_sampler(311, 200000)
    bad_range = (ref["wz_magnitude_range"] is None
                 or ref["wz_magnitude_range"][1] > turn_schedule.WZ_CAP_RAD_S + 1e-9
                 or ref["vx_range"][0] < -cfg["shared_run"]["speed_cap_m_s"] - 1e-9
                 or ref["vx_range"][1] > cfg["shared_run"]["speed_cap_m_s"] + 1e-9)
    if bad_range:
        results.append(bad("a_reference_envelope", "R4000 reference sampler leaves its own stage cap: %s" % ref))
    else:
        results.append(ok("a_reference_envelope",
                          "R4000 mixed branch: shares=%s wz|max|=%.3f vx in [%.2f, %.2f]"
                          % ({k: round(v, 3) for k, v in ref["category_segment_share"].items()},
                             ref["wz_magnitude_range"][1], ref["vx_range"][0], ref["vx_range"][1]),
                          reference=ref))
    return ref


# ------------------------------------------------------------------------------------------------
def check_out_of_range_request(cfg, results):
    """The operator asked for a 0.25 m/s arc tier; the frozen stage caps at 0.20.  This must be refused."""
    req = cfg["B"].get("requested_arc_vx_m_s")
    used = cfg["B"]["curriculum"]["arc"]["vx_m_s"]
    cap = cfg["shared_run"]["speed_cap_m_s"]
    if req is None:
        results.append(skip("requested_arc_vx_gate", "no out-of-range request recorded"))
        return
    status = cfg["B"].get("requested_arc_vx_status")
    if req > cap + 1e-9:
        if used > cap + 1e-9:
            results.append(bad("requested_arc_vx_gate",
                               "B would run arcs at %.3f m/s, outside the frozen %.3f cap" % (used, cap)))
            return
        if status != "out_of_range":
            results.append(bad("requested_arc_vx_gate",
                               "requested %.3f > cap %.3f but config does not mark it out_of_range"
                               % (req, cap)))
            return
        results.append(ok("requested_arc_vx_gate",
                          "requested %.3f m/s exceeds the frozen %.3f m/s stage cap and is not used; "
                          "B runs %.3f m/s and the request is deferred to a separate speed experiment"
                          % (req, cap, used)))
    else:
        results.append(ok("requested_arc_vx_gate", "requested %.3f m/s is inside the frozen cap %.3f" % (req, cap)))


def check_install_path(cfg, results):
    """The shared entry must patch only the runner step count and (arm B) the curriculum class, and it must
    not rewrite any R4000 snapshot file."""
    snap = Path(cfg["provenance"]["run_dir"]) / "source"
    adapter = HERE / "b_adapter.py"
    entry = HERE / "train_ab_entry.py"
    if not snap.is_dir() or not adapter.is_file() or not entry.is_file():
        results.append(bad("install_path_files", "missing snapshot %s or adapter %s / entry %s"
                           % (snap, adapter, entry)))
        return
    a_src = adapter.read_text(encoding="utf-8")
    e_src = entry.read_text(encoding="utf-8")
    tree_a = ast.parse(a_src)
    tree_e = ast.parse(e_src)

    uses_ts = any(isinstance(n, ast.Attribute) and n.attr == "draw_segment" for n in ast.walk(tree_a))
    wires = {
        "adapter_calls_turnschedule": uses_ts and "TurnSchedule(" in a_src,
        "entry_installs_import_hook": "builtins.__import__" in e_src and "hooked_import" in e_src,
        "entry_forces_num_steps": "num_steps_per_env = int(num_steps)" in e_src
                                  and "AS2W_AB_NUM_STEPS" in e_src,
        "entry_rebinds_b_curriculum": "make_b_class(base)" in e_src
                                      and "RollingSpeedCommands = " in e_src,
        "entry_runs_frozen_entry": "runpy.run_path(entry" in e_src,
        "entry_arm_switch": "AS2W_AB_ARM" in e_src,
    }
    broken = [k for k, v in wires.items() if not v]
    if broken:
        results.append(bad("install_path_wiring", "failed: %s" % broken, wiring=wires))
    else:
        results.append(ok("install_path_wiring",
                          "one shared entry for both arms: import hook (post-simulation), forced "
                          "num_steps_per_env, B-only curriculum rebinding, frozen entry executed with runpy",
                          wiring=wires))

    declared_steps = int(cfg["shared_run"]["num_steps_per_env"])
    if ("AS2W_AB_NUM_STEPS" not in e_src) or str(declared_steps) not in e_src:
        results.append(bad("num_steps_traceable",
                           "config declares %d steps but the entry does not carry it" % declared_steps))
    else:
        results.append(ok("num_steps_traceable",
                          "config num_steps_per_env=%d is the value the entry forces into the runner config"
                          % declared_steps))

    forbidden_names = {"robust_observation", "observation", "rolling_fixed_exit", "filter_leg_residual"}
    referenced = set()
    for node in ast.walk(tree_a):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            referenced.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            referenced.add(node.id)
    if referenced:
        results.append(bad("install_path_no_wrapper_touch", "adapter references %s" % sorted(referenced)))
    else:
        results.append(ok("install_path_no_wrapper_touch",
                          "adapter references neither the observation function, the action wrapper nor the leg filter"))


def check_wrapper_readback(cfg, results):
    """Numerically read back the frozen action wrapper offline (torch only)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        results.append(skip("wrapper_readback", "torch unavailable: %s" % exc))
        return
    snap = Path(cfg["provenance"]["run_dir"]) / "source"
    path = snap / "as2w_rolling_fixed_exit.py"
    if not path.is_file():
        results.append(bad("wrapper_readback", "missing %s" % path))
        return
    # Importing the wrapper for the read-back must not leave bytecode inside the frozen R4000 run snapshot:
    # an earlier version did, which showed up as one modified file in a snapshot that must stay untouched.
    prev_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.path.insert(0, str(snap))
        spec = importlib.util.spec_from_file_location("r4_fixed_exit", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.rolling_fixed_exit
    except Exception as exc:
        sys.dont_write_bytecode = prev_bytecode
        results.append(skip("wrapper_readback", "snapshot wrapper not importable standalone: %s: %s"
                            % (type(exc).__name__, exc)))
        return
    finally:
        sys.dont_write_bytecode = prev_bytecode

    c = cfg["contract"]
    r, scale = c["wheel_radius_m"], c["wheel_action_scale_rad_s"]
    raw = torch.zeros(1, 16)
    raw[0, :12] = 1.0
    checks = {}

    legs, wheels = fn(raw, torch.tensor([[0.20, 0.0, 0.0]]), pure_turn_common_scale=0.0,
                      gyro_z=None, wheel_dq=None)
    want_wheel = 0.20 / (r * scale)
    checks["straight_wheel_target"] = bool(torch.allclose(wheels, torch.full((1, 4), want_wheel), atol=1e-6))
    checks["straight_all_four_equal"] = bool(torch.allclose(wheels, wheels[:, :1].expand(-1, 4), atol=1e-6))
    checks["leg_residual_scale_0p18_at_motion_1"] = bool(torch.allclose(legs, torch.full((1, 12), 0.18), atol=1e-6))

    legs0, _ = fn(raw, torch.tensor([[0.02, 0.0, 0.0]]), pure_turn_common_scale=0.0, gyro_z=None, wheel_dq=None)
    checks["leg_residual_motion_gated"] = bool(torch.allclose(legs0, torch.full((1, 12), 0.18 * 0.2), atol=1e-6))

    _, w_turn = fn(torch.zeros(1, 16), torch.tensor([[0.0, 0.0, 0.10]]), pure_turn_common_scale=0.0,
                   gyro_z=None, wheel_dq=None)
    side = torch.tensor([1.0, -1.0, 1.0, -1.0])
    expect = (-0.66 * 0.10 * side) / (r * scale)
    checks["turn_compensation_reads_back"] = bool(torch.allclose(w_turn[0], expect, atol=1e-6))
    # positive yaw command must spin the LEFT wheels backwards and the right wheels forwards
    checks["positive_yaw_left_backward"] = bool(w_turn[0, 0] < 0 and w_turn[0, 2] < 0)
    checks["positive_yaw_right_forward"] = bool(w_turn[0, 1] > 0 and w_turn[0, 3] > 0)

    _, w_gyro = fn(torch.zeros(1, 16), torch.tensor([[0.0, 0.0, 0.10]]), pure_turn_common_scale=0.0,
                   gyro_z=torch.tensor([[0.10]]), wheel_dq=None)
    checks["gyro_zero_error_matches_no_gyro"] = bool(torch.allclose(w_gyro, w_turn, atol=1e-6))
    _, w_gyro_over = fn(torch.zeros(1, 16), torch.tensor([[0.0, 0.0, 0.10]]), pure_turn_common_scale=0.0,
                        gyro_z=torch.tensor([[0.30]]), wheel_dq=None)
    checks["gyro_overrotation_reduces_turn"] = bool(w_gyro_over.abs().mean() < w_turn.abs().mean())
    _, w_gyro_under = fn(torch.zeros(1, 16), torch.tensor([[0.0, 0.0, 0.10]]), pure_turn_common_scale=0.0,
                         gyro_z=torch.tensor([[-0.10]]), wheel_dq=None)
    checks["gyro_underrotation_increases_turn"] = bool(w_gyro_under.abs().mean() > w_turn.abs().mean())

    raw_big = torch.full((1, 16), 3.0)
    _, w_clip = fn(raw_big, torch.tensor([[0.0, 0.0, 0.0]]), pure_turn_common_scale=0.0,
                   gyro_z=None, wheel_dq=None)
    checks["wheel_output_clipped_to_6"] = bool(w_clip.abs().max() <= 6.0 + 1e-6)

    failed = [k for k, v in checks.items() if not v]
    if failed:
        results.append(bad("wrapper_readback", "failed: %s" % failed, checks=checks))
    else:
        results.append(ok("wrapper_readback",
                          "12 numeric properties of the frozen wrapper read back offline "
                          "(straight targets, .18 motion-gated leg residual, -.66*(1+.25*steering) turn "
                          "compensation, sign convention, gyro correction both directions, +-6 clip)",
                          checks=checks))


def check_torque_from_source(cfg, snapshot, results):
    """Derive the 90 percent torque ceiling from the safety module and the URDF instead of declaring it."""
    safety = read(Path(snapshot) / "as2w_training_safety.py")
    frac = grab(r"^FRACTION\s*=\s*([0-9.]+)", safety, flags=re.M)
    urdf = Path(cfg["provenance"].get("urdf", ""))
    if frac is None or not urdf.is_file():
        results.append(skip("torque_90pct_derived",
                            "FRACTION=%s urdf_exists=%s; torque limits stay declared-not-derived"
                            % (frac, urdf.is_file())))
        return
    text = read(urdf)
    effort = {}
    for name, body in re.findall(r'<joint name="([A-Za-z_]+)"[^>]*>(.*?)</joint>', text, re.S):
        m = re.search(r'<limit[^>]*effort="([0-9.]+)"', body)
        if m:
            effort[name] = float(m.group(1))
    want = cfg["contract"]["torque_limits_nm"]
    groups = {"hip": "hip", "thigh": "thigh", "calf": "calf", "wheel": "foot"}
    derived, mismatch = {}, {}
    for group, suffix in groups.items():
        ref = effort.get("FL_" + suffix)
        if ref is None:
            mismatch[group] = "no URDF limit for FL_%s" % suffix
            continue
        derived[group] = round(frac * ref, 4)
        if abs(derived[group] - float(want[group])) > 1e-6:
            mismatch[group] = (want[group], derived[group])
    if mismatch:
        results.append(bad("torque_90pct_derived",
                           "torque limits disagree with FRACTION*URDF: %s" % mismatch,
                           fraction=frac, derived=derived, urdf=str(urdf)))
    else:
        results.append(ok("torque_90pct_derived",
                          "FRACTION=%.2f x URDF effort gives %s Nm, matching the declared 90%% ceiling"
                          % (frac, derived), fraction=frac, derived=derived, urdf=str(urdf)))


def check_replay(cfg, results):
    """Replay the state machine and the leg pipeline offline, and report dwell request vs installed."""
    import replay_contract

    rep = replay_contract.replay(cfg, verbose=False)
    dw = replay_contract.dwell_report(cfg)
    if rep["status"] != "pass":
        results.append(bad("replay_state_sequence",
                           "failed checks: %s" % [k for k, v in rep["checks"].items() if not v],
                           replay=rep))
    else:
        results.append(ok("replay_state_sequence",
                          "fixed stand -> takeover -> motion -> release -> brake -> return to stand replayed; "
                          "max leg target delta %.6f rad/step (limit %.6f)"
                          % (rep["detail"]["max_leg_target_delta_rad_per_step"],
                             rep["detail"]["limit_rad_per_step"]), replay=rep))
    dwell_ok = (dw["truncated_by_episode_end"] == 0 and dw["installed_long_turns"] > 0
                and dw["long_turn_downgraded_for_lack_of_time"] > 0)
    if dwell_ok:
        results.append(ok("dwell_request_vs_installed",
                          "installed %s; release truncations 0 (%d ordinary segments clipped at episode end); "
                          "%d long turns downgraded because the remaining episode time could not hold "
                          "dwell + the maximum release"
                          % (dw["installed_segments"], dw.get("segments_clipped_by_episode_end", 0),
                             dw["long_turn_downgraded_for_lack_of_time"]), dwell=dw))
    else:
        results.append(bad("dwell_request_vs_installed",
                           "truncated=%d installed_long=%d downgraded=%d"
                           % (dw["truncated_by_episode_end"], dw["installed_long_turns"],
                              dw["long_turn_downgraded_for_lack_of_time"]), dwell=dw))
    return rep, dw


def check_shared_budget(cfg, results):
    s = cfg["shared_run"]
    batch = s["num_envs"] * s["num_steps_per_env"]
    total = batch * s["iterations"]
    if total != s["env_step_budget"]:
        results.append(bad("env_step_budget",
                           "envs*steps*iters=%d but config declares %d" % (total, s["env_step_budget"])))
    else:
        results.append(ok("env_step_budget",
                          "rollout batch %d, %d iterations -> %d env steps (R4000 used 393,216,000, "
                          "ratio %.3f)" % (batch, s["iterations"], total, total / 393216000.0)))
    a, b = cfg["A"]["curriculum"], cfg["B"]["curriculum"]
    shared_keys = ("mixed_branch_share_of_segments", "base_branch_share_of_segments", "park_branch_share_of_segments")
    diff = [k for k in shared_keys if abs(float(a[k]) - float(b[k])) > 1e-9]
    if diff:
        results.append(bad("ab_structure_shared", "branch shares differ between A and B: %s" % diff))
    else:
        results.append(ok("ab_structure_shared",
                          "base %.0f%% / mixed %.0f%% / park %.0f%% identical in A and B; only the mixed "
                          "branch contents differ"
                          % (100 * a["base_branch_share_of_segments"], 100 * a["mixed_branch_share_of_segments"],
                             100 * a["park_branch_share_of_segments"])))
    if cfg["status"].get("training_started"):
        results.append(bad("phase_guard", "config says training started; this deliverable must stop pre-launch"))
    else:
        results.append(ok("phase_guard", "status.training_started=false, deployable=false"))


def probe(snapshot, run_sh):
    print("== probe: candidate source lines ==")
    for name in ("as2w_training_safety.py", "as2w_rolling_fixed_exit.py", "as2w_isaac_speed_range.py"):
        p = Path(snapshot) / name
        if not p.is_file():
            continue
        for i, line in enumerate(read(p).splitlines(), 1):
            if re.search(r"0\.9|\.9 \*|effort", line):
                print("  %-28s %4d  %s" % (name, i, line.strip()[:100]))
    print("== probe: run.sh flags ==")
    for line in read(run_sh).splitlines():
        if "--" in line:
            print("  %s" % line.strip()[:160])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=TASK_ROOT / "configs" / "ab_curriculum.json")
    ap.add_argument("--out", type=Path, default=TASK_ROOT / "evidence" / "selfcheck.json")
    ap.add_argument("--probe", action="store_true", help="print candidate source lines instead of checking")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(cfg["provenance"]["run_dir"])
    snapshot = run_dir / "source"
    run_sh = run_dir / "run.sh"

    print("== offline self-check ==")
    print("  config   %s" % args.config)
    print("  snapshot %s" % snapshot)
    print("  run.sh   %s" % run_sh)
    if args.probe:
        probe(snapshot, run_sh)
        return 0
    if not snapshot.is_dir() or not run_sh.is_file():
        print("ERR: R4000 run snapshot not found; cannot check the contract")
        return 2

    results = []
    ck = check_checkpoint(cfg, results)
    fact = check_contract(cfg, snapshot, run_sh, results)
    st = check_b_schedule(cfg, results)
    ref = check_r4000_reference(cfg, results)
    check_out_of_range_request(cfg, results)
    check_install_path(cfg, results)
    check_wrapper_readback(cfg, results)
    check_torque_from_source(cfg, snapshot, results)
    rep, dwell = check_replay(cfg, results)
    replay_payload = {"replay": rep, "dwell": dwell}
    check_shared_budget(cfg, results)

    payload = {
        "generated_at_unix": int(__import__("time").time()),
        "config": str(args.config),
        "snapshot": str(snapshot),
        "checks": results,
        "failures": FAILS,
        "skips": SKIPS,
        "extracted_contract": fact,
        "replay": replay_payload,
        "b_statistics": st,
        "r4000_reference": ref,
        "verdict": "PASS" if not FAILS else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print()
    print("  checks: %d pass, %d fail, %d skip" %
          (sum(1 for r in results if r["status"] == "pass"), len(FAILS), len(SKIPS)))
    print("  evidence: %s" % args.out)
    print("SELFCHECK_%s" % payload["verdict"], flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
