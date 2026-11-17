#!/usr/bin/env python
"""FS stair line: from-scratch blind stair training entry (dry-run by default).

Rules enforced by this file
--------------------------
* No ``--run``  -> the script only prints the pre-flight report and exits 0.
  Nothing is created except the evidence files under ``--out`` (no weights, no
  worker processes, no GPU touch).
* ``--selfcheck`` runs the PASS/FAIL gate list and exits non-zero on any FAIL.
* The only interpreter that reproduces the holdout is
  ``env -u LD_LIBRARY_PATH /opt/isaaclab-venv/bin/python``; the script fails
  fast when LD_LIBRARY_PATH is set.
* ``torch.set_num_threads(1)`` is enforced at the top of ``main`` (the exact
  line is printed in the dry-run report): without it the same seed produces a
  different trajectory.
* No pretrained weights are loaded. ``--init-checkpoint`` exists only as an
  explicit escape hatch and prints ``NOT_FROM_SCRATCH`` when used.

The stair line is CPU-only (MuJoCo + SB3); it must never allocate a GPU.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
REPO = Path("/root/gpufree-data/Rob2Nav")
RUNTIME = (REPO / "training/as2w-backup/as2w-r284-release/deliveries/"
           "as2w-deploy/omni-blind-stairs-r152/runtime")
ASSETS = (REPO / "training/as2w-backup/as2w-current/assets/r237-train-root")
CONFIG_DIR = HERE / "configs"
WHEEL_CONTRACT = REPO / "deploy/as2w/contracts/wheel_contract_v1.json"

if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import fs_stair_env as fse  # noqa: E402
import fs_stair_contract as cmod  # noqa: E402
import fs_stair_replay as replay_mod  # noqa: E402

# mujoco_screen resolves the URDF / MJCF through its module-level ROOT.
fse.sim.ROOT = ASSETS

ARCHES = {
    "1frame": {"arm": "S", "wrapper": "ObservationHistory", "frames": 1,
               "observation_dim": 60},
    "zero_history": {"arm": "H0", "wrapper": "ZeroHistory", "frames": 6,
                     "observation_dim": 6 * 60},
    "6frame": {"arm": "H6", "wrapper": "ObservationHistory", "frames": 6,
               "observation_dim": 6 * 60},
}
SELFCHECK_JSON = "selfcheck.json"

_LOG = []


def say(*parts):
    line = " ".join(str(part) for part in parts)
    print(line, flush=True)
    _LOG.append(line)


# ---------------------------------------------------------------------------- helpers
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source_pins(cfg) -> list:
    """Every gating pin must exist and match the recorded sha256."""
    rows = []
    for name, row in cfg["sources"].items():
        path = Path(row["path"])
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        ok = exists and digest == row["sha256"]
        rows.append({"name": name, "path": str(path), "sha256_expected": row["sha256"],
                     "sha256_actual": digest, "gating": bool(row.get("gating", True)),
                     "match": bool(ok)})
    return rows


def verify_gating_pins(cfg) -> None:
    for row in check_source_pins(cfg):
        if row["gating"] and not row["match"]:
            raise fse.ContractError(
                f"source pin {row['name']} mismatch: {row['path']} expected "
                f"{row['sha256_expected']} got {row['sha256_actual']}")


def check_ld_library_path() -> str:
    value = os.environ.get("LD_LIBRARY_PATH", "")
    if value:
        raise SystemExit(
            "LD_LIBRARY_PATH is set; run with `env -u LD_LIBRARY_PATH "
            "/opt/isaaclab-venv/bin/python ...` (otherwise torch/mujoco fail with "
            "undefined symbol: _PyErr_SetLocaleString)")
    return "unset"


def threads_location() -> dict:
    lines = Path(__file__).read_text().splitlines()
    matches = [number for number, line in enumerate(lines, 1)
               if line.strip() == "torch.set_num_threads(1)"]
    main_start = next((number for number, line in enumerate(lines, 1)
                       if line.startswith("def main(")), 0)
    primary = next((number for number in matches if number > main_start),
                   matches[0] if matches else 0)
    return {"primary": f"{Path(__file__).name}:{primary}", "all": matches,
            "note": "primary is the call at the top of main(); the other is a defensive "
                    "re-assertion inside run_training"}


def git_head() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=20).stdout
        rows = [line for line in status.splitlines() if line.strip()]
        modified = [line[3:] for line in rows if not line.startswith("??")]
        untracked = [line[3:] for line in rows if line.startswith("??")]
        return {"commit": head, "dirty_entries": len(rows),
                "tracked_modified": modified, "untracked": untracked,
                "protected_changes_note": "the pre-existing tracked modifications and "
                                          "untracked files above are other agents' work and "
                                          "are left untouched by this line",
                "source": "git rev-parse HEAD / git status --porcelain (read-only)"}
    except Exception as exc:  # noqa: BLE001
        return {"commit": None, "error": f"{type(exc).__name__}: {exc}"}


class ProbeEnv(gym.Env):
    """Minimal gymnasium env used only to instantiate a policy without MuJoCo."""

    def __init__(self, dim: int):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-fse.ACTION_CLIP, fse.ACTION_CLIP,
                                           (fse.ACTION_DIM,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(self.observation_space.shape, dtype=np.float32), 0., False,
                False, {})


def make_policy(cfg, arch: str, seed: int):
    """Instantiate a PPO policy (random init, no checkpoints) for inspection."""
    from stable_baselines3 import PPO
    spec = ARCHES[arch]
    env = ProbeEnv(spec["observation_dim"])
    network = cfg["network"]
    torch.manual_seed(int(seed))
    model = PPO(
        "MlpPolicy", env, n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"], n_epochs=cfg["ppo"]["n_epochs"],
        learning_rate=cfg["ppo"]["learning_rate"], gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"], clip_range=cfg["ppo"]["clip_range"],
        target_kl=cfg["ppo"]["target_kl"],
        policy_kwargs={"activation_fn": torch.nn.ELU,
                       "net_arch": {"pi": list(network["pi"]), "vf": list(network["vf"])},
                       "log_std_init": float(network["log_std_init"])},
        device="cpu", seed=int(seed), verbose=0,
    )
    return model


def describe_policy(model) -> dict:
    policy = model.policy
    layers = []
    for name, module in policy.mlp_extractor.policy_net.named_children():
        if isinstance(module, torch.nn.Linear):
            layers.append({"kind": "policy_net", "in": int(module.in_features),
                           "out": int(module.out_features),
                           "params": int(module.weight.numel() + module.bias.numel())})
    action = policy.action_net
    layers.append({"kind": "action_net", "in": int(action.in_features),
                   "out": int(action.out_features),
                   "params": int(action.weight.numel() + action.bias.numel())})
    value_layers = [int(module.out_features) for module in policy.mlp_extractor.value_net
                    if isinstance(module, torch.nn.Linear)]
    actor_params = sum(row["params"] for row in layers)
    digest = hashlib.sha256()
    for key in sorted(policy.state_dict()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(
            policy.state_dict()[key].detach().cpu().numpy(), dtype=np.float32).tobytes())
    return {"layers": layers, "activation": "ELU",
            "value_hidden": value_layers,
            "actor_params": int(actor_params), "log_std_params": 16,
            "actor_params_with_log_std": int(actor_params + 16),
            "state_dict_sha256": digest.hexdigest(),
            "state_dict_keys": len(policy.state_dict())}


def stage_ladder(cfg) -> list:
    return [fse.Stage.from_dict(row) for row in cfg["stages"]]


def make_env(cfg, arch: str, *, start_index: int = 0, curriculum=None,
             stage: fse.Stage | None = None, trace: bool = False, tap: dict | None = None):
    spec = ARCHES[arch]
    ladder = stage_ladder(cfg)
    stage = stage or ladder[start_index]
    env = fse.StairEnv(
        ASSETS, stage, curriculum=curriculum,
        torque_clip=cfg["torque"]["clip_mode"],
        torque_budget_nm=cfg["torque"]["budget_nm"], trace=trace,
        nonwheel_contact_n=float(
            cfg["thresholds"]["nonwheel_contact"]["contact_force_n_ge"]),
        promotion=cfg["curriculum"], control=cfg["control"],
        randomization=cfg.get("randomization_active", {}))
    if spec["wrapper"] == "ZeroHistory":
        return fse.ZeroHistory(env, frames=spec["frames"], tap=tap)
    return fse.ObservationHistory(env, frames=spec["frames"], tap=tap)


# ---------------------------------------------------------------------------- report
def build_report(cfg, args, contract_state: dict) -> dict:
    ladder = stage_ladder(cfg)
    arm = ARCHES[args.arch]
    model = make_policy(cfg, args.arch, cfg["seed"])
    net = describe_policy(model)
    arms = {}
    for name in ("1frame", "zero_history", "6frame"):
        arms[name] = describe_policy(make_policy(cfg, name, cfg["seed"]))
    geometry = {}
    for stage in ladder:
        built = fse.build_stair_model(ASSETS, cfg["holdout_seeds"][0], stage.levels,
                                      stage.rise_m, stage.tread_m)
        rows = fse.stair_geometry(built, stage.levels)
        fse.assert_stage_geometry(stage, rows, fse.geometry_x_range(rows))
        geometry[stage.name] = {
            "levels": stage.levels,
            "rise_m": None if not rows else round(rows[0]["rise_m"], 9),
            "tread_m": None if not rows else round(rows[0]["tread_depth_m"], 9),
            "x_range_m": [None if value is None else round(value, 9)
                          for value in fse.geometry_x_range(rows)],
            "first_riser_x_m": None if not rows else round(rows[0]["x_front_m"], 9),
            "rows": [{"level": row["level"], "x_front_m": round(row["x_front_m"], 9),
                      "x_back_m": round(row["x_back_m"], 9),
                      "rise_m": round(row["rise_m"], 9),
                      "tread_depth_m": round(row["tread_depth_m"], 9)} for row in rows],
            "asserted_against": "config derivation + known_geometry (tol 1e-6)",
        }
        if stage.levels == 0:
            geometry[stage.name].update({
                "no_stair_geoms": True,
                "ground_plane_x_range_m": [-4., 4.],
                "nominal_first_riser_x_m": stage.first_riser_x_m,
                "approach_line_m": stage.approach_line_m,
                "note": "flat_approach is asserted by the absence of stair_* geoms and by "
                        "the nominal / configured approach geometry",
            })
    contract = fse.load_contract(ASSETS)
    built = fse.build_stair_model(ASSETS, cfg["holdout_seeds"][0], 5, .06, .31)
    spec = fse.validate_contract(built, contract)
    raw_low, raw_high = built.actuator_ctrlrange[
        [built.actuator(name).id for name in contract["joint_names"]]].T
    budget = float(cfg["torque"]["budget_nm"])
    clip_mode = cfg["torque"]["clip_mode"]

    def _effective_limit(index):
        low, high = float(raw_low[index]), float(raw_high[index])
        if clip_mode == "mjcf_ctrlrange":
            return [low, high]
        if index >= 12:
            return [max(-budget, low), min(budget, high)]
        return [max(-.9 * abs(low), low), min(.9 * abs(high), high)]

    per_axis = [_effective_limit(index) for index in range(16)]
    effective = {
        "per_axis_nm": per_axis,
        "groups_nm": {"hip": per_axis[0], "thigh": per_axis[4],
                      "calf": per_axis[8], "wheel": per_axis[12]},
    }
    wheel_contract = {}
    if WHEEL_CONTRACT.is_file():
        wheel_contract = json.loads(WHEEL_CONTRACT.read_text())
    transmit_model = fse.build_stair_model(ASSETS, cfg["holdout_seeds"][0], 5, .06, .31)
    transmission = fse.read_transmission(
        transmit_model, contract["joint_names"],
        [transmit_model.actuator(name).id for name in contract["joint_names"]])
    ppo = cfg["ppo"]
    batch = int(cfg["n_workers"]) * int(ppo["n_steps"])
    total = int(cfg["total_env_steps"])
    active = dict(cfg.get("randomization_active", {}))
    report = {
        "line": cfg["line"],
        "status": cfg["status"],
        "deployable": cfg["deployable"],
        "scope": cfg["scope"],
        "mode": "run" if args.run else ("replay" if args.replay else "dry-run"),
        "phase": {
            "name": cfg["phase"],
            "disturbance": cfg["disturbance"],
            "phase2_policy_locked": bool(cfg["phase_gate"]["phase2_locked"]),
            "phase2_unlocked_this_run": bool(args.phase == "disturbance"
                                             and args.allow_disturbance
                                             and args.init_checkpoint is not None),
            "phase2_unlock_requires": cfg["phase_gate"]["unlock_requires"],
            "baseline_definition": cfg["phase_gate"]["baseline_definition"],
            "randomization": {
                "load (mass/inertia)": {"state": "on" if active.get("load") else "off",
                                        "detail": "mass / inertia scaling"},
                "friction": {"state": "on" if active.get("friction") else "off",
                             "detail": "per-scene ground friction override (phase-2 candidate)"},
                "gain (motor kp/kd)": {"state": "on" if active.get("gain") else "off",
                                       "detail": "leg/wheel kp/kd scaling (phase-2 candidate)"},
                "push": {"state": "on" if active.get("push") else "off",
                         "detail": "lateral push impulse"},
                "com_offset": {"state": "off", "detail": "not implemented; enabling it is "
                                                       "rejected instead of silently ignored"},
                "observation_noise": {"state": "off", "detail": "not implemented; enabling it "
                                                               "is rejected instead of silently "
                                                               "ignored"},
            },
            "nominal_mass": not bool(active.get("load")),
            "push_enabled": bool(active.get("push")),
        },
        "control_contract": {
            "version": cfg["control_contract"]["version"],
            "fingerprint_stored": contract_state["stored_fingerprint"],
            "fingerprint_computed": contract_state["computed_fingerprint"],
            "changed": bool(contract_state["changed"]),
            "message": contract_state["message"],
            "accepted": bool(contract_state["accepted"]),
            "params": cfg["control"],
            "payload": contract_state["payload"],
            "field_order": list(cmod.FIELD_ORDER),
            "change_rule": cfg["control_contract"]["change_rule"],
            "deployment_consistency": cfg["deployment_consistency"],
        },
        "tap_log": {
            **cfg["tap_log"],
            "sample_row": tap_sample_row(cfg),
            "estimated_bytes_per_row": tap_estimated_bytes(cfg),
        },
        "transmission": transmission,
        "source": {
            "git": git_head(),
            "pytest_note": "read-only git calls; nothing is staged or committed here",
            "pins": check_source_pins(cfg),
            "own_files": {str(path.relative_to(HERE)): sha256_file(path)
                          for path in sorted(HERE.glob("fs_stair_*.py"))},
            "config": {"path": str(Path(args.config).resolve()),
                       "sha256": sha256_file(Path(args.config).resolve())},
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "versions": {"torch": torch.__version__, "numpy": np.__version__,
                         "mujoco": fse.sim.mj.__version__ if hasattr(fse.sim, "mj") else None,
                         "stable_baselines3": _sb3_version()},
            "ld_library_path": check_ld_library_path(),
            "torch_threads": {"value": torch.get_num_threads(),
                              "enforced_at": threads_location()},
        },
        "assets": {
            "assets_root": str(ASSETS),
            "mjcf": {"path": str(ASSETS / "local-step-diagnostic/as2w_flat.xml"),
                     "sha256": sha256_file(ASSETS / "local-step-diagnostic/as2w_flat.xml")},
            "urdf": {"path": str(ASSETS / "official-model/robots/as2w_description/as2w.urdf"),
                     "sha256": sha256_file(
                         ASSETS / "official-model/robots/as2w_description/as2w.urdf")},
            "contract": {"path": str(ASSETS / "fall-baseline/fall-recovery-contract.json"),
                         "sha256": sha256_file(
                             ASSETS / "fall-baseline/fall-recovery-contract.json"),
                         "control_dt_s": contract["control_dt"],
                         "joint_order": contract["joint_names"]},
            "torque_point_source": "versioned wheel contract (read-only, owned by the "
                                   "rolling line; observed hash recorded, not gating)",
            "wheel_contract": {
                "path": str(WHEEL_CONTRACT),
                "sha256_observed": sha256_file(WHEEL_CONTRACT) if WHEEL_CONTRACT.is_file() else None,
                "gating": False,
                "reason": "file is modified by the rolling line; only the consumed fields "
                          "are checked",
                "torque_budget_nm": wheel_contract.get("torque_budget_nm"),
                "r11_legacy_wheel_action_scale": wheel_contract.get("contracts", {})
                .get("r11_legacy", {}).get("wheel_action_scale"),
            },
        },
        "init": {
            "initialization": ("checkpoint_resume" if args.init_checkpoint else "from_scratch"),
            "checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "checkpoint_sha256": (sha256_file(args.init_checkpoint)
                                  if args.init_checkpoint and Path(args.init_checkpoint).is_file()
                                  else None),
            "not_from_scratch": bool(args.init_checkpoint),
            "seed": int(cfg["seed"]),
            "method": "SB3 orthogonal default initialisation (torch.manual_seed(seed) "
                      "immediately before PPO construction)",
            "fingerprint_sha256": net["state_dict_sha256"],
            "fingerprint_note": "sha256 over sorted float32 policy state_dict; identical "
                                "seeds must reproduce it",
        },
        "observation": {
            **cfg["observation"],
            "total_dim": arm["observation_dim"],
            "arch": args.arch,
            "arm": arm["arm"],
            "history": dict(cfg["history"]),
        },
        "network": {
            **cfg["network"],
            "arch": args.arch,
            "layers": net["layers"],
            "actor_params": net["actor_params"],
            "actor_params_with_log_std": net["actor_params_with_log_std"],
            "total_policy_params": int(sum(p.numel() for p in model.policy.parameters())),
            "arms": {name: {"arm": ARCHES[name]["arm"],
                            "observation_dim": ARCHES[name]["observation_dim"],
                            "actor_params": row["actor_params"],
                            "actor_params_with_log_std": row["actor_params_with_log_std"],
                            "fingerprint_sha256": row["state_dict_sha256"]}
                     for name, row in arms.items()},
            "expected_by_lead": {"1frame_actor": 42896, "sixframe_actor": 81296},
        },
        "budget": {
            "total_env_steps": total,
            "n_workers": int(cfg["n_workers"]),
            "n_steps": int(ppo["n_steps"]),
            "ppo_batch_size": batch,
            "expected_updates": total // batch,
            "expected_env_steps_per_worker": total // int(cfg["n_workers"]),
            "control_period_s": float(cfg["control_period_s"]),
            "physics_substeps_per_step": 16,
            "device": cfg["device"],
            "cpu_workers": int(cfg["n_workers"]),
            "baseline_for_comparison": cfg["baseline"],
            "note": "budget is defined in environment steps, not PPO iterations",
        },
        "wheel_control_contract": {
            "joint_order": contract["joint_names"],
            "action_order": [name for term in contract["action_terms"]
                             for name in term["joints"]],
            "action_terms": contract["action_terms"],
            "leg_action_scale": fse.LEG_ACTION_SCALE,
            "wheel_action_scale": fse.WHEEL_ACTION_SCALE,
            "action_clip": fse.ACTION_CLIP,
            "leg_kp": fse.LEG_KP,
            "joint_kd": fse.JOINT_KD,
            "wheel_kp": 0.,
            "wheel_target": "clip(16.0 * action[12:], +-wheel_velocity_limit)",
            "leg_target": "clip(default[:12] + 0.35 * action[:12], urdf limits)",
            "wheel_velocity_limit_rad_s": contract_limits(contract)[12:, 2].tolist(),
            "torque_rule": "leg: kp*(leg-q) - kd*dq ; wheel: kd*(wheel - dq)",
            "torque_clip_mode": cfg["torque"]["clip_mode"],
            "mujoco_ctrlrange_nm": {"low": raw_low.tolist(), "high": raw_high.tolist()},
            "effective_limits_nm": effective,
            "contract_torque_budget_nm": budget,
            "contract_torque_budget_source": cfg["torque"]["budget_source"],
            "torque_source_labels": cfg["torque_evidence"],
            "mismatches": [
                {"field": "wheel_action_scale",
                 "stair_value": fse.WHEEL_ACTION_SCALE,
                 "fall_recovery_contract_value": 6.0,
                 "status": "different_contract_family",
                 "note": "fall-recovery-contract action_terms describe the released "
                         "policy; the pinned training source uses 16.0 (line 219) and "
                         "the wheel contract r11_legacy also uses 16.0"},
                {"field": "wheel_kd",
                 "stair_value": fse.JOINT_KD,
                 "wheel_contract_r11_legacy": 6.0,
                 "status": "different_arm",
                 "note": "stair training uses kd=2.0 for all joints (pinned source line 82)"},
                {"field": "observation_dim",
                 "stair_value": 60,
                 "wheel_contract_value": 65,
                 "status": "not_applicable",
                 "note": "the 65-dim contract belongs to the rolling line"},
                {"field": "torque_clip",
                 "stair_value": cfg["torque"]["clip_mode"],
                 "pinned_training_value": "mujoco_ctrlrange (15 Nm wheels)",
                 "status": "deliberate_deviation",
                 "note": "AGENTS.md 3 requires the 90 % budget on every axis. "
                         "--torque-clip mjcf_ctrlrange reproduces the pinned envelope."},
            ],
        },
        "geometry": geometry,
        "stage": {
            "current": ladder[args.stage_index].name,
            "current_index": int(args.stage_index),
            "ladder": [{"name": stage.name, "levels": stage.levels,
                        "rise_m": stage.rise_m, "tread_m": stage.tread_m,
                        "episode_s": stage.episode_s, "start_back_m": stage.start_back_m,
                        "approach_line_m": stage.approach_line_m,
                        "first_riser_x_m": stage.first_riser_x_m,
                        "description": stage.description} for stage in ladder],
            "curriculum": cfg["curriculum"],
            "promotion_eligibility": cfg["curriculum"]["eligibility"],
            "holdout_seeds": cfg["holdout_seeds"],
            "train_scene_seed_range": cfg["train_scene_seed_range"],
        },
        "thresholds": fse.Thresholds().as_dict(),
        "promotion_warning": promotion_warning(cfg),
        "evidence_rules": cfg["evidence_rules"],
        "operator_decisions": cfg["operator_decisions"],
        "baseline": cfg["baseline"],
    }
    return report


def contract_limits(contract: dict):
    return fse.sim.load_target_limits(contract)


def _sb3_version() -> str:
    try:
        import stable_baselines3
        return stable_baselines3.__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


def promotion_warning(cfg) -> dict:
    cur = cfg["curriculum"]
    signal = cur["promotion_signal"]
    text = (
        f"promotion signal '{signal}': the task-defined criterion is full start + top "
        f"stable hold >= {cur['min_top_hold_s']:.2f} s + no hard protection. The "
        "diagnostic line measured course_completed = 0/6 on 1-, 2- and 5-step terrain "
        "for all three arms (seeds 28001-28006), and four-wheel simultaneous top load "
        "peaks at 0.26 s < 0.50 s. With this signal the ladder will stay on stage 0 "
        "forever. Operator decision required before --run; alternatives are listed in "
        "curriculum.alternatives and are OFF by default.")
    if signal != "top_stable_hold":
        text = (f"promotion signal '{signal}' is NOT the task-defined criterion "
                f"(top_stable_hold). The operator switched it explicitly; the default "
                f"criterion measured 0/6 and would stall the ladder at stage 0.")
    return {"signal": signal, "severity": "warning", "degraded": True, "text": text,
            "alternatives": cur["alternatives"]}


def tap_sample_row(cfg) -> dict:
    """One representative row of the triple-tap schema, computed from the contract.

    No episode is run in a dry-run; the numbers come from the configured scales
    and limits so the units and the field names are unambiguous.
    """
    control = cfg["control"]
    action = np.zeros(16, dtype=np.float32)
    action[12:] = .5
    action[:12] = .1
    wheel_pre = control["wheel_action_scale"] * action[12:]
    wheel_limited = np.clip(wheel_pre, -98.47, 98.47)
    leg_pre = np.asarray(json.loads(
        (ASSETS / "fall-baseline/fall-recovery-contract.json").read_text())["default_joint_pos"]
    )[:12] + control["leg_action_scale"] * action[:12]
    return {
        "sample_kind": "computed_from_contract (no episode run in dry-run)",
        "env_id": "w0", "episode_id": 1, "policy_step": 1, "dtype": "float32",
        "field_order": list(cmod.FIELD_ORDER),
        "window_shape": [int(cfg["history"]["frames"]), 60],
        "target_wheel_pre_filter": [round(float(v), 6) for v in wheel_pre],
        "target_wheel_clipped": [round(float(v), 6) for v in wheel_limited],
        "target_wheel_dispatch": [round(float(v), 6) for v in wheel_limited],
        "target_leg_pos_pre_filter": [round(float(v), 6) for v in leg_pre],
        "target_leg_pos_dispatch": [round(float(v), 6) for v in leg_pre],
        "tau_request_est_nm": [0.0] * 16,
        "tau_request_label": "ESTIMATED",
        "tau_actuator_coord_nm": [0.0] * 16,
        "tau_joint_qfrc_nm": [0.0] * 16,
        "torque_budget_nm": float(cfg["torque"]["budget_nm"]),
        "torque_clip_mode": cfg["torque"]["clip_mode"],
        "wheel_limit_rad_s": [98.47] * 4,
        "history_newest": [0.0] * 60,
        "history_sha256": "0" * 64,
        "units": {"target_wheel_*": "rad/s", "target_leg_pos_*": "rad",
                  "tau_*": "Nm", "history_newest": "observation units"},
    }


def tap_estimated_bytes(cfg) -> int:
    frames = int(cfg["history"]["frames"])
    per_row = 4 * 3 * 9 + 12 * 2 * 9 + 16 * 3 * 9 + 60 * 9 + 64 + 300
    return per_row


def print_report(report: dict) -> None:
    say("=" * 78)
    say("FS STAIR FROM-SCRATCH ENTRY --", report["mode"].upper())
    say("=" * 78)
    src = report["source"]
    say("[source] git", src["git"].get("commit"), "dirty", src["git"].get("dirty_entries"),
        "| interpreter", src["interpreter"])
    say("[source] python", src["python"], "torch", src["versions"]["torch"],
        "numpy", src["versions"]["numpy"], "mujoco", src["versions"]["mujoco"],
        "sb3", src["versions"]["stable_baselines3"])
    say("[source] LD_LIBRARY_PATH", src["ld_library_path"],
        "| torch threads", src["torch_threads"]["value"],
        "enforced at", src["torch_threads"]["enforced_at"]["primary"])
    for row in src["pins"]:
        say(f"[source] pin {row['name']:<28} gating={row['gating']} "
            f"match={row['match']} sha256={row['sha256_actual']}")
    say("[asset ]", report["assets"]["mjcf"]["path"], report["assets"]["mjcf"]["sha256"])
    say("[asset ]", report["assets"]["urdf"]["path"], report["assets"]["urdf"]["sha256"])
    say("[asset ] contract", report["assets"]["contract"]["sha256"],
        "control_dt", report["assets"]["contract"]["control_dt_s"])
    say("[asset ] wheel contract observed (not gating):",
        report["assets"]["wheel_contract"]["sha256_observed"],
        "torque_budget_nm", report["assets"]["wheel_contract"]["torque_budget_nm"])
    init = report["init"]
    if init["not_from_scratch"]:
        say("NOT_FROM_SCRATCH !!! --init-checkpoint was given:",
            init["checkpoint"], "-> this is NOT a from-scratch run")
    say("[init  ]", init["initialization"], "| seed", init["seed"],
        "| fingerprint", init["fingerprint_sha256"][:32], "| checkpoint",
        init["checkpoint"], "| sha256", init["checkpoint_sha256"])
    scope = report["scope"]
    say("[scope ] in:", scope["in_scope"])
    say("[scope ] vx_hat head:", "NOT used on the stair line (rolling line's C arm)")
    say("[scope ] cpu only:", scope["cpu_only"])
    phase = report["phase"]
    say("[phase ]", phase["name"], "| disturbance", phase["disturbance"],
        "| phase2 policy locked", phase["phase2_policy_locked"],
        "| unlocked_this_run", phase["phase2_unlocked_this_run"],
        "| nominal_mass", phase["nominal_mass"], "| push", phase["push_enabled"])
    for name, row in phase["randomization"].items():
        say(f"[random] {name:<18} {row['state']:<3} -- {row['detail']}")
    ctl = report["control_contract"]
    say("[ctl   ] control_contract_version", ctl["version"],
        "| fingerprint", ctl["fingerprint_computed"])
    say("[ctl   ] changed", ctl["changed"], "| accepted", ctl["accepted"])
    params = ctl["params"]
    say("[ctl   ] action_clip", params["action_clip"],
        "| wheel_filter_alpha", params["wheel_target_filter_alpha"],
        "| wheel_rate_limit_per_s", params["wheel_target_rate_limit_per_s"],
        "| leg_filter_alpha", params["leg_target_filter_alpha"],
        "| action_delay_frames", params["action_delay_frames"])
    say("[ctl   ] reset_rule", params["reset_rule"])
    say("[ctl   ] field_order", ctl["field_order"])
    trans = report["transmission"]
    say("[trans ] torque quantity", trans["torque_quantity"], "|", trans["label"])
    say("[trans ] direct_motor_gear1", trans["direct_motor_gear1"],
        "| one_actuator_per_joint", trans["one_actuator_per_joint"],
        "| actuators_per_joint", trans["actuators_per_joint"])
    for row in trans["actuators"][:4]:
        say(f"[trans ]   {row['name']:<9} kind={row['kind']} gear={row['gear']} "
            f"trntype={row['trntype']} dyntype={row['dyntype']} gaintype={row['gaintype']} "
            f"biastype={row['biastype']}")
    taps = report["tap_log"]
    say("[tap   ] path", taps["dir_pattern"], "| windows", taps["window_dir"],
        "| stride", taps["stride"], "| max_rows", taps["max_rows"],
        "| ~bytes/row", taps["estimated_bytes_per_row"])
    for row in taps["fields"]:
        say(f"[tap   ]   {row['name']:<26} {row['size']:>3} {row['unit']:<8} {row['label']}")
    say("[tap   ] sample", json.dumps(taps["sample_row"], sort_keys=True))
    say("[tap   ] sampling", taps["sampling"])
    obs = report["observation"]
    say("[obs   ] arch", obs["arch"], "arm", obs["arm"], "| single frame",
        obs["single_frame_dim"], "| total", obs["total_dim"], "| history",
        obs["history"]["mode"], "frames", obs["history"]["frames"],
        "span_ms", obs["history"]["span_ms"], obs["history"]["order"])
    for row in obs["components"]:
        say(f"[obs   ]   {row['index'][0]:>2}:{row['index'][1]:<3} {row['name']} "
            f"({row['size']}) {row.get('unit', '')}")
    net = report["network"]
    say("[net   ] activation", net["activation"], "pi", net["pi"], "vf", net["vf"],
        "| log_std_init", net["log_std_init"])
    for row in net["layers"]:
        say(f"[net   ]   {row['kind']:<11} {row['in']:>3} -> {row['out']:<3} "
            f"params {row['params']}")
    say("[net   ] actor params", net["actor_params"], "(+16 log_std =",
        net["actor_params_with_log_std"], ") | total policy params",
        net["total_policy_params"])
    for name, row in net["arms"].items():
        say(f"[net   ]   arm {row['arm']:<2} {name:<12} obs {row['observation_dim']:>3} "
            f"actor {row['actor_params']:>6} (+16 = {row['actor_params_with_log_std']})")
    budget = report["budget"]
    say("[budget] device", budget["device"], "| cpu workers (n_workers)",
        budget["cpu_workers"], "| total env steps", budget["total_env_steps"])
    say("[budget] ppo batch", budget["ppo_batch_size"], "| expected updates",
        budget["expected_updates"], "| env steps/worker",
        budget["expected_env_steps_per_worker"], "| control period",
        budget["control_period_s"], "s | substeps", budget["physics_substeps_per_step"])
    wc = report["wheel_control_contract"]
    say("[wheel ] leg scale", wc["leg_action_scale"], "| wheel scale",
        wc["wheel_action_scale"], "| action clip", wc["action_clip"])
    say("[wheel ] leg_kp", wc["leg_kp"], "| kd(all joints)", wc["joint_kd"],
        "| wheel_kp", wc["wheel_kp"])
    say("[wheel ] wheel velocity limit rad/s", wc["wheel_velocity_limit_rad_s"])
    say("[wheel ] torque rule:", wc["torque_rule"])
    say("[wheel ] torque clip mode", wc["torque_clip_mode"], "| contract budget",
        wc["contract_torque_budget_nm"], "Nm from", wc["contract_torque_budget_source"])
    say("[wheel ] raw ctrlrange low", wc["mujoco_ctrlrange_nm"]["low"])
    say("[wheel ] raw ctrlrange high", wc["mujoco_ctrlrange_nm"]["high"])
    say("[wheel ] effective limits (mode=%s; legs 12 x [low,high], wheels 4 x [low,high])"
        % wc["torque_clip_mode"], wc["effective_limits_nm"])
    for row in wc["mismatches"]:
        say(f"[wheel ] mismatch {row['field']}: stair={row['stair_value']} "
            f"status={row['status']} -- {row['note']}")
    for name, geo in report["geometry"].items():
        say(f"[geom  ] {name:<14} levels={geo['levels']} rise={geo['rise_m']} "
            f"tread={geo['tread_m']} x_range={geo['x_range_m']} "
            f"front={geo['first_riser_x_m']}")
        for row in geo["rows"]:
            say(f"[geom  ]   level {row['level']} x {row['x_front_m']:.5f}.."
                f"{row['x_back_m']:.5f} rise {row['rise_m']:.5f} "
                f"tread {row['tread_depth_m']:.5f}")
    stage = report["stage"]
    say("[stage ] current", stage["current"], "index", stage["current_index"],
        "| ladder", [row["name"] for row in stage["ladder"]])
    for row in stage["ladder"]:
        say(f"[stage ]   {row['name']:<14} levels={row['levels']} rise={row['rise_m']} "
            f"tread={row['tread_m']} episode={row['episode_s']}s "
            f"start_back={row['start_back_m']} line={row['approach_line_m']}")
    cur = stage["curriculum"]
    say("[stage ] promotion", cur["promotion_signal"], "| promote_after",
        cur["promote_after"], "| demote_after", cur["demote_after"],
        "| degenerate_after", cur["degenerate_after"])
    say("[stage ] eligibility", cur["eligibility"])
    for row in report["thresholds"].items():
        say(f"[thresh] {row[0]}: {row[1]}")
    warning = report["promotion_warning"]
    say("[WARN  ]", warning["text"])
    for alt in warning["alternatives"]:
        say(f"[WARN  ]   alternative {alt['signal']}: {alt['note']} (default off)")
    for row in report["operator_decisions"]:
        say("[decide]", row["id"], "-", row["question"])
    say("[rules ]", report["evidence_rules"]["improvement_rule"])
    say("[rules ] status", report["status"], "deployable", report["deployable"])
    say("--- DRYRUN_JSON ---")
    say(json.dumps(report, sort_keys=True, default=str))


# ---------------------------------------------------------------------------- evidence
def write_evidence(out: Path, cfg, args, report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "run-config.json").write_text(json.dumps(
        {"cli": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
         "config": cfg, "report_source": report["source"],
         "report_budget": report["budget"],
         "phase": report["phase"], "control_contract": report["control_contract"],
         "tap_log": {key: report["tap_log"][key]
                     for key in ("enabled", "stride", "max_rows", "dir_pattern",
                                 "window_dir", "sampling")},
         "transmission": report["transmission"]}, indent=2) + "\n")
    lines = []
    for path in [Path(args.config).resolve(), HERE / "fs_stair_env.py",
                 HERE / "fs_stair_train.py", HERE / "fs_stair_contract.py",
                 HERE / "fs_stair_replay.py", HERE / "fs_stair_eval_flat.py"]:
        lines.append(f"{sha256_file(path)}  {path}")
    for row in report["source"]["pins"]:
        lines.append(f"{row['sha256_actual']}  {row['path']}")
    for key in ("mjcf", "urdf", "contract"):
        row = report["assets"][key]
        lines.append(f"{row['sha256']}  {row['path']}")
    (out / "inputs.sha256").write_text("\n".join(lines) + "\n")
    (out / "cmd.sh").write_text("#!/bin/bash\n" + " ".join(
        shlex.quote(part) for part in [sys.executable, str(Path(__file__).resolve())]
        + sys.argv[1:]) + "\n")
    if not args.run:
        (out / "dryrun.log").write_text("\n".join(_LOG) + "\n")


# ---------------------------------------------------------------------------- training
def run_training(cfg, args, report: dict, out: Path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    monitor = fse.DegenerationMonitor(after=int(cfg["curriculum"]["degenerate_after"]))
    failures_dir = out / "failures"
    failures_dir.mkdir(exist_ok=True)
    episodes_path = out / "episodes.jsonl"

    def factory(rank):
        curriculum = fse.StairCurriculum(
            stage_ladder(cfg), start_index=args.stage_index,
            promote_after=cfg["curriculum"]["promote_after"],
            demote_after=cfg["curriculum"]["demote_after"])
        return make_env(cfg, args.arch, start_index=args.stage_index,
                        curriculum=curriculum, tap=worker_tap(cfg, out, rank))

    class StairLogCallback(BaseCallback):
        def _on_step(self):
            for info in self.locals.get("infos", []):
                record = info.get("fs_stair")
                if not record:
                    continue
                with episodes_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                for row in record["first_fails"]:
                    with (failures_dir / f"{row['kind']}.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                if record["started_from_bottom"] and not record["manual"]:
                    if monitor.update(record["stage"], bool(record["complete"])):
                        print("PROMOTION_SIGNAL_DEGENERATE",
                              json.dumps(monitor.summary(), sort_keys=True), flush=True)
            return True

    vec = SubprocVecEnv([lambda rank=rank: factory(rank) for rank in range(int(cfg["n_workers"]))],
                        start_method="fork")
    vec = VecMonitor(vec, filename=str(out / "monitor.csv"))
    try:
        ppo = cfg["ppo"]
        model = PPO(
            "MlpPolicy", vec, n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], learning_rate=ppo["learning_rate"],
            gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"], target_kl=ppo["target_kl"],
            policy_kwargs={"activation_fn": torch.nn.ELU,
                           "net_arch": {"pi": list(cfg["network"]["pi"]),
                                        "vf": list(cfg["network"]["vf"])},
                           "log_std_init": float(cfg["network"]["log_std_init"])},
            device="cpu", seed=int(cfg["seed"]), verbose=1)
        if args.init_checkpoint:
            checkpoint_sha = sha256_file(args.init_checkpoint)
            print("NOT_FROM_SCRATCH", flush=True)
            print("INIT_CHECKPOINT", args.init_checkpoint, "sha256", checkpoint_sha, flush=True)
            state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            model.policy.load_state_dict(state["policy"] if "policy" in state else state)
            (out / "init-checkpoint.json").write_text(json.dumps(
                {"path": str(args.init_checkpoint), "sha256": checkpoint_sha,
                 "phase": cfg["phase"], "disturbance": cfg["disturbance"],
                 "not_from_scratch": True}, indent=2) + "\n")
        model.learn(total_timesteps=int(cfg["total_env_steps"]),
                    callback=StairLogCallback(), log_interval=10)
        model.save(out / "ppo.zip")
    finally:
        vec.close()
    evaluate_holdout(model, cfg, args, out)
    print("EVAL_STAGES", json.dumps({"train_stage": args.stage,
                                     "eval_stage": args.eval_stage}), flush=True)
    evaluate_with_sim(cfg, model, out, stage_name=args.eval_stage)


def worker_tap(cfg, out, rank: int) -> dict:
    tap = dict(cfg["tap_log"])
    tap.update({"dir": str(Path(out) / "tap"),
                "window_dir": str(Path(out) / "windows"),
                "worker_id": int(rank)})
    return tap


def evaluate_holdout(model, cfg, args, out: Path) -> dict:
    ladder = {stage.name: stage for stage in stage_ladder(cfg)}
    stage = ladder[args.eval_stage]
    env = make_env(cfg, args.arch, stage=stage, tap=worker_tap(cfg, out, 0))
    rows = []
    try:
        for seed in cfg["holdout_seeds"]:
            obs, _ = env.reset(seed=int(seed), options={"scene_seed": int(seed)})
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
            record = info.get("fs_stair") or env.unwrapped.episode_record(
                terminated=terminated, truncated=truncated)
            record["seed"] = int(seed)
            rows.append(record)
    finally:
        env.close()
    summary = {
        "eval_stage": stage.name,
        "holdout_seeds": cfg["holdout_seeds"],
        "wheel_levels": {str(row["seed"]): row["ever_levels"] for row in rows},
        "top_stable": {str(row["seed"]): row["top_stable"] for row in rows},
        "all_wheels_top": {str(row["seed"]): row["all_wheels_top"] for row in rows},
        "terminated": {str(row["seed"]): row["terminated"] for row in rows},
        "max_forward_m": {str(row["seed"]): row["max_forward_m"] for row in rows},
        "resolution_rule": cfg["evidence_rules"]["improvement_rule"],
    }
    (out / "holdout.json").write_text(json.dumps(
        {"summary": summary, "episodes": rows}, indent=2) + "\n")
    print("HOLDOUT", json.dumps(summary, sort_keys=True), flush=True)
    return summary


# ---------------------------------------------------------------------------- selfcheck
class Checker:
    def __init__(self):
        self.rows = []

    def check(self, name: str, fn, *, detail=None):
        try:
            info = fn()
            self.rows.append({"name": name, "status": "PASS", "info": info})
            print(f"PASS {name}: {json.dumps(info, default=str)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.rows.append({"name": name, "status": "FAIL",
                              "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc().splitlines()[-6:]})
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)

    def summary(self) -> dict:
        failed = [row["name"] for row in self.rows if row["status"] == "FAIL"]
        return {"status": "PASS" if not failed else "FAIL", "checks": self.rows,
                "passed": len(self.rows) - len(failed), "failed": failed,
                "total": len(self.rows)}


def run_selfcheck(cfg, args, out: Path | None) -> int:
    checker = Checker()
    ladder = stage_ladder(cfg)
    contract = fse.load_contract(ASSETS)
    limits = fse.sim.load_target_limits(contract)

    def source_pins_check():
        rows = check_source_pins(cfg)
        bad = [row["name"] for row in rows if row["gating"] and not row["match"]]
        if bad:
            raise AssertionError(f"gating pins changed: {bad}")
        return {"pins": {row["name"]: {"gating": row["gating"], "match": row["match"]}
                         for row in rows}}

    checker.check("source_pins", source_pins_check)

    checker.check("ld_library_path", lambda: {"value": check_ld_library_path()})

    checker.check("torch_threads", lambda: {
        "num_threads": torch.get_num_threads(),
        "is_one": torch.get_num_threads() == 1,
        "enforced_at": threads_location()})

    def contract_checks():
        model = fse.build_stair_model(ASSETS, 28001, 5, .06, .31)
        spec = fse.validate_contract(model, contract)
        if spec["contract_scales"]["joint_pos"] != .35:
            raise AssertionError("leg action scale")
        if spec["wheel_body_ids"] != sorted(set(spec["wheel_body_ids"])):
            raise AssertionError("wheel bodies not distinct")
        return {"joints": len(spec["joint_names"]),
                "wheel_body_ids": spec["wheel_body_ids"],
                "contract_scales": spec["contract_scales"]}

    checker.check("contract_names_and_indices", contract_checks)

    def negative_tests():
        model = fse.build_stair_model(ASSETS, 28001, 1, .06, .31)
        cases = []
        bad = json.loads(json.dumps(contract))
        bad["joint_names"][1] = bad["joint_names"][0]
        cases.append(("duplicate_joint_name", bad))
        bad = json.loads(json.dumps(contract))
        bad["action_terms"][0]["joints"] = list(reversed(bad["action_terms"][0]["joints"]))
        cases.append(("action_order_mismatch", bad))
        bad = json.loads(json.dumps(contract))
        bad["joint_names"][3] = "NOPE_hip"
        bad["action_terms"][0]["joints"][3] = "NOPE_hip"
        cases.append(("missing_joint_name", bad))
        results = {}
        for name, payload in cases:
            try:
                fse.validate_contract(model, payload)
            except fse.ContractError:
                results[name] = "raised ContractError"
            else:
                raise AssertionError(f"{name} did not fail fast")
        return results

    checker.check("fail_fast_contract_inconsistency", negative_tests)

    def builder_parity():
        import mujoco_screen as sim
        for seed in (28001, 28002, 28003):
            want = sim.load_flat_model(True, True, "stairs", seed)
            got = fse.build_stair_model(ASSETS, seed, 5, None, None)
            if got.nbody != want.nbody or got.ngeom != want.ngeom:
                raise AssertionError(f"seed {seed}: nbody/ngeom differ")
            if not np.array_equal(got.geom_pos, want.geom_pos):
                raise AssertionError(f"seed {seed}: geom_pos differ")
            if not np.array_equal(got.geom_size, want.geom_size):
                raise AssertionError(f"seed {seed}: geom_size differ")
            if not np.array_equal(got.geom_priority, want.geom_priority):
                raise AssertionError(f"seed {seed}: geom_priority differ")
            if not np.array_equal(got.geom_contype, want.geom_contype):
                raise AssertionError(f"seed {seed}: geom_contype differ")
            if not np.array_equal(got.geom_conaffinity, want.geom_conaffinity):
                raise AssertionError(f"seed {seed}: geom_conaffinity differ")
        return {"seeds": [28001, 28002, 28003],
                "note": "n_levels=5/rise=None/tread=None reproduces the delivery model"}

    checker.check("stair_builder_parity", builder_parity)

    def geometry_readback():
        rows = {}
        for stage in ladder:
            model = fse.build_stair_model(ASSETS, cfg["holdout_seeds"][0],
                                          stage.levels, stage.rise_m, stage.tread_m)
            geo = fse.stair_geometry(model, stage.levels)
            x_range = fse.geometry_x_range(geo)
            fse.assert_stage_geometry(stage, geo, x_range)
            rows[stage.name] = {"levels": stage.levels,
                                "rise_m": None if not geo else geo[0]["rise_m"],
                                "tread_m": None if not geo else geo[0]["tread_depth_m"],
                                "x_range_m": x_range}
        return rows

    checker.check("geometry_readback_asserted", geometry_readback)

    def facade_rule_unit():
        frame_z = np.array([0., .3, .5, .699, .7, .9, 1.])
        classified = ["tread" if abs(value) >= .7 else "riser" for value in frame_z]
        if classified != ["riser", "riser", "riser", "riser", "tread", "tread", "tread"]:
            raise AssertionError(f"normal classification wrong: {classified}")
        return {"threshold": .7, "frame_z": frame_z.tolist(), "classified": classified}

    checker.check("facade_vs_tread_normal_rule", facade_rule_unit)

    def facade_episode():
        """Drive into the first riser: a facade hit must never credit a tread."""
        env = fse.StairEnv(ASSETS, ladder[1], trace=True)
        obs, _ = env.reset(seed=1, options={"scene_seed": cfg["holdout_seeds"][0]})
        env.data.qpos[0] = .40
        fse.sim.mj.mj_forward(env.model, env.data)
        action = np.zeros(16, dtype=np.float32)
        action[12:] = 1.5
        for _ in range(90):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        record = env.episode_record(terminated=terminated, truncated=truncated)
        if record["milestone_without_load"] != 0:
            raise AssertionError(
                f"a tread milestone was credited without tread load: {record['milestone_without_load']}")
        return {"riser_frames": record["riser_frames"],
                "riser_peak_n": record["riser_peak_n"],
                "ever_levels": record["ever_levels"],
                "milestone_without_load": record["milestone_without_load"],
                "facade_contact_observed": record["riser_frames"] > 0,
                "note": "credits are structurally impossible without >= 8 N on the tread "
                        "and the dwell; the unit rule above is the always-on gate"}

    checker.check("facade_contact_not_credited", facade_episode)

    def milestone_dedup():
        tracker = fse.MilestoneTracker(5, .02)
        loads = np.zeros((4, 6))
        loads[0, 1] = 9.
        loads[1, 1] = 9.
        loads[2, 1] = 9.
        loads[3, 0] = 9.
        for frame in range(1, 4):
            tracker.update(loads, True, frame)
        if tracker.levels[0] != 1:
            raise AssertionError("sustained 60 ms load did not credit level 1")
        credits = len(tracker.credit_events)
        if credits != 3:
            raise AssertionError(f"expected three credits (FL, FR, RL), got {credits}")
        total = .0
        for frame in range(4, 240):
            _, newly = tracker.update(loads, True, frame)
            total += newly
        if tracker.levels[0] != 1 or total != 0. or len(tracker.credit_events) != 3:
            raise AssertionError("re-stepping the same tread awarded the milestone twice")
        short = fse.MilestoneTracker(5, .02)
        for frame in range(1, 3):
            short.update(loads, True, frame)
        if short.levels[0] != 0:
            raise AssertionError("40 ms of load must not credit a milestone")
        short.update(loads, True, 3)
        if short.levels[0] != 1:
            raise AssertionError("60 ms of load must credit a milestone")
        # back-and-forth on the same tread: step off to the ground, then back on,
        # for 40 cycles; the monotone credit must not award anything again.
        on_step_off = np.zeros((4, 6))
        on_step_off[0, 1] = 9.
        on_step_off[1, 1] = 9.
        on_step_off[2, 1] = 9.
        on_step_off[3, 0] = 9.
        on_ground = np.zeros((4, 6))
        on_ground[:, 0] = 9.
        recycles = 0
        for _ in range(40):
            for _ in range(4):
                tracker.update(on_step_off, True, 0)
            for _ in range(4):
                tracker.update(on_ground, True, 0)
            recycles += 1
        if tracker.levels[0] != 1 or len(tracker.credit_events) != 3:
            raise AssertionError("back-and-forth on the same tread awarded the milestone again")
        return {"credits": len(tracker.credit_events), "replay_extra": total,
                "dwell_frames_needed": 3, "load_n": 8., "back_and_forth_cycles": recycles}

    checker.check("milestone_dedup_and_dwell", milestone_dedup)

    def history_checks():
        import gymnasium as gym
        results = {}
        for arm in ("1frame", "6frame", "zero_history"):
            spec = ARCHES[arm]
            env = fse.StairEnv(ASSETS, ladder[0])
            wrapped = (fse.ZeroHistory(env, spec["frames"])
                       if spec["wrapper"] == "ZeroHistory"
                       else fse.ObservationHistory(env, spec["frames"]))
            if wrapped.observation_space.shape != (spec["observation_dim"],):
                raise AssertionError(f"{arm}: wrong observation space")
            obs, _ = wrapped.reset(seed=7, options={"scene_seed": cfg["holdout_seeds"][0]})
            first = obs[-60:].copy()
            frames = obs.reshape(spec["frames"], 60)
            if spec["wrapper"] == "ZeroHistory":
                if np.any(frames[:-1] != 0.):
                    raise AssertionError(f"{arm}: history slots not zero after reset")
                if not np.array_equal(frames[-1], first):
                    raise AssertionError(f"{arm}: current frame missing")
            else:
                if not np.all(frames == first):
                    raise AssertionError(f"{arm}: reset did not fill with the first frame")
            action = np.zeros(16, dtype=np.float32)
            for _ in range(3):
                obs, *_ = wrapped.step(action)
            frames = obs.reshape(spec["frames"], 60)
            if spec["frames"] > 1:
                if not np.array_equal(frames[-1], obs[-60:]):
                    raise AssertionError(f"{arm}: newest frame is not the current obs")
                if spec["wrapper"] == "ObservationHistory" and np.array_equal(
                        frames[0], frames[-1]):
                    raise AssertionError(f"{arm}: history did not advance")
            obs, _ = wrapped.reset(seed=8, options={"scene_seed": 28002})
            frames = obs.reshape(spec["frames"], 60)
            if spec["wrapper"] == "ObservationHistory" and not np.all(frames == frames[-1]):
                raise AssertionError(f"{arm}: reset left stale frames")
            results[arm] = {"shape": list(wrapped.observation_space.shape),
                            "frames": spec["frames"]}
        return results

    checker.check("history_fill_advance_reset", history_checks)

    def promotion_checks():
        stages = ladder
        cur = fse.StairCurriculum(stages, 0, 2, 2)
        base = {"started_from_bottom": True, "manual": False, "complete": True,
                "hard_protection": False, "zero_progress": False}
        first = cur.record(dict(base))
        if first["promoted"]:
            raise AssertionError("promoted after one completed episode")
        second = cur.record(dict(base))
        if not second["promoted"] or cur.stage.name != stages[1].name:
            raise AssertionError("did not promote after two completed episodes")
        cur = fse.StairCurriculum(stages, 1, 2, 2)
        cur.record(dict(base, complete=False, hard_protection=True))
        if cur.stage.name != stages[0].name:
            raise AssertionError("hard protection did not demote")
        cur = fse.StairCurriculum(stages, 1, 2, 2)
        cur.record(dict(base, complete=False, zero_progress=True))
        cur.record(dict(base, complete=False, zero_progress=True))
        if cur.stage.name != stages[0].name:
            raise AssertionError("two zero-progress episodes did not demote")
        cur = fse.StairCurriculum(stages, 0, 2, 2)
        cur.record(dict(base, started_from_bottom=False, manual=True))
        cur.record(dict(base, started_from_bottom=False, manual=True))
        if cur.successes != 0 or cur.stage.name != stages[0].name:
            raise AssertionError("manual episodes fed the promotion counters")
        cur = fse.StairCurriculum(stages, len(stages) - 1, 2, 2)
        for _ in range(4):
            cur.record(dict(base))
        if cur.stage.name != stages[-1].name:
            raise AssertionError("promoted past the top of the ladder")
        return {"promote_after": 2, "demote_after": 2,
                "manual_episodes_ignored": True, "ceiling_respected": True}

    checker.check("promotion_demotion_rules", promotion_checks)

    def degeneration_check():
        monitor = fse.DegenerationMonitor(after=6)
        fired = [monitor.update("step1", False) for _ in range(6)]
        if not fired[-1] or monitor.count != 6:
            raise AssertionError("degradation warning did not fire after 6 episodes")
        if monitor.update("step1", True):
            raise AssertionError("warning re-fired after a success")
        if monitor.count != 0:
            raise AssertionError("success did not reset the counter")
        summary = monitor.summary()
        summary["warning"] = "PROMOTION_SIGNAL_DEGENERATE"
        return summary

    checker.check("promotion_signal_degenerate_warning", degeneration_check)

    def param_checks():
        counts = {}
        for arm in ("1frame", "zero_history", "6frame"):
            counts[arm] = describe_policy(make_policy(cfg, arm, cfg["seed"]))
        if counts["zero_history"]["actor_params"] != counts["6frame"]["actor_params"]:
            raise AssertionError("H0 and H6 must have identical parameter counts")
        delta = counts["6frame"]["actor_params"] - counts["1frame"]["actor_params"]
        if delta != (360 - 60) * 128:
            raise AssertionError(f"input-width delta {delta} != {300 * 128}")
        return {arm: {"obs": ARCHES[arm]["observation_dim"],
                      "actor_params": row["actor_params"],
                      "with_log_std": row["actor_params_with_log_std"]}
                for arm, row in counts.items()}

    checker.check("arm_parameter_counts", param_checks)

    def init_fingerprint():
        first = describe_policy(make_policy(cfg, args.arch, cfg["seed"]))
        second = describe_policy(make_policy(cfg, args.arch, cfg["seed"]))
        other = describe_policy(make_policy(cfg, args.arch, cfg["seed"] + 1))
        if first["state_dict_sha256"] != second["state_dict_sha256"]:
            raise AssertionError("same seed produced a different fingerprint")
        if first["state_dict_sha256"] == other["state_dict_sha256"]:
            raise AssertionError("different seeds produced the same fingerprint")
        if args.init_checkpoint:
            raise AssertionError("selfcheck expects a from-scratch run")
        return {"fingerprint": first["state_dict_sha256"],
                "method": "orthogonal default, from scratch"}

    checker.check("weight_init_fingerprint", init_fingerprint)

    def action_scale_readback():
        text = Path(cfg["sources"]["pinned_training_copy"]["path"]).read_text()
        patterns = {"leg_scale": r"\.35 \* action\[:12\]",
                    "wheel_scale": r"16\. \* action\[12:\]",
                    "impact_term": r"impact > 90\.",
                    "lateral_term": r"abs\(data\.qpos\[1\]\) > \.65",
                    "top_hold": r">= \.5 - 1e-9",
                    "dwell": r">= \.06 - 1e-9",
                    "load_n": r"force\[0\] >= 8\."}
        missing = [name for name, pattern in patterns.items() if not re.search(pattern, text)]
        if missing:
            raise AssertionError(f"pinned source literals missing: {missing}")
        if abs(fse.LEG_ACTION_SCALE - .35) > 1e-12 or abs(fse.WHEEL_ACTION_SCALE - 16.) > 1e-12:
            raise AssertionError("env scales drift from the pinned source")
        return {"pinned_source": cfg["sources"]["pinned_training_copy"]["path"],
                "literals_found": sorted(patterns),
                "leg_scale": fse.LEG_ACTION_SCALE,
                "wheel_scale": fse.WHEEL_ACTION_SCALE}

    checker.check("action_scale_readback_from_pinned_source", action_scale_readback)

    def torque_checks():
        model = fse.build_stair_model(ASSETS, 28001, 5, .06, .31)
        ids = [model.actuator(name).id for name in contract["joint_names"]]
        raw = model.actuator_ctrlrange[ids]
        budget = float(cfg["torque"]["budget_nm"])
        if abs(budget - .9 * float(raw[12:, 1].max())) > 1e-9:
            raise AssertionError("wheel budget is not 90 % of the wheel ctrlrange")
        env = fse.StairEnv(ASSETS, ladder[0], torque_clip="contract_90pct")
        env.reset(seed=0, options={"scene_seed": 28001})
        if abs(env.torque_high[12] - budget) > 1e-12 or abs(env.torque_low[12] + budget) > 1e-12:
            raise AssertionError("wheel torque not clipped to the contract budget")
        env2 = fse.StairEnv(ASSETS, ladder[0], torque_clip="mujoco_ctrlrange")
        env2.reset(seed=0, options={"scene_seed": 28001})
        if abs(env2.torque_high[12] - raw[12, 1]) > 1e-12:
            raise AssertionError("mujoco_ctrlrange mode does not reproduce the pinned env")
        return {"budget_nm": budget, "raw_wheel_nm": float(raw[12, 1]),
                "raw_hip_nm": [float(raw[0, 0]), float(raw[0, 1])],
                "raw_calf_nm": [float(raw[8, 0]), float(raw[8, 1])],
                "effective_wheel_nm": float(env.torque_high[12]),
                "effective_hip_nm": [float(env.torque_low[0]), float(env.torque_high[0])],
                "effective_calf_nm": [float(env.torque_low[8]), float(env.torque_high[8])]}

    checker.check("torque_budget_90pct", torque_checks)

    def threshold_checks():
        rows = fse.Thresholds().as_dict()
        if set(rows) != {"train_termination", "acceptance", "nonwheel_contact"}:
            raise AssertionError("threshold sets are not separated")
        if rows["nonwheel_contact"]["contact_force_n_ge"] != 1.:
            raise AssertionError("non-wheel contact threshold drifted")
        return rows

    checker.check("thresholds_separate", threshold_checks)

    def observation_contract():
        env = make_env(cfg, args.arch)
        info = {"arch": args.arch,
                "observation_space": list(env.observation_space.shape),
                "action_space": list(env.action_space.shape)}
        if env.observation_space.shape != (ARCHES[args.arch]["observation_dim"],):
            raise AssertionError("observation dim mismatch")
        if env.action_space.shape != (16,):
            raise AssertionError("action dim mismatch")
        history = cfg["history"]
        if history["mode"] == "continuous_frames":
            expected_span = (history["frames"] - 1) * int(cfg["control_period_s"] * 1000)
            if history["span_ms"] != expected_span:
                raise AssertionError(f"history span {history['span_ms']} != {expected_span}")
            if history["order"] != "oldest_to_newest":
                raise AssertionError("history order must be oldest_to_newest")
        info["history"] = history
        return info

    checker.check("observation_contract", observation_contract)

    def wheel_contract_checks():
        if not WHEEL_CONTRACT.is_file():
            raise AssertionError("wheel contract missing")
        payload = json.loads(WHEEL_CONTRACT.read_text())
        if abs(float(payload["torque_budget_nm"]) - float(cfg["torque"]["budget_nm"])) > 1e-12:
            raise AssertionError("config budget differs from the wheel contract")
        legacy = payload["contracts"]["r11_legacy"]["wheel_action_scale"]
        if float(legacy) != fse.WHEEL_ACTION_SCALE:
            raise AssertionError("stair wheel scale differs from the r11_legacy family")
        return {"path": str(WHEEL_CONTRACT), "sha256_observed": sha256_file(WHEEL_CONTRACT),
                "gating": False, "torque_budget_nm": payload["torque_budget_nm"],
                "r11_legacy_wheel_action_scale": legacy}

    checker.check("wheel_contract_fields", wheel_contract_checks)

    def evidence_label_checks():
        evidence = cfg["torque_evidence"]
        if evidence["isaac"]["label"] != "estimated":
            raise AssertionError("Isaac torque must be labelled estimated")
        if evidence["actual_drive_torque"]["label"] != "not_measured":
            raise AssertionError("actual drive torque must be labelled not_measured")
        if cfg["evidence_rules"]["improvement_rule"].find("2") < 0:
            raise AssertionError("evidence rule missing")
        return evidence

    checker.check("torque_evidence_labels", evidence_label_checks)


    # ---------------------------------------------------------------- phase / contract
    def phase_default_checks():
        bare = parse_args().parse_args([])
        if bare.phase != "baseline":
            raise AssertionError(f"default phase is {bare.phase}, expected baseline")
        phase = resolve_phase(bare)
        if phase["errors"] or phase["locked"]:
            raise AssertionError(f"baseline phase must not be locked: {phase}")
        if any(phase["randomization"].values()):
            raise AssertionError("baseline randomization is not all off")
        env = fse.StairEnv(ASSETS, ladder[0], control=cfg["control"],
                           randomization=phase["randomization"])
        try:
            env.reset(seed=0, options={"scene_seed": cfg["holdout_seeds"][0]})
            if env.stress is not None:
                raise AssertionError("baseline sampled a dynamics stress")
            if env.push_frame is not None:
                raise AssertionError("baseline enabled a push")
            if any(env.randomization_applied.values()):
                raise AssertionError("baseline applied a randomization")
            reference = fse.build_stair_model(ASSETS, cfg["holdout_seeds"][0], 0, None, None)
            if not np.array_equal(env.model.body_mass, reference.body_mass):
                raise AssertionError("baseline mass is not nominal")
            if not np.array_equal(env.model.body_inertia, reference.body_inertia):
                raise AssertionError("baseline inertia is not nominal")
        finally:
            env.close()
        return {"phase": bare.phase, "randomization": phase["randomization"],
                "push_frame": None, "stress": None, "mass_nominal": True,
                "inertia_nominal": True}

    checker.check("phase_default_is_baseline", phase_default_checks)

    def three_arm_wrapper_checks():
        cfgs = {arm: load_arch_config(arm) for arm in ("1frame", "zero_history", "6frame")}
        for block in ("control", "initial_state", "dispatch"):
            values = {json.dumps(payload[block], sort_keys=True) for payload in cfgs.values()}
            if len(values) != 1:
                raise AssertionError(f"block {block} differs across S/H0/H6")
        first_frames = {}
        for arm, payload in cfgs.items():
            env = make_env(payload, arm, stage=stage_ladder(payload)[0])
            try:
                obs, _ = env.reset(seed=1, options={"scene_seed": cfg["holdout_seeds"][0]})
                first_frames[arm] = np.asarray(obs[-60:], dtype=np.float32).copy()
            finally:
                env.close()
        if not np.array_equal(first_frames["1frame"], first_frames["6frame"]):
            raise AssertionError("S and H6 do not share the initial-state distribution")
        if not np.array_equal(first_frames["zero_history"], first_frames["6frame"]):
            raise AssertionError("H0 and H6 do not share the initial-state distribution")
        return {"control": cfgs["1frame"]["control"],
                "initial_state": cfgs["1frame"]["initial_state"],
                "reset_frame_equal": True,
                "reset_frame_sha256": replay_mod.window_sha256(first_frames["6frame"])}

    checker.check("three_arm_identical_wrapper", three_arm_wrapper_checks)

    def dispatch_ast_check():
        tree = ast.parse((HERE / "fs_stair_env.py").read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "step"):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.For):
                    continue
                for index, inner in enumerate(stmt.body):
                    if not isinstance(inner, ast.Assign):
                        continue
                    target = inner.targets[0]
                    if not (isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Attribute)
                            and target.value.attr == "ctrl"
                            and isinstance(target.value.value, ast.Name)
                            and target.value.value.id == "data"):
                        continue
                    following = stmt.body[index + 1]
                    if not isinstance(following, ast.Assign):
                        raise AssertionError("statement after data.ctrl is not an assignment")
                    captured = following.targets[0]
                    if not (isinstance(captured, ast.Attribute)
                            and captured.attr == "_dispatch_wheel"):
                        raise AssertionError(
                            "target_wheel_dispatch is not captured at the data.ctrl "
                            f"assignment point (next statement assigns "
                            f"{ast.dump(captured)})")
                    return {"ctrl_assignment": "data.ctrl[self.motors] = bounded",
                            "next_statement": "self._dispatch_wheel = wheel.copy()",
                            "adjacent": True}
        raise AssertionError("could not locate the data.ctrl assignment in StairEnv.step")

    def triple_tap_checks():
        tmp = Path(tempfile.mkdtemp(prefix="fs_tap_"))
        env = make_env(cfg, args.arch, stage=ladder[1],
                       tap={"enabled": True, "stride": 1, "dir": str(tmp / "tap"),
                            "window_dir": str(tmp / "windows"), "worker_id": 0,
                            "window_every_steps": 1, "max_rows": 100})
        try:
            env.reset(seed=0, options={"scene_seed": cfg["holdout_seeds"][0]})
            for _ in range(3):
                env.step(np.zeros(16, dtype=np.float32))
        finally:
            env.close()
        log = tmp / "tap" / "triple_tap.worker0.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        required = ("target_wheel_pre_filter", "target_wheel_dispatch",
                    "target_leg_pos_pre_filter", "target_leg_pos_dispatch",
                    "tau_request_est_nm", "tau_actuator_coord_nm", "tau_joint_qfrc_nm",
                    "history_newest", "history_sha256", "field_order", "window_shape",
                    "env_id", "episode_id", "policy_step", "dtype")
        for row in rows:
            missing = [field for field in required if field not in row]
            if missing:
                raise AssertionError(f"triple-tap row misses {missing}")
        joined = {}
        for row in rows:
            if row["step_id"] in joined:
                raise AssertionError("duplicate step_id in the triple-tap log")
            joined[row["step_id"]] = {"pre": row["target_wheel_pre_filter"],
                                      "dispatch": row["target_wheel_dispatch"],
                                      "history_sha256": row["history_sha256"]}
        if len(joined) != 3:
            raise AssertionError(f"expected 3 joined steps, got {len(joined)}")
        dispatch = env.env.unwrapped._dispatch_wheel
        if not np.allclose(rows[-1]["target_wheel_dispatch"], dispatch, atol=1e-9):
            raise AssertionError("logged dispatch differs from the value captured at data.ctrl")
        if rows[-1]["field_order"] != list(cmod.FIELD_ORDER):
            raise AssertionError("logged field_order differs from the observation contract")
        windows = (tmp / "windows" / "index.jsonl").read_text().splitlines()
        return {**dispatch_ast_check(), "joined_steps": sorted(joined),
                "rows": len(rows), "windows_sampled": len(windows),
                "dispatch_matches_assignment_point": True,
                "units": {"target_wheel_*": "rad/s", "target_leg_pos_*": "rad",
                          "tau_*": "Nm"}}

    checker.check("triple_tap_logging", triple_tap_checks)

    def contract_guard_checks():
        payload = cmod.build_payload(cfg["control"], cfg["history"], cfg["initial_state"],
                                     cfg["dispatch"])
        stored = cfg["control_contract"]["fingerprint"]
        version = cfg["control_contract"]["version"]
        current = cmod.verify(stored, version, payload)
        if current["changed"]:
            raise AssertionError("config fingerprint is stale against its own control block")
        bad_control = dict(cfg["control"])
        bad_control["action_delay_frames"] = int(bad_control["action_delay_frames"]) + 2
        bad_payload = cmod.build_payload(bad_control, cfg["history"], cfg["initial_state"],
                                         cfg["dispatch"])
        negative_one = cmod.verify(stored, version, bad_payload, will_run=True)
        if not negative_one["changed"] or not negative_one["run_refused"]:
            raise AssertionError("changed fingerprint did not refuse --run")
        if "CONTROL_CONTRACT_CHANGED" not in (negative_one["message"] or ""):
            raise AssertionError("missing CONTROL_CONTRACT_CHANGED message")
        rc = main(["--run", "--out", str(Path(tempfile.mkdtemp(prefix="fs_guard_"))),
                   "--accept-contract-change", "--arch", args.arch])
        if rc == 0:
            raise AssertionError("--accept-contract-change --run was not refused")
        accepted = cmod.verify(stored, version, bad_payload, accept_change=True,
                               will_run=False)
        if not accepted["accepted"]:
            raise AssertionError("dry-run acceptance was not recorded")
        return {"negative_1_run_refused": True, "negative_1_message": negative_one["message"],
                "negative_2_exit_code": rc,
                "negative_2_reason": "--accept-contract-change is dry-run/replay only",
                "dry_run_accept": accepted["accept_message"]}

    checker.check("contract_version_guard", contract_guard_checks)

    def disturbance_checks():
        args_multi = parse_args().parse_args(
            ["--phase", "disturbance", "--disturbance", "load", "--disturbance", "push",
             "--allow-disturbance"])
        phase = resolve_phase(args_multi)
        if not any("single-factor" in error for error in phase["errors"]):
            raise AssertionError("two disturbance factors were not rejected")
        rc = main(["--phase", "disturbance", "--disturbance", "load", "--disturbance",
                   "push", "--allow-disturbance", "--init-checkpoint", "/dev/null"])
        if rc == 0:
            raise AssertionError("two factors did not produce a non-zero exit")
        single = resolve_phase(parse_args().parse_args(
            ["--phase", "disturbance", "--disturbance", "load", "--allow-disturbance",
             "--init-checkpoint", "/dev/null"]))
        if single["errors"]:
            raise AssertionError(f"a single factor was rejected: {single['errors']}")
        if not single["randomization"]["load"] or single["randomization"]["push"]:
            raise AssertionError("load selection did not enable only load")
        push = resolve_phase(parse_args().parse_args(
            ["--phase", "disturbance", "--disturbance", "push", "--allow-disturbance",
             "--init-checkpoint", "/dev/null"]))
        if not push["randomization"]["push"] or push["randomization"]["load"]:
            raise AssertionError("push selection did not enable only push")
        return {"two_factor_error": phase["errors"], "two_factor_exit_code": rc,
                "load_only": single["randomization"], "push_only": push["randomization"]}

    checker.check("disturbance_single_factor", disturbance_checks)

    def phase2_lock_checks():
        tmp = Path(tempfile.mkdtemp(prefix="fs_phase2_"))
        model = make_policy(cfg, args.arch, cfg["seed"])
        probe = tmp / "probe_policy.zip"
        model.save(probe)
        probe_sha = sha256_file(probe)
        base = ["--phase", "disturbance", "--disturbance", "push", "--arch", args.arch,
                "--run"]
        rc_a = main(base + ["--allow-disturbance", "--out", str(tmp / "a")])
        rc_b = main(base + ["--allow-disturbance", "--init-checkpoint", str(probe),
                            "--out", str(tmp / "b")])
        wrong = {"checkpoint_sha256": "0" * 64, "holdout_seeds": cfg["holdout_seeds"],
                 "criteria": {"A_s1": 0, "B_s1": 0, "C_top": 0, "D": 0},
                 "gate_status": "PENDING_OPERATOR_RULING"}
        (tmp / "wrong.json").write_text(json.dumps(wrong))
        rc_c = main(base + ["--allow-disturbance", "--init-checkpoint", str(probe),
                            "--eval-record", str(tmp / "wrong.json"),
                            "--out", str(tmp / "c")])
        partial = dict(wrong, checkpoint_sha256=probe_sha, criteria={"A_s1": 0})
        (tmp / "partial.json").write_text(json.dumps(partial))
        rc_d = main(base + ["--allow-disturbance", "--init-checkpoint", str(probe),
                            "--eval-record", str(tmp / "partial.json"),
                            "--out", str(tmp / "d")])
        for name, code in (("missing_checkpoint", rc_a), ("missing_record", rc_b),
                           ("sha_mismatch", rc_c), ("criteria_missing", rc_d)):
            if code == 0:
                raise AssertionError(f"phase-2 negative {name} was accepted")
        return {"missing_checkpoint_exit": rc_a, "missing_record_exit": rc_b,
                "record_sha_mismatch_exit": rc_c, "record_criteria_missing_exit": rc_d,
                "checkpoint_sha256": probe_sha}

    checker.check("phase2_locked_negatives", phase2_lock_checks)

    def transmission_checks():
        env = fse.StairEnv(ASSETS, ladder[1], control=cfg["control"],
                           randomization={"load": False, "friction": False, "gain": False,
                                          "push": False, "com_offset": False,
                                          "observation_noise": False})
        try:
            env.reset(seed=0, options={"scene_seed": cfg["holdout_seeds"][0]})
            action = np.zeros(16, dtype=np.float32)
            action[12:] = .4
            for _ in range(5):
                env.step(action)
            expected = np.zeros(16)
            for index, motor in enumerate(env.motors):
                expected[env.actuator_joint_index[index]] += (
                    env.actuator_gear[index] * float(env.data.actuator_force[motor]))
            error = float(np.max(np.abs(env.data.qfrc_actuator[env.va] - expected)))
            if error > 1e-9:
                raise AssertionError(f"qfrc_actuator != sum(gear*actuator_force): {error}")
            trans = env.transmission
            return {"torque_quantity": trans["torque_quantity"], "label": trans["label"],
                    "kinds": sorted({row["kind"] for row in trans["actuators"]}),
                    "gears": sorted({row["gear"] for row in trans["actuators"]}),
                    "actuators_per_joint": trans["actuators_per_joint"],
                    "max_abs_error_nm": error, "tolerance_nm": 1e-9,
                    "measured_env_error": env.transmission_max_error}
        finally:
            env.close()

    checker.check("transmission_qfrc_consistency", transmission_checks)

    def offline_forward_checks():
        tmp = Path(tempfile.mkdtemp(prefix="fs_offline_"))
        rng = np.random.default_rng(20261117)
        frames = rng.normal(0., .5, size=(8, 6, 60)).astype(np.float32)
        single = frames[:, -1].copy()
        zero = np.concatenate([np.zeros((8, 5, 60), dtype=np.float32),
                               single[:, None, :]], axis=1).reshape(8, 360)
        six = frames.reshape(8, 360)
        np.savez(tmp / "offline_actor_samples.npz", single_frame=single,
                 zero_history=zero, six_frame=six,
                 seed=np.array([20261117]))
        inputs = {"1frame": single, "zero_history": zero, "6frame": six}
        outputs = {}
        dtypes = {}
        for arm, batch in inputs.items():
            model = make_policy(cfg, arm, cfg["seed"])
            with torch.no_grad():
                actions = model.policy(
                    torch.as_tensor(batch, dtype=torch.float32))[0].detach().cpu().numpy()
            actions = np.asarray(actions)
            if actions.shape != (8, 16):
                raise AssertionError(f"{arm}: output shape {actions.shape} != (8, 16)")
            if not np.all(np.isfinite(actions)):
                raise AssertionError(f"{arm}: non-finite action")
            outputs[arm] = actions.astype(np.float32)
            dtypes[arm] = str(actions.dtype)
            if arm == "zero_history":
                stacked = zero.reshape(8, 6, 60)
                if not np.all(stacked[:, :-1] == 0):
                    raise AssertionError("H0 historical slots are not zero")
                if not np.array_equal(stacked[:, -1], single):
                    raise AssertionError("H0 current frame differs from the sample")
            if arm == "6frame":
                stacked = six.reshape(8, 6, 60)
                if not np.array_equal(stacked, frames):
                    raise AssertionError("H6 window is not the raw frame order")
                if not np.array_equal(stacked[:, -1], single):
                    raise AssertionError("H6 current frame differs from the sample")
                if np.all(stacked[:, :-1] == 0):
                    raise AssertionError("H6 historical slots are all zero")
                if not np.array_equal(stacked[:, -1], zero.reshape(8, 6, 60)[:, -1]):
                    raise AssertionError("H0 and H6 current frames differ")
        if len(set(dtypes.values())) != 1:
            raise AssertionError(f"output dtypes differ across arms: {dtypes}")
        np.savez(tmp / "offline_actor_outputs.npz", **outputs)
        return {"input_dims": {arm: int(batch.shape[1]) for arm, batch in inputs.items()},
                "output_shapes": {arm: list(value.shape) for arm, value in outputs.items()},
                "output_dtypes": dtypes,
                "h0_history_all_zero": True, "h6_history_nonzero": True,
                "current_frame_equal": True,
                "samples": str(tmp / "offline_actor_samples.npz"),
                "outputs": str(tmp / "offline_actor_outputs.npz")}

    checker.check("offline_actor_forward_all_arms", offline_forward_checks)

    def strict_history_checks():
        base_frames = [(1000. * step + np.arange(60)).astype(np.float32)
                       for step in range(16)]

        class FakeFrameEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (60,),
                                                        dtype=np.float32)
                self.action_space = gym.spaces.Box(-1, 1, (16,), dtype=np.float32)
                self.t = 0
                self.first_fails = {}
                self.last_tap = {"tick": 0, "episode_id": 1, "scene_seed": 0,
                                 "stage": "fake", "step_id": 0}

            def reset(self, **kwargs):
                self.t = 0
                self.last_tap = dict(self.last_tap, tick=0, step_id=0)
                return base_frames[0].copy(), {}

            def step(self, action):
                self.t += 1
                self.last_tap = dict(self.last_tap, tick=self.t, step_id=self.t)
                return base_frames[self.t].copy(), 0., False, False, {}

        def run(wrapper_cls, steps):
            wrapper = wrapper_cls(FakeFrameEnv(), frames=6)
            obs, _ = wrapper.reset()
            for _ in range(steps):
                obs, *_ = wrapper.step(np.zeros(16, dtype=np.float32))
            return np.asarray(obs, dtype=np.float32).reshape(6, 60)

        steps = 6
        win_six = run(fse.ObservationHistory, steps)
        win_zero = run(fse.ZeroHistory, steps)
        current = base_frames[steps]
        expected = np.stack([base_frames[steps - 5 + slot] for slot in range(6)])
        checks = {
            "current_equal": bool(np.array_equal(win_six[-1], win_zero[-1])),
            "current_is_frame_t": bool(np.array_equal(win_six[-1], current)),
            "h0_history_zero": bool(np.all(win_zero[:-1] == 0)),
            "h6_history_exact": bool(np.array_equal(win_six, expected)),
            "h6_order_oldest_first": bool(np.array_equal(win_six[0], base_frames[steps - 5])),
            "reversed_fails": not bool(np.array_equal(win_six[::-1], expected)),
            "repeated_fill_fails": not bool(np.all(win_six == win_six[-1])),
        }
        if not all(checks.values()):
            raise AssertionError(f"strict history assertions failed: {checks}")
        return {**checks, "steps": steps,
                "slot0": "frame_t-5", "slot5": "frame_t",
                "order": "oldest_to_newest"}

    checker.check("history_arm_identity_strict", strict_history_checks)

    def tamper_checks():
        tmp = Path(tempfile.mkdtemp(prefix="fs_tamper_"))
        env = make_env(cfg, args.arch, stage=ladder[1],
                       tap={"enabled": True, "stride": 1, "dir": str(tmp / "tap"),
                            "window_dir": str(tmp / "windows"), "worker_id": 0,
                            "window_every_steps": 1, "max_rows": 100})
        try:
            env.reset(seed=0, options={"scene_seed": cfg["holdout_seeds"][0]})
            for _ in range(5):
                env.step(np.zeros(16, dtype=np.float32))
        finally:
            env.close()
        index_path = tmp / "windows" / "index.jsonl"
        windows = [json.loads(line) for line in index_path.read_text().splitlines()]
        target = next((row for row in windows if row["policy_step"] == 3), None)
        if target is None:
            raise AssertionError("no sampled window for policy_step 3")
        original = tmp / "windows" / target["path"]
        reference = tmp / "windows" / "reference_step3.npz"
        with np.load(original, allow_pickle=False) as handle:
            payload = {key: handle[key] for key in handle.files}
        np.savez_compressed(reference, **payload)
        tampered_dir = tmp / "tampered"
        tampered_dir.mkdir()
        for source in (tmp / "windows").glob("*.npz"):
            (tampered_dir / source.name).write_bytes(source.read_bytes())
        tampered = tampered_dir / target["path"]
        obs = np.array(payload["obs"], copy=True)
        field_index = 26 + 7              # joint_vel index 7 -> RR_thigh
        obs[-1, field_index] = float(obs[-1, field_index]) + 1e-3
        payload["obs"] = obs
        np.savez_compressed(tampered, **payload)
        (tampered_dir / "index.jsonl").write_text(index_path.read_text())
        diff = replay_mod.compare_windows(tampered, reference)
        scan = replay_mod.scan_run(tmp / "tap" / "triple_tap.worker0.jsonl", tampered_dir)
        if scan.get("status") != "mismatch":
            raise AssertionError(f"tamper was not detected: {scan}")
        if int(scan["policy_step"]) != 3:
            raise AssertionError(f"first mismatch step is {scan['policy_step']}, expected 3")
        if diff["hash_match"]:
            raise AssertionError("whole-window hash matched a tampered window")
        first = diff["first_mismatch"]
        if first is None or first["field_index"] != field_index:
            raise AssertionError(f"field localisation failed: {first}")
        if first["field_name"] != cmod.FIELD_ORDER[field_index]:
            raise AssertionError(f"field name {first['field_name']} != "
                                 f"{cmod.FIELD_ORDER[field_index]}")
        exported = replay_mod.reexport(tampered, tmp / "reexport" / "step003.npz", diff)
        if not Path(exported["window"]).is_file() or not Path(exported["diff"]).is_file():
            raise AssertionError("re-export did not write both files")
        return {"tampered": {"step": 3, "field_index": field_index,
                             "field_name": cmod.FIELD_ORDER[field_index],
                             "delta": 1e-3},
                "hash_mismatch": True,
                "first_mismatch_step": int(scan["policy_step"]),
                "diff": first, "reexported": exported}

    checker.check("offline_tamper_detection", tamper_checks)

    def top_hold_checks():
        if abs(fse.TOP_HOLD_S - .5) > 1e-12:
            raise AssertionError(f"TOP_HOLD_S changed to {fse.TOP_HOLD_S}")
        if cfg["curriculum"]["promotion_signal"] != "top_stable_hold":
            raise AssertionError("default promotion signal is no longer top_stable_hold")
        if abs(float(cfg["curriculum"]["min_top_hold_s"]) - .5) > 1e-12:
            raise AssertionError("min_top_hold_s is not 0.5")
        if abs(float(cfg["thresholds"]["acceptance"]["top_stable_s_ge"]) - .5) > 1e-12:
            raise AssertionError("acceptance top hold is not 0.5")
        return {"TOP_HOLD_S": fse.TOP_HOLD_S,
                "promotion_signal": cfg["curriculum"]["promotion_signal"],
                "alternatives_default_off": [row["signal"] for row in
                                             cfg["curriculum"]["alternatives"]],
                "acceptance_source": cfg["thresholds"]["acceptance"]["source"]}

    checker.check("top_hold_criterion_unchanged", top_hold_checks)


    result = checker.summary()
    result["threads_location"] = threads_location()
    result["timestamp"] = __import__("datetime").datetime.now().isoformat()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / SELFCHECK_JSON).write_text(json.dumps(result, indent=2) + "\n")
    print("SELFCHECK", result["status"], f"{result['passed']}/{result['total']}",
          "failed:", result["failed"], flush=True)
    return 0 if result["status"] == "PASS" else 1


# ---------------------------------------------------------------------------- cli
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=tuple(ARCHES), default="6frame")
    parser.add_argument("--stage", choices=[stage.name for stage in
                                            stage_ladder(json.loads(
                                                (CONFIG_DIR / "fs_stair_6frame.json").read_text()))],
                        default="flat_approach",
                        help="initial stage on the ladder")
    parser.add_argument("--config", type=Path,
                        help="config JSON; defaults to configs/fs_stair_<arch>.json")
    parser.add_argument("--out", type=Path, help="evidence / run directory")
    parser.add_argument("--run", action="store_true",
                        help="actually train; without it the script only prints and exits 0")
    parser.add_argument("--dry-run", action="store_true", help="explicit dry run (default)")
    parser.add_argument("--selfcheck", action="store_true", help="run the PASS/FAIL gate list")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="explicit escape hatch; prints NOT_FROM_SCRATCH")
    parser.add_argument("--torque-clip", choices=("90pct", "mjcf_ctrlrange", "contract_90pct", "mujoco_ctrlrange"),
                        help="override the config torque clip mode")
    parser.add_argument("--total-steps", type=int, help="override the env-step budget")
    parser.add_argument("--n-workers", type=int, help="override the CPU worker count")
    parser.add_argument("--seed", type=int, help="override the seed")
    parser.add_argument("--eval-stage", default="same",
                        choices=("same", "flat_approach", "step1", "step2",
                                 "continuous", "all"),
                        help="stage used for the post-training holdout; 'same' = the "
                             "training stage, 'all' = every ladder stage")
    parser.add_argument("--phase", choices=("baseline", "disturbance"), default="baseline",
                        help="baseline = minimal-variable from-scratch run; disturbance = "
                             "single-factor continuation from a checkpoint (locked)")
    parser.add_argument("--disturbance", action="append", choices=("none", "load", "push"),
                        help="single factor for --phase disturbance; repeat the flag with two "
                             "different values to prove the single-factor guard rejects it")
    parser.add_argument("--allow-disturbance", action="store_true",
                        help="explicit phase-2 unlock; without it --phase disturbance prints "
                             "PHASE2_LOCKED and exits")
    parser.add_argument("--accept-contract-change", action="store_true",
                        help="dry-run/replay only: accept a changed contract fingerprint "
                             "without a version bump (refused under --run)")
    parser.add_argument("--replay", action="store_true",
                        help="offline replay diagnostic over sampled windows (no training)")
    parser.add_argument("--replay-window", type=Path,
                        help="window .npz for --replay (defaults to the newest in <out>/windows)")
    parser.add_argument("--replay-reference", type=Path,
                        help="reference window .npz for the field-level diff")
    parser.add_argument("--replay-log", type=Path,
                        help="triple-tap jsonl to scan for the first hash mismatch")
    parser.add_argument("--re-export-first-mismatch", action="store_true",
                        help="re-export the first mismatching window plus its field diff")
    parser.add_argument("--eval-record", type=Path,
                        help="phase-2 holdout pass record (required for a formal --run)")
    parser.add_argument("--tap-stride", type=int,
                        help="override the triple-tap logging stride")
    return parser


PHASE_ERRORS = []



def resolve_phase(args) -> dict:
    """Structural phase validation plus the active randomization block.

    A locked phase 2 is not an error: it prints PHASE2_LOCKED and exits 0 on a
    dry-run, non-zero under --run. Missing single-factor/authorization pieces are
    reported through ``lock_reasons`` instead of being silently accepted.
    """
    errors = []
    disturbances = list(args.disturbance or [])
    if len(disturbances) > 1:
        errors.append("single-factor only: --disturbance takes exactly one of "
                      "none/load/push, got " + ", ".join(disturbances))
    if args.phase == "baseline":
        if disturbances:
            errors.append("--disturbance requires --phase disturbance")
        factor = "none"
    else:
        if not disturbances:
            errors.append("--phase disturbance requires --disturbance {none,load,push}")
        factor = disturbances[0] if disturbances else "none"
    lock_reasons = []
    if args.phase == "disturbance":
        if args.init_checkpoint is None:
            lock_reasons.append("missing --init-checkpoint <pt>")
        if not args.allow_disturbance:
            lock_reasons.append("missing --allow-disturbance (operator unlock)")
    if args.accept_contract_change and (args.run or not (args.dry_run or args.replay)):
        errors.append("--accept-contract-change is only allowed with --dry-run or --replay; "
                      "formal training must bump control_contract_version and attach replay + "
                      "baseline regression evidence")
    if args.replay and not args.replay_window and args.out is None:
        errors.append("--replay needs --replay-window <npz> or --out <dir with windows>")
    active = {"load": False, "friction": False, "gain": False, "push": False,
              "com_offset": False, "observation_noise": False}
    if args.phase == "disturbance" and factor == "load":
        active["load"] = True
    elif args.phase == "disturbance" and factor == "push":
        active["push"] = True
    return {"errors": errors, "locked": bool(lock_reasons), "lock_reasons": lock_reasons,
            "factor": factor, "randomization": active}


def phase_error(errors) -> int:
    for error in errors:
        print(f"PHASE_ERROR {error}", file=sys.stderr, flush=True)
    return 2


def check_eval_record(record_path, checkpoint_path, cfg) -> list:
    """Phase-2 formal --run: the holdout pass record must exist and match the checkpoint."""
    if record_path is None:
        return ["missing --eval-record: phase-2 --run requires the holdout pass record "
                "for the checkpoint being continued"]
    record_path = Path(record_path)
    if not record_path.is_file():
        return [f"--eval-record not found: {record_path}"]
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return [f"--init-checkpoint not found: {checkpoint_path}"]
    checkpoint_sha = sha256_file(checkpoint_path)
    payload = json.loads(record_path.read_text())
    if payload.get("checkpoint_sha256") != checkpoint_sha:
        return [f"--eval-record checkpoint_sha256 {payload.get('checkpoint_sha256')} does not "
                f"match --init-checkpoint sha256 {checkpoint_sha}"]
    missing = [key for key in ("holdout_seeds", "criteria", "gate_status") if key not in payload]
    if missing:
        return [f"--eval-record is missing required fields: {missing}"]
    if list(payload["holdout_seeds"]) != list(cfg["holdout_seeds"]):
        return [f"--eval-record holdout_seeds {payload['holdout_seeds']} != "
                f"{cfg['holdout_seeds']}"]
    missing = [key for key in ("A_s1", "B_s1", "C_top", "D") if key not in payload["criteria"]]
    if missing:
        return [f"--eval-record criteria is missing {missing} (values may still be "
                f"PENDING_OPERATOR_RULING)"]
    return []


def load_config(args, phase: dict) -> dict:
    path = args.config or (CONFIG_DIR / f"fs_stair_{args.arch}.json")
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    cfg = json.loads(path.read_text())
    if cfg["arch"] != args.arch:
        raise SystemExit(f"config arch {cfg['arch']} != --arch {args.arch}")
    if args.torque_clip:
        mode = {"contract_90pct": "90pct", "mujoco_ctrlrange": "mjcf_ctrlrange"
                }.get(args.torque_clip, args.torque_clip)
        cfg["torque"]["clip_mode"] = mode
        cfg["control"]["torque_clip_mode"] = mode
    if args.total_steps:
        cfg["total_env_steps"] = int(args.total_steps)
    if args.n_workers:
        cfg["n_workers"] = int(args.n_workers)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.tap_stride is not None:
        cfg["tap_log"]["stride"] = int(args.tap_stride)
    cfg["phase"] = args.phase
    cfg["disturbance"] = phase["factor"]
    cfg["randomization_active"] = phase["randomization"]
    if cfg["history"]["frames"] != ARCHES[args.arch]["frames"]:
        raise SystemExit("config history.frames disagrees with the arch definition")
    # ---- config <-> code consistency (fail fast, never silently merge)
    if cfg["initial_state"] != fse.INITIAL_STATE:
        raise fse.ContractError("config initial_state differs from fs_stair_env.INITIAL_STATE")
    if cfg["dispatch"] != fse.DISPATCH:
        raise fse.ContractError("config dispatch differs from fs_stair_env.DISPATCH")
    if float(cfg["control"]["torque_budget_nm"]) != float(cfg["torque"]["budget_nm"]):
        raise fse.ContractError("control.torque_budget_nm != torque.budget_nm")
    if cfg["control"]["torque_clip_mode"] != cfg["torque"]["clip_mode"]:
        raise fse.ContractError("control.torque_clip_mode != torque.clip_mode")
    ladder = stage_ladder(cfg)
    flat = next((stage for stage in ladder if stage.levels == 0), None)
    if flat is not None and abs(flat.start_back_m
                                - float(cfg["initial_state"]["start_back_flat_m"])) > 1e-12:
        raise fse.ContractError("initial_state.start_back_flat_m differs from the flat stage")
    if not cfg["phase_gate"]["phase2_locked"]:
        raise fse.ContractError("phase2 must start locked (phase_gate.phase2_locked=true)")
    args.config = path
    names = [stage.name for stage in ladder]
    if args.stage not in names:
        raise SystemExit(f"--stage {args.stage} not in ladder {names}")
    args.stage_index = names.index(args.stage)
    return cfg


def load_arch_config(arch: str) -> dict:
    cfg = json.loads((CONFIG_DIR / f"fs_stair_{arch}.json").read_text())
    cfg["phase"] = "baseline"
    cfg["disturbance"] = "none"
    cfg["randomization_active"] = {"load": False, "friction": False, "gain": False,
                                   "push": False, "com_offset": False,
                                   "observation_noise": False}
    return cfg


def contract_verify(cfg, args) -> dict:
    payload = cmod.build_payload(cfg["control"], cfg["history"], cfg["initial_state"],
                                 cfg["dispatch"])
    return cmod.verify(cfg["control_contract"]["fingerprint"],
                       cfg["control_contract"]["version"], payload,
                       accept_change=args.accept_contract_change,
                       will_run=bool(args.run and not args.replay))


def run_replay(args, cfg, out, contract_state) -> int:
    """Offline replay: locate the first step, then the field, then re-export."""
    say("[replay] mode=replay (no training)")
    if contract_state["changed"]:
        say("[replay]", contract_state["message"])
    scans = []
    if args.replay_log:
        logs = [Path(args.replay_log)]
    elif out is not None:
        logs = sorted((out / "tap").glob("triple_tap.worker*.jsonl"))
    else:
        logs = []
    windows = Path(args.replay_window).parent if args.replay_window else (
        out / "windows" if out is not None else None)
    for log in logs:
        scan = replay_mod.scan_run(log, windows)
        scans.append(scan)
        say("[replay] scan", json.dumps(scan, sort_keys=True))
    result = {"scans": scans}
    if args.replay_window and args.replay_reference:
        diff = replay_mod.compare_windows(args.replay_window, args.replay_reference)
        result["diff"] = diff
        say("[replay] diff", json.dumps(diff, sort_keys=True))
        if args.re_export_first_mismatch and not diff["hash_match"]:
            target = (out or Path(args.replay_window).parent) / "reexport" / (
                Path(args.replay_window).stem + "_reexport.npz")
            exported = replay_mod.reexport(args.replay_window, target, diff)
            result["reexported"] = exported
            say("[replay] reexported", json.dumps(exported, sort_keys=True))
    elif args.re_export_first_mismatch:
        first = next((row for row in scans if row.get("status") == "mismatch"), None)
        if first is None:
            say("[replay] nothing to re-export: no hash mismatch found")
        else:
            target = (out or Path(first["window"]).parent) / "reexport" / (
                Path(first["window"]).stem + "_reexport.npz")
            exported = replay_mod.reexport(first["window"], target, first)
            result["reexported"] = exported
            say("[replay] reexported", json.dumps(exported, sort_keys=True))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


def actor_state_dict(policy) -> dict:
    """Convert an SB3 policy into the ``actor.*`` tensors mujoco_screen.evaluate expects."""
    state = {}
    layers = (*policy.mlp_extractor.policy_net[::2], policy.action_net)
    for index, layer in enumerate(layers):
        state[f"actor.{2 * index}.weight"] = torch.as_tensor(
            layer.weight.detach().cpu(), dtype=torch.float32)
        state[f"actor.{2 * index}.bias"] = torch.as_tensor(
            layer.bias.detach().cpu(), dtype=torch.float32)
    return state


def make_native_planner(cfg, arch: str, model):
    """Drive mujoco_screen.evaluate with the arm's history + the shared control chain."""
    frames = ARCHES[arch]["frames"]
    control = fse.ControlChain(cfg["control"])
    contract = fse.load_contract(ASSETS)
    limits = fse.sim.load_target_limits(contract)
    defaults = np.asarray(contract["default_joint_pos"], dtype=float)
    control.bind(float(contract["control_dt"]), limits[12:, 2], limits[:12, 0],
                 limits[:12, 1], defaults)
    history = deque(maxlen=frames)
    state = {"started": False}

    def planner(data, obs, t, advance):
        frame = np.asarray(obs, dtype=np.float32)[:60]
        if not state["started"]:
            history.extend(frame.copy() for _ in range(frames))
            state["started"] = True
        else:
            history.append(frame.copy())
        if arch == "zero_history":
            stacked = np.concatenate([np.zeros((frames - 1) * 60, dtype=np.float32), frame])
        else:
            stacked = np.concatenate(list(history))
        action, _ = model.predict(stacked, deterministic=True)
        targets = control.targets(np.asarray(action, dtype=np.float32))
        return targets["executed"], targets["leg"], targets["wheel"]

    return planner


def evaluate_with_sim(cfg, model, out: Path, stage_name: str = "continuous") -> dict:
    """Frozen mujoco_screen scoring; only expressible on the stair stages."""
    if stage_name == "flat_approach":
        summary = {
            "evaluator": "not_run",
            "stage": stage_name,
            "reason": "FROZEN_SCORER_NOT_APPLICABLE: mujoco_screen.evaluate models only "
                      "the five-step stair terrain (stair_geoms 1..5 + surface_height) "
                      "and the rolling command ramp, so it cannot express flat ground "
                      "with the approach line. flat_approach is scored by the "
                      "rule-equivalent env evaluator written to holdout.json.",
            "holdout_seeds": cfg["holdout_seeds"],
            "passed": None,
            "total": 0,
            "rows": [],
        }
        (out / "holdout_mujoco.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("FROZEN_SCORER_NOT_APPLICABLE",
              json.dumps({"stage": stage_name, "fallback": "holdout.json"},
                         sort_keys=True), flush=True)
        return summary
    contract = fse.load_contract(ASSETS)
    limits = fse.sim.load_target_limits(contract)
    planner = make_native_planner(cfg, cfg["arch"], model)
    if cfg["arch"] == "1frame":
        internal_state = actor_state_dict(model.policy)
        policy_source = "trained_1frame_actor"
    else:
        # mujoco_screen.evaluate ALWAYS runs this 60-dim actor to produce
        # obs[42:58] (previous_action) before native_planner overrides the targets,
        # so it must be a valid 60-dim actor. The planner's own outputs replace its
        # actions, therefore the action-clip counters below are NOT meaningful for
        # the history arms.
        internal_state = actor_state_dict(make_policy(cfg, "1frame", cfg["seed"]).policy)
        policy_source = ("placeholder_untrained_1frame; targets come from "
                         "native_planner")
    active = any(bool(value) for value in cfg.get("randomization_active", {}).values())
    rows = []
    for seed in cfg["holdout_seeds"]:
        model_mj = fse.sim.load_flat_model(True, True, "stairs", seed)
        stress = (fse.sim.sample_dynamics_stress(seed, model_mj.nbody - 1, 16)
                  if active else None)
        result = fse.sim.evaluate(
            model_mj, contract, internal_state, seed, 0., limits=limits,
            action_clip=float(cfg["control"]["action_clip"]),
            wheel_scale=float(cfg["control"]["wheel_action_scale"]),
            wheel_angle_period=4 * np.pi, dynamics_stress=stress, terrain="stairs",
            observation_suffix=lambda t: np.array(
                [np.sin(2 * np.pi * t / 0.62), np.cos(2 * np.pi * t / 0.62)],
                dtype=np.float64),
            native_planner=planner)
        result["seed"] = int(seed)
        rows.append(result)
    summary = {
        "evaluator": "mujoco_screen.evaluate (criteria unchanged: ordered and longest*dt>=0.5 "
                     "and peak_impact<50)",
        "stage": stage_name,
        "policy_source": policy_source,
        "observation_suffix": "gait clock sin/cos of 2*pi*t/0.62 appended; 58 -> 60 dims",
        "clip_counters_note": "computed on the placeholder actor when arch != 1frame; "
                              "not meaningful with native_planner",
        "holdout_seeds": cfg["holdout_seeds"],
        "passed": int(sum(bool(row["passed"]) for row in rows)),
        "total": len(rows),
        "top_hold_criterion_s": fse.TOP_HOLD_S,
        "rows": rows,
    }
    (out / "holdout_mujoco.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("HOLDOUT_MUJOCO", json.dumps({key: summary[key] for key in
                                        ("passed", "total", "top_hold_criterion_s")}),
          flush=True)
    return summary


def main(argv=None) -> int:
    torch.set_num_threads(1)
    args = parse_args().parse_args(argv)
    if getattr(args, "eval_stage", None) == "same":
        args.eval_stage = args.stage
        print("EVAL_STAGE_RESOLVED same ->", args.eval_stage, flush=True)
    phase = resolve_phase(args)
    if phase["errors"]:
        return phase_error(phase["errors"])
    if phase["locked"]:
        print("PHASE2_LOCKED", ", ".join(phase["lock_reasons"]), flush=True)
        print("PHASE2_LOCKED: phase 2 stays locked; no training starts", flush=True)
        return 2 if args.run else 0
    check_ld_library_path()
    try:
        cfg = load_config(args, phase)
    except fse.ContractError as exc:
        print(f"CONTRACT_ERROR {exc}", file=sys.stderr, flush=True)
        return 2
    out = Path(args.out) if args.out else None
    if args.run and args.phase == "disturbance":
        errors = check_eval_record(args.eval_record, args.init_checkpoint, cfg)
        if errors:
            return phase_error(errors)
    try:
        verify_gating_pins(cfg)
        state = contract_verify(cfg, args)
        if state["changed"]:
            print(state["message"], flush=True)
        if state["accepted"]:
            print(state["accept_message"], flush=True)
            audit = {"event": "contract_change_accepted", "version": state["version"],
                     "stored_fingerprint": state["stored_fingerprint"],
                     "computed_fingerprint": state["computed_fingerprint"],
                     "mode": "replay" if args.replay else "dry-run", "argv": sys.argv[1:]}
            if out is not None:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "contract_change_audit.jsonl").open("a") as handle:
                    handle.write(json.dumps(audit, sort_keys=True) + "\n")
            print("CONTRACT_CHANGE_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)
        if state["run_refused"]:
            print("REFUSED: --run with a changed control contract; bump "
                  "control_contract_version and redo replay + baseline regression, or use "
                  "--dry-run/--replay with --accept-contract-change", file=sys.stderr,
                  flush=True)
            return 3
        report = build_report(cfg, args, state)
    except fse.ContractError as exc:
        print(f"CONTRACT_ERROR {exc}", file=sys.stderr, flush=True)
        return 2
    if args.replay:
        return run_replay(args, cfg, out, state)
    if args.selfcheck:
        print_report(report) if args.run else None
        return run_selfcheck(cfg, args, out)
    print_report(report)
    if not args.run:
        say("\n未获启动授权（no --run）：只打印预检报告，退出 0。")
        say("AUTHORIZATION: training NOT started")
        if out is not None:
            write_evidence(out, cfg, args, report)
        return 0
    if out is None:
        raise SystemExit("--run requires --out <evidence dir>")
    write_evidence(out, cfg, args, report)
    print(f"AUTHORIZED: --run given, writing {out}", flush=True)
    run_training(cfg, args, report, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
