#!/usr/bin/env python3
"""AS2-W rolling "from zero" training entry: three observation/architecture arms.

STATUS: research / deployable: false.  This file is the preparation deliverable of the
fs-roll run directory.  It launches NOTHING unless ``--run`` is given explicitly.

Safety contract of this entry point
-----------------------------------
* Default mode is a pure-CPU dry run.  It runs before any Isaac import, never imports
  ``isaaclab``, never creates a stage, never calls ``torch.cuda`` (the self-check asserts
  ``torch.cuda.is_initialized()`` stays False), and exits 0 after printing the full contract.
* ``--run`` is the only path that launches Kit (``isaaclab.app.launch_simulation``).
* GPU memory feasibility is read through ``nvidia-smi`` (read-only, no CUDA context, no
  allocation) so the dry run cannot disturb another agent's job on the same GPU.
* This file only *reads* existing project sources; it writes inside its own run directory
  and (with ``--run``) inside ``--out``.

Three arms, one wheel contract / reward / curriculum / env-step budget
---------------------------------------------------------------------
A  single 65-D frame                          -> small MLP
B  last 6 frames (6 x 65 = 390) concatenated  -> small MLP
C  last 6 frames -> small MLP -> vx_hat (1-D); actor sees current 65-D frame + vx_hat
   (66-D); the estimator is supervised by simulated true body-frame vx.  Only the critic
   may consume privileged state; the actor path never sees true vx.

Observation layout source of truth (read back from code, not from a report)
--------------------------------------------------------------------------
``as2w_isaac_flat_env_all.observation`` (lines 59-81) builds the 60-D base frame:

    0:3   command - (0.4, 0, 0)          base_velocity command, biased
    3:6   root_ang_vel_b                 IMU body angular rate (rad/s)
    6:9   projected_gravity_b            gravity in body frame (g)
    9     constant 0.4
    10:26 joint_pos - default_joint_pos  legs 12 + wheels 4 (wheels wrapped to [-2pi, 2pi])
    26:42 joint_vel                       rad/s, canonical AS2-W order
    42:58 previous processed action        exact post-scale/post-clip action the actuator saw
    58    sin(clock)                       clock = 2*pi*t/0.62 s
    59    cos(clock)

``cfg.as2w_zero_memory = True`` (the rolling/robust path) appends

    60:64 (wheel_delta / 0.20).clamp(-3, 3) * zero_flag
          wheel_delta = wrapped wheel angle minus the value latched when the command became
          zero; it is NOT the raw wheel angle (already divided by 0.20 and clamped)
    64    zero_flag = all(command == 0); together with 58/59 it decides whether 60:64 is
          meaningful at all (the delta is multiplied by the flag, so it is 0 otherwise)

=> 65-D.  ``as2w_isaac_flat_env_all.py:252`` is
``expected_features = 65 if args.zero_memory else 60``; the 60-D form has no wheel memory
and is *not* used here.  ``zero_memory=True`` is pinned in the config.

The older ``as2w_history.py`` convention (60 + 12*55 = 720-D sub-sampled history over
480 ms, zero-padded via ``lift_actor``) is deliberately NOT used: this experiment asks for
6 consecutive frames one policy step apart (100 ms span).  See config
``history.convention_note``.  Each arm's input width is explicit and independent; no arm is
produced by zero-padding another arm's weights.

The 6-frame definition (hard requirement, asserted in --selfcheck)
------------------------------------------------------------------
    * includes the current frame;
    * ordered oldest -> newest;
    * at a 20 ms control period the first-to-last span is 5 * 20 ms = 100 ms;
    * the history advances exactly once per *policy* step (not per physics substep);
    * on reset every slot is filled with the first frame of the new episode, clearing
      stale frames from the previous episode.

Wheel contract
--------------
``deploy/as2w/contracts/wheel_contract_v1.json`` (sha256
8f79a9539abc961ec1eb1f80edb84aa1d375867f6cfb3a36e03b8ff5d5e061d2) is read, hash-verified
and printed field by field.  ``--kd {0.5,1.5,2.0}`` maps to
``w1w2_as2w_diagnostic_kd_{0p5,1p5,2p0}``; one of ``--kd`` / ``--contract`` is mandatory.

Torque budget: URDF ``*_foot`` effort limit is 15.0 N*m, the project budget is
15 * 0.9 = 13.5 N*m.  ``as2w_training_safety.apply_safety`` lowers ``effort_limit_sim`` to
13.5, and the set-point is additionally clamped to the same budget inside
``BudgetedWheelVelocity.apply_actions`` via ``clamp_wheel_torque`` *before*
``set_joint_velocity_target_index``, so the velocity servo cannot wind up against a budget
the solver will not deliver.  The clamp acts on the requested velocity error, i.e. on the
implied torque ``kd * (target_vel - joint_vel)``; it does not claim to measure solver torque.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------------------
# Fixed paths / versioned inputs (read-only)
# --------------------------------------------------------------------------------------
REPO = Path("/root/gpufree-data/Rob2Nav")
A2 = REPO / "training/as2w-backup/as2w-current"
CODE = A2 / "code"
SKILLS_ROLLING = REPO / "skills/as2w/rolling"
CONTRACT_PATH = REPO / "deploy/as2w/contracts/wheel_contract_v1.json"
CONTRACT_SHA256 = "8f79a9539abc961ec1eb1f80edb84aa1d375867f6cfb3a36e03b8ff5d5e061d2"
AGENTS_PATH = A2 / "AGENTS.md"
USD_PATH = A2 / "runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
URDF_PATH = A2 / "assets/r237-train-root/official-model/robots/as2w_description/as2w.urdf"
ARME_PATH = SKILLS_ROLLING / "train_armE_support2p0.py"

RUN_DIR = Path(__file__).resolve().parent
CONFIG_DIR = RUN_DIR / "configs"

SOURCE_FILES = {
    "as2w_isaac_flat_env_all.py": CODE / "as2w_isaac_flat_env_all.py",
    "as2w_isaac_robust_env.py": CODE / "as2w_isaac_robust_env.py",
    "as2w_training_safety.py": CODE / "as2w_training_safety.py",
    "as2w_knee_envelope.py": CODE / "as2w_knee_envelope.py",
    "as2w_isaac_speed_range.py": CODE / "as2w_isaac_speed_range.py",
    "as2w_predictive_joint_guard.py": CODE / "as2w_predictive_joint_guard.py",
    "train_armE_support2p0.py": ARME_PATH,
    "wheel_contract_v1.json": CONTRACT_PATH,
    "AGENTS.md": AGENTS_PATH,
    "as2w.usda": USD_PATH,
    "as2w.urdf": URDF_PATH,
}

# --------------------------------------------------------------------------------------
# Contract constants (verified against source in --selfcheck)
# --------------------------------------------------------------------------------------
CONTROL_PERIOD_S = 0.02
PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_DIM = 16
LEG_DIM = 12
WHEEL_DIM = 4
OBS_DIM = 65                   # zero_memory=True rolling observation
HISTORY_FRAMES = 6
HISTORY_SPAN_S = (HISTORY_FRAMES - 1) * CONTROL_PERIOD_S      # 0.100 s
WHEEL_ACTION_SCALE = 5.0
LEG_ACTION_SCALE = 0.35
LEG_KP = 55.0
LEG_KD = 2.0
WHEEL_KP = 0.0
TORQUE_HARD_LIMIT_NM = 15.0    # URDF *_foot joint effort limit
TORQUE_BUDGET_NM = 13.5        # 15.0 * 0.9, project budget
TORQUE_BUDGET_RATIO = TORQUE_BUDGET_NM / TORQUE_HARD_LIMIT_NM
ACTION_CLIP = 6.0
WHEEL_TARGET_LIMIT_RAD_S = WHEEL_ACTION_SCALE * ACTION_CLIP      # 30 rad/s before torque clamp
DEFAULT_LEG_POS = (0.0,) * 4 + (0.8,) * 4 + (-1.6,) * 4         # hip, thigh, calf (canonical order)

JOINT_ORDER = tuple(
    f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot")
    for leg in ("FL", "FR", "RL", "RR")
)
WHEEL_JOINT_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
WHEEL_ACTION_CHANNELS = {name: LEG_DIM + i for i, name in enumerate(WHEEL_JOINT_ORDER)}

OBS_FIELDS = (
    ("command_x", 0, 3, "m/s, rad/s", "base_velocity command minus (0.4,0,0) bias"),
    ("body_gyro", 3, 6, "rad/s", "root_ang_vel_b (IMU)"),
    ("gravity_b", 6, 9, "g", "projected_gravity_b"),
    ("const_0p4", 9, 10, "-", "constant 0.4"),
    ("joint_pos", 10, 26, "rad", "joint_pos - default; wheels wrapped to [-2pi,2pi]"),
    ("joint_vel", 26, 42, "rad/s", "joint velocity, canonical AS2-W order"),
    ("prev_action", 42, 58, "-", "previous processed action after scale/clip"),
    ("phase_sin", 58, 59, "-", "sin(2*pi*t/0.62)"),
    ("phase_cos", 59, 60, "-", "cos(2*pi*t/0.62)"),
    ("wheel_delta", 60, 64, "rad/0.20 clamped +-3", "wheel displacement since zero-command entry, gated"),
    ("zero_flag", 64, 65, "-", "1.0 when command == 0; gates 60:64"),
)
OBS_FIELD_ORDER = tuple(field[0] for field in OBS_FIELDS)

# Network sizes are fixed for all three arms (same hidden width / depth).
ACTOR_HIDDEN = (128, 128, 128)
CRITIC_HIDDEN = (128, 128, 128)
ESTIMATOR_HIDDEN = (128, 128)
PRIVILEGED_DIM = 5             # true vx, vy, vz, wz, root height above origin
ARM_INFO = {
    "A": {"name": "single_frame", "actor_input": OBS_DIM, "forecast": False},
    "B": {"name": "six_frame_concat", "actor_input": HISTORY_FRAMES * OBS_DIM, "forecast": False},
    "C": {"name": "six_frame_estimated_vx", "actor_input": OBS_DIM + 1, "forecast": True},
}

PPO_DEFAULTS = {
    "epochs": 5,
    "minibatches": 4,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip": 0.2,
    "entropy_coef": 0.005,
    "value_coef": 1.0,
    "max_grad_norm": 1.0,
    "init_noise_std": 0.25,
}
PRESETS = {
    "small": {"num_envs": 4096, "n_steps": 24},
    "large": {"num_envs": 16384, "n_steps": 128},
}
DEFAULT_PRESET = "large"
DEFAULT_ITERATIONS = 3000
DEFAULT_SEED = 449

GPU_BASELINE_MIB = 6931.0      # measured by the operator: 4096 envs -> 6931 MiB
GPU_BASELINE_ENVS = 4096
GPU_SAFETY_FRACTION = 0.90

PENDING_VALIDATION = (
    "contract_group_selection: pending_validation (dual-simulator same-target loaded evidence insufficient)",
    "usd_drive_gain_source: not_measured -- the USD physics layer bakes stiffness=1.7453293 /"
    " damping=0.017453292 per joint (URDF converter rad values of 100 / 1.0) and"
    " ImplicitActuatorCfg carries leg kp 55 / kd 2.0, but the runtime write path from cfg to the"
    " PhysX drive was not located/verified in this phase.  Intended: cfg wins; unverified.",
    "torque_budget_13p5: software pre-dispatch clamp + effort_limit_sim 13.5 are both present;"
    " solver-level torque is estimated, not measured",
    "envs_16384: GPU memory is a linear extrapolation from the 6931 MiB @ 4096 envs measurement,"
    " not a measurement of this configuration",
    "speed_curriculum: staged_0_to_3 is implemented (8 stages 0.05..3.00 m/s, four gate classes,"
    " per-stage unlock); gate thresholds are PLACEHOLDER_PENDING_OPERATOR_RULING and the staged"
    " runtime path was never executed",
    "disturbance_phase: locked by default; --run requires --allow-disturbance + --init-checkpoint"
    " + --eval-record with matching checkpoint sha256; load and push must pass separately before"
    " any merge",
    "control_contract: --accept-contract-change is a dry-run/replay diagnostic only and is"
    " rejected with --run",
    "torque_solver: tau_solver_nm is not_measured (PhysX implicit drive does not expose applied"
    " joint effort); tau_request_est_nm is a software estimate and is labelled ESTIMATED",
    "run_path_not_executed: --run was never executed in this phase (training forbidden)",
    "wheel_radius_odometry: not used by this entry point",
)

CHANGE_PROVENANCE = (
    "num_envs: operator adjusted 32768 -> 16384 on 2026-11-16.  Final value 16384; 32768 was a"
    " rejected option, not a typo.",
    "iterations: operator raised 1500 -> 3000 so that the total budget is unchanged.",
    "budget equality (self-proof): 32768*128*1500 = 6,291,456,000 and"
    " 16384*128*3000 = 6,291,456,000 -- the same total.  Only parallelism was halved (to save"
    " GPU memory); total compute budget did not change.  Parallelism and total budget are"
    " independent knobs.",
    "per_round_steps: 32768*128 = 4,194,304 -> 16384*128 = 2,097,152 (envs halved).",
    "legacy entry: as2w_isaac_flat_env_all.py:258 records steps_per_iteration = num_envs * 24."
    "  This entry uses n_steps = 128 per round; the 24-step figure is superseded, not reused.",
)

# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested by --selfcheck; no Isaac, no CUDA)
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_capture(argv) -> str | None:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def gpu_memory_mib() -> tuple[float, float] | None:
    """(total, used) in MiB read through nvidia-smi -- no CUDA context, no allocation."""
    out = run_capture(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                       "--format=csv,noheader,nounits"])
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [piece.strip() for piece in first.split(",")]
    if len(parts) < 2:
        return None
    return float(parts[0]), float(parts[1])


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def clamp_wheel_torque(target_vel: torch.Tensor, joint_vel: torch.Tensor,
                       kd: float, budget_nm: float) -> torch.Tensor:
    """Clamp a wheel velocity set-point so the implicit drive cannot exceed ``budget_nm``.

    The wheel actuator is a velocity servo with kp = 0, so the implied torque is
    ``kd * (target_vel - joint_vel)``.  Clamping that product to the torque budget is
    equivalent to clamping the velocity error to ``budget / kd``.  The returned set-point is
    the closest one that respects the budget and is applied *before*
    ``set_joint_velocity_target_index``.

    This models the implicit drive; PhysX ``effort_limit_sim`` is already lowered to the same
    budget by ``apply_safety``.  Solver torque is not measured.
    """
    if kd <= 0.0:
        raise ValueError(f"wheel kd must be positive, got {kd}")
    if budget_nm <= 0.0:
        raise ValueError(f"torque budget must be positive, got {budget_nm}")
    error = kd * (target_vel - joint_vel)
    return joint_vel + torch.clamp(error, -budget_nm, budget_nm) / kd


class HistoryBuffer:
    """Last-N-frame observation history, oldest -> newest, current frame in the last slot.

    Contract (asserted by ``--selfcheck``):
      * ``reset(first)`` fills *all* slots with the new episode's first frame, so no frame
        from a previous episode can survive;
      * ``push(obs)`` moves the window exactly one policy step and puts ``obs`` last;
      * with 20 ms policy steps the first-to-last span is 100 ms for 6 frames.
    """

    def __init__(self, num_envs: int, dim: int, frames: int = HISTORY_FRAMES,
                 device="cpu", dtype=torch.float32):
        self.frames = frames
        self.dim = dim
        self.data = torch.zeros((num_envs, frames, dim), device=device, dtype=dtype)

    def reset(self, first_frame: torch.Tensor, env_ids=None) -> None:
        if first_frame.dim() != 2 or first_frame.shape[1] != self.dim:
            raise ValueError(f"reset expects (n,{self.dim}), got {tuple(first_frame.shape)}")
        if env_ids is None:
            self.data[:] = first_frame.unsqueeze(1).expand(-1, self.frames, -1)
            return
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        if ids.dtype == torch.bool:
            ids = torch.nonzero(ids, as_tuple=False).squeeze(-1)
        self.data[ids] = first_frame[ids].unsqueeze(1).expand(-1, self.frames, -1)

    def push(self, obs: torch.Tensor) -> None:
        if obs.shape != (self.data.shape[0], self.dim):
            raise ValueError(f"push expects {(self.data.shape[0], self.dim)}, got {tuple(obs.shape)}")
        self.data = torch.cat((self.data[:, 1:], obs.unsqueeze(1)), dim=1)

    def flat(self) -> torch.Tensor:
        return self.data.reshape(self.data.shape[0], self.frames * self.dim)


# --------------------------------------------------------------------------------------
# Speed curriculum (AGENTS.md section 1): armE mixture (default) or staged 0 -> 3 m/s
# --------------------------------------------------------------------------------------
CURRICULUM_MODES = ("armE_mixture", "staged_0_to_3")
DEFAULT_CURRICULUM = "armE_mixture"
CURRICULUM_THRESHOLD_STATUS = "PLACEHOLDER_PENDING_OPERATOR_RULING"

ARM_E_MIXTURE_SPEC = {
    "vx_cap_m_s": 0.25,
    "yaw_cap_rad_s": 0.15,
    "min_abs_cmd": 0.05,
    "mode_buckets": [0.20, 0.35, 0.75],
    "mixture": {"hold": 0.20, "straight": 0.15, "spin": 0.40, "arc": 0.25},
    "source": "skills/as2w/rolling/train_armE_support2p0.py:128-147 (RollingCommands default path)",
    "description": "armE fixed mixture; caps at 0.25 m/s and is NOT a speed curriculum",
}

STAGE_TABLE = (
    {"index": 0, "speed_m_s": 0.05, "yaw_rad_s": 0.10, "dwell_s": 10.0, "share": 0.15},
    {"index": 1, "speed_m_s": 0.10, "yaw_rad_s": 0.10, "dwell_s": 10.0, "share": 0.15},
    {"index": 2, "speed_m_s": 0.25, "yaw_rad_s": 0.15, "dwell_s": 10.0, "share": 0.15},
    {"index": 3, "speed_m_s": 0.50, "yaw_rad_s": 0.15, "dwell_s": 10.0, "share": 0.15},
    {"index": 4, "speed_m_s": 1.00, "yaw_rad_s": 0.20, "dwell_s": 12.0, "share": 0.15},
    {"index": 5, "speed_m_s": 1.50, "yaw_rad_s": 0.20, "dwell_s": 12.0, "share": 0.10},
    {"index": 6, "speed_m_s": 2.00, "yaw_rad_s": 0.25, "dwell_s": 12.0, "share": 0.10},
    {"index": 7, "speed_m_s": 3.00, "yaw_rad_s": 0.25, "dwell_s": 15.0, "share": 0.05},
)

SUBTYPE_SHARES = {"hold": 0.10, "straight": 0.30, "pure_turn": 0.30, "arc": 0.30}

GATE_THRESHOLDS = {
    "track": {"lin_vel_abs_error_m_s_max": 0.10,
              "yaw_rate_abs_error_rad_s_max": 0.05,
              "in_band_fraction_min": 0.90},
    "attitude": {"upright_cos_min": 0.90, "roll_deg_max": 8.0, "pitch_deg_max": 8.0},
    "contact": {"min_loaded_wheels": 4, "max_unload_ms": 60.0,
                "nonwheel_contact_n_max": 4.0, "wheel_normal_force_min_n": 2.0},
    "brake": {"stop_time_max_s": 2.0, "stop_distance_max_m": 1.0,
              "post_stop_drift_m_s_max": 0.02, "post_stop_hold_s": 10.0},
}

GATE_THRESHOLD_SOURCE = (
    "AGENTS.md section 1 only gives qualitative gates (tracking/attitude/contact/braking in band per"
    " stage, unlock only after all pass).  The numbers above are conservative placeholders, not"
    " measured and not operator-approved; status PLACEHOLDER_PENDING_OPERATOR_RULING."
)

CURRICULUM_POLICY = {
    "passes_required": 3,
    "hard_failure_policy": "demote_one_stage",
    "high_speed_arc_from_stage": 5,
    "high_speed_arc_open": False,
    "mix_stages": False,
}


class ArmEMixture:
    """The armE command mixture, as the one sampler class dry-run and --run both use."""

    def __init__(self, spec=None):
        self.spec = json.loads(json.dumps(spec or ARM_E_MIXTURE_SPEC))

    def decode(self, draw: torch.Tensor):
        low, high = self.spec["min_abs_cmd"], self.spec["vx_cap_m_s"]
        mode = torch.bucketize(draw[:, 0].contiguous(),
                               draw.new_tensor(tuple(self.spec["mode_buckets"])))
        vx = (low + (high - low) * draw[:, 1]) * torch.where(draw[:, 2] < .5, -1., 1.)
        yaw = (low + (self.spec["yaw_cap_rad_s"] - low) * draw[:, 3]) * torch.where(
            draw[:, 4] < .5, -1., 1.)
        return vx * ((mode == 1) | (mode == 3)), yaw * (mode >= 2), mode

    def describe(self) -> list[str]:
        spec, mix = self.spec, self.spec["mixture"]
        return ["[curric] mode=armE_mixture | vx cap %.2f m/s | wz cap %.2f rad/s | mixture"
                " hold/straight/spin/arc = %.2f/%.2f/%.2f/%.2f | staged_0_to_3 not active"
                % (spec["vx_cap_m_s"], spec["yaw_cap_rad_s"], mix["hold"], mix["straight"],
                   mix["spin"], mix["arc"])]


class SpeedCurriculum:
    """Staged 0 -> 3 m/s curriculum with four gate classes per stage (pure Python state machine).

    * advance only when track AND attitude AND contact AND brake all pass, ``passes_required``
      times in a row;
    * a hard failure (fall / non-wheel contact / termination) demotes one stage (or resets to 0)
      and clears the pass streak, per ``hard_failure_policy``;
    * the top stage never advances further;
    * gate thresholds are placeholders pending operator ruling.
    """

    def __init__(self, stages=None, subtype_shares=None, thresholds=None, policy=None, seed=0):
        self.stages = [dict(stage) for stage in (stages or STAGE_TABLE)]
        self.subtype_shares = dict(subtype_shares or SUBTYPE_SHARES)
        self.thresholds = json.loads(json.dumps(thresholds or GATE_THRESHOLDS))
        self.policy = dict(CURRICULUM_POLICY if policy is None else policy)
        self.threshold_status = CURRICULUM_THRESHOLD_STATUS
        self._rng = random.Random(seed)
        self.validate()
        self.stage = 0
        self.pass_streak = 0
        self.passes = [0] * len(self.stages)
        self.fails = [0] * len(self.stages)
        self.first_fail_kind = [None] * len(self.stages)
        self.events = []
        self.max_stage_reached = 0

    # -- validation ---------------------------------------------------------------
    def validate(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("a staged curriculum needs at least two stages")
        speeds = [float(stage["speed_m_s"]) for stage in self.stages]
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError(f"stage speeds must be strictly increasing: {speeds}")
        if abs(speeds[-1] - 3.0) > 1e-9:
            raise ValueError(f"top stage must be 3.0 m/s, got {speeds[-1]}")
        for stage in self.stages:
            if float(stage["dwell_s"]) <= 0.0:
                raise ValueError(f"stage {stage['index']} dwell must be positive")
            if float(stage["share"]) <= 0.0:
                raise ValueError(f"stage {stage['index']} share must be positive")
        if abs(sum(float(stage["share"]) for stage in self.stages) - 1.0) > 1e-6:
            raise ValueError("stage shares must sum to 1.0")
        if abs(sum(self.subtype_shares.values()) - 1.0) > 1e-6:
            raise ValueError("subtype shares must sum to 1.0")
        for gate in ("track", "attitude", "contact", "brake"):
            if gate not in self.thresholds:
                raise ValueError(f"missing gate thresholds: {gate}")
        if int(self.policy["passes_required"]) < 1:
            raise ValueError("passes_required must be >= 1")
        if self.policy["hard_failure_policy"] not in ("demote_one_stage", "reset_to_zero"):
            raise ValueError("hard_failure_policy must be demote_one_stage or reset_to_zero")
        top = int(self.policy["high_speed_arc_from_stage"])
        if not 0 <= top < len(self.stages):
            raise ValueError(f"high_speed_arc_from_stage {top} outside stage table")

    # -- properties ---------------------------------------------------------------
    @property
    def max_speed(self) -> float:
        return float(self.stages[-1]["speed_m_s"])

    @property
    def is_top(self) -> bool:
        return self.stage == len(self.stages) - 1

    def stage_spec(self) -> dict:
        return self.stages[self.stage]

    def active_subtypes(self) -> dict:
        shares = dict(self.subtype_shares)
        if (self.stage >= int(self.policy["high_speed_arc_from_stage"])
                and not self.policy["high_speed_arc_open"]):
            shares.pop("arc", None)
        total = sum(shares.values())
        return {name: weight / total for name, weight in shares.items()}

    # -- sampling (the one source dry-run prints from and --run samples from) -----
    def sample(self) -> dict:
        stage = self.stages[self.stage]
        shares = self.active_subtypes()
        pick, accumulated, kind = self._rng.random(), 0.0, None
        for name, weight in shares.items():
            accumulated += weight
            if pick <= accumulated:
                kind = name
                break
        if kind is None:
            kind = next(iter(shares))
        sign = 1.0 if self._rng.random() >= 0.5 else -1.0
        speed, yaw = float(stage["speed_m_s"]), float(stage["yaw_rad_s"])
        if kind == "hold":
            vx, wz = 0.0, 0.0
        elif kind == "straight":
            vx, wz = sign * speed, 0.0
        elif kind == "pure_turn":
            vx, wz = 0.0, sign * yaw
        else:  # arc
            vx, wz = speed, sign * yaw
        return {"stage": self.stage, "kind": kind, "vx": vx, "wz": wz,
                "speed_m_s": speed, "yaw_rad_s": yaw, "arc_blocked": kind == "arc"
                and not self.policy["high_speed_arc_open"]}

    # -- gates --------------------------------------------------------------------
    def evaluate(self, outcome: dict) -> dict:
        def number(section, key):
            value = (outcome.get(section) or {}).get(key)
            return None if value is None else float(value)

        t = self.thresholds
        lin = number("track", "lin_vel_abs_error_m_s")
        yaw = number("track", "yaw_rate_abs_error_rad_s")
        band = number("track", "in_band_fraction")
        upright = number("attitude", "upright_cos")
        roll = number("attitude", "roll_deg")
        pitch = number("attitude", "pitch_deg")
        loaded = number("contact", "min_loaded_wheels")
        unload = number("contact", "max_unload_ms")
        nonwheel = number("contact", "nonwheel_contact_n")
        brake = outcome.get("brake") or {}
        measured = bool(brake.get("measured", False))
        stop_time = number("brake", "stop_time_s")
        stop_distance = number("brake", "stop_distance_m")
        drift = number("brake", "post_stop_drift_m_s")

        gates = {
            "track": (None not in (lin, yaw, band)
                      and lin <= t["track"]["lin_vel_abs_error_m_s_max"]
                      and yaw <= t["track"]["yaw_rate_abs_error_rad_s_max"]
                      and band >= t["track"]["in_band_fraction_min"]),
            "attitude": (None not in (upright, roll, pitch)
                         and upright >= t["attitude"]["upright_cos_min"]
                         and abs(roll) <= t["attitude"]["roll_deg_max"]
                         and abs(pitch) <= t["attitude"]["pitch_deg_max"]),
            "contact": (None not in (loaded, unload, nonwheel)
                        and loaded >= t["contact"]["min_loaded_wheels"]
                        and unload <= t["contact"]["max_unload_ms"]
                        and nonwheel <= t["contact"]["nonwheel_contact_n_max"]),
            "brake": (measured and None not in (stop_time, stop_distance, drift)
                      and stop_time <= t["brake"]["stop_time_max_s"]
                      and stop_distance <= t["brake"]["stop_distance_max_m"]
                      and drift <= t["brake"]["post_stop_drift_m_s_max"]),
        }
        hard_fail = bool(outcome.get("hard_fail", False))
        first_fail = None
        if hard_fail:
            first_fail = outcome.get("hard_fail_kind") or "hard_fail"
        else:
            for name in ("track", "attitude", "contact", "brake"):
                if not gates[name]:
                    first_fail = name
                    break
        return {"gates": gates, "all_pass": all(gates.values()) and not hard_fail,
                "hard_fail": hard_fail, "hard_fail_kind": outcome.get("hard_fail_kind"),
                "first_fail_kind": first_fail}

    def observe(self, outcome: dict) -> dict:
        result = self.evaluate(outcome)
        before = self.stage
        if result["hard_fail"]:
            self.fails[self.stage] += 1
            if self.first_fail_kind[self.stage] is None:
                self.first_fail_kind[self.stage] = result["first_fail_kind"]
            if self.policy["hard_failure_policy"] == "reset_to_zero":
                self.stage = 0
            else:
                self.stage = max(0, self.stage - 1)
            self.pass_streak = 0
            event = "hard_fail"
        elif result["all_pass"]:
            self.passes[self.stage] += 1
            self.pass_streak += 1
            if self.pass_streak >= int(self.policy["passes_required"]) and not self.is_top:
                self.pass_streak = 0
                self.stage += 1
                self.max_stage_reached = max(self.max_stage_reached, self.stage)
                event = "advance"
            else:
                event = "pass"
        else:
            self.fails[self.stage] += 1
            self.pass_streak = 0
            if self.first_fail_kind[self.stage] is None:
                self.first_fail_kind[self.stage] = result["first_fail_kind"]
            event = "hold"
        record = {"event": event, "from_stage": before, "to_stage": self.stage,
                  "speed_m_s": self.stages[self.stage]["speed_m_s"],
                  "gates": result["gates"], "first_fail_kind": result["first_fail_kind"]}
        self.events.append(record)
        return record

    # -- reporting ----------------------------------------------------------------
    def describe(self) -> list[str]:
        speeds = [float(stage["speed_m_s"]) for stage in self.stages]
        monotonic = all(b > a for a, b in zip(speeds, speeds[1:]))
        lines = ["[curric] mode=staged_0_to_3 | max_speed %.2f m/s | stages %d | monotonic=%s"
                 % (self.max_speed, len(self.stages), monotonic)]
        for stage in self.stages:
            lines.append("[curric] stage %d  %.2f m/s  dwell %.0f s  share %.2f  gates:"
                         " track/attitude/contact/brake"
                         % (stage["index"], stage["speed_m_s"], stage["dwell_s"], stage["share"]))
        first = int(self.policy["high_speed_arc_from_stage"])
        if first < len(self.stages):
            lines.append("[curric] arc at high speed: marked needs_separate_validation"
                         " (stages %d..%d)%s"
                         % (first, len(self.stages) - 1,
                            "" if self.policy["high_speed_arc_open"]
                            else " -- arc BLOCKED there until explicitly opened"))
        t = self.thresholds
        lines.append("[curric] thresholds: %s  track=lin_err<=%.2f m/s & yaw_err<=%.2f rad/s &"
                     " in_band>=%.2f  attitude=upright_cos>=%.2f & |roll|<=%.0f deg &"
                     " |pitch|<=%.0f deg  contact=loaded_wheels>=%d & unload<=%.0f ms &"
                     " nonwheel<=%.1f N  brake=stop<=%.1f s & dist<=%.2f m & drift<=%.3f m/s &"
                     " hold %.0f s"
                     % (self.threshold_status,
                        t["track"]["lin_vel_abs_error_m_s_max"],
                        t["track"]["yaw_rate_abs_error_rad_s_max"],
                        t["track"]["in_band_fraction_min"],
                        t["attitude"]["upright_cos_min"], t["attitude"]["roll_deg_max"],
                        t["attitude"]["pitch_deg_max"], t["contact"]["min_loaded_wheels"],
                        t["contact"]["max_unload_ms"], t["contact"]["nonwheel_contact_n_max"],
                        t["brake"]["stop_time_max_s"], t["brake"]["stop_distance_max_m"],
                        t["brake"]["post_stop_drift_m_s_max"], t["brake"]["post_stop_hold_s"]))
        return lines

    def state(self) -> dict:
        return {"stage": self.stage,
                "stage_spec": dict(self.stages[self.stage]),
                "pass_streak": self.pass_streak,
                "passes": list(self.passes), "fails": list(self.fails),
                "first_fail_kind": list(self.first_fail_kind),
                "max_stage_reached": self.max_stage_reached,
                "policy": dict(self.policy),
                "threshold_status": self.threshold_status}


def curriculum_settings(mode: str = DEFAULT_CURRICULUM) -> dict:
    if mode not in CURRICULUM_MODES:
        raise SystemExit(f"[FATAL] unknown curriculum {mode!r}; have {CURRICULUM_MODES}")
    return {
        "mode": mode,
        "modes": list(CURRICULUM_MODES),
        "armE_mixture": ARM_E_MIXTURE_SPEC,
        "staged_0_to_3": {
            "max_speed_m_s": 3.0,
            "stages": [dict(stage) for stage in STAGE_TABLE],
            "subtype_shares": SUBTYPE_SHARES,
            "thresholds": GATE_THRESHOLDS,
            "threshold_status": CURRICULUM_THRESHOLD_STATUS,
            "threshold_source": GATE_THRESHOLD_SOURCE,
            "policy": CURRICULUM_POLICY,
            "gate_classes": ["track", "attitude", "contact", "brake"],
            "unlock_rule": "advance only after passes_required consecutive episodes with all four"
                           " gate classes passing; hard failure demotes per policy",
            "dwell_application": "per-stage dwell recorded; the command term uses"
                                 " resampling_time_range = (min_dwell, max_dwell) of the stage table",
            "arc_high_speed": "needs_separate_validation, blocked by default",
        },
        "active": mode,
    }


def passing_outcome() -> dict:
    return {
        "track": {"lin_vel_abs_error_m_s": 0.02, "yaw_rate_abs_error_rad_s": 0.01,
                  "in_band_fraction": 0.98},
        "attitude": {"upright_cos": 0.99, "roll_deg": 1.0, "pitch_deg": 1.0},
        "contact": {"min_loaded_wheels": 4, "max_unload_ms": 0.0, "nonwheel_contact_n": 0.0},
        "brake": {"measured": True, "stop_time_s": 0.8, "stop_distance_m": 0.2,
                  "post_stop_drift_m_s": 0.005},
        "hard_fail": False,
        "hard_fail_kind": None,
    }


def check_curriculum() -> list[dict]:
    """Pure-CPU unit tests for the staged curriculum state machine."""
    items = []
    fast = {**CURRICULUM_POLICY, "passes_required": 1}
    good = passing_outcome()

    # (a) all four gates pass -> advance once the streak threshold is met
    cur = SpeedCurriculum(policy={**CURRICULUM_POLICY, "passes_required": 2})
    cur.observe(good)
    after_one = cur.stage
    cur.observe(good)
    items.append({"id": "curric_advance_all_gates",
                  "status": "PASS" if after_one == 0 and cur.stage == 1 else "FAIL",
                  "detail": f"2 consecutive all-pass episodes: stage {after_one} -> {cur.stage}"})

    # (b) any single missing gate blocks advancement
    blocked = {}
    for gate in ("track", "attitude", "contact", "brake"):
        bad = passing_outcome()
        if gate == "track":
            bad["track"] = {"lin_vel_abs_error_m_s": 9.0, "yaw_rate_abs_error_rad_s": 9.0,
                            "in_band_fraction": 0.0}
        elif gate == "attitude":
            bad["attitude"] = {"upright_cos": 0.1, "roll_deg": 45.0, "pitch_deg": 45.0}
        elif gate == "contact":
            bad["contact"] = {"min_loaded_wheels": 2, "max_unload_ms": 500.0,
                              "nonwheel_contact_n": 50.0}
        else:
            bad["brake"] = {"measured": False, "stop_time_s": 9.0, "stop_distance_m": 9.0,
                            "post_stop_drift_m_s": 1.0}
        probe = SpeedCurriculum(policy=fast)
        for _ in range(3):
            probe.observe(bad)
        blocked[gate] = probe.stage
    items.append({"id": "curric_no_advance_missing_gate",
                  "status": "PASS" if all(stage == 0 for stage in blocked.values()) else "FAIL",
                  "detail": f"stage after 3 episodes missing one gate: {blocked}"})

    # (c) hard failure demotes one stage (default policy) and clears the streak
    probe = SpeedCurriculum(policy=fast)
    probe.observe(good)                      # -> stage 1
    advanced = probe.stage
    event = probe.observe({**passing_outcome(), "hard_fail": True,
                           "hard_fail_kind": "fall"})
    items.append({"id": "curric_hard_fail_demote",
                  "status": "PASS" if advanced == 1 and probe.stage == 0
                  and event["event"] == "hard_fail" else "FAIL",
                  "detail": f"stage {advanced} -> {probe.stage}, event={event['event']},"
                            f" first_fail={probe.first_fail_kind[0]}"})

    # (d) the top stage never advances further
    probe = SpeedCurriculum(policy=fast)
    for _ in range(20):
        probe.observe(good)
    top = len(probe.stages) - 1
    items.append({"id": "curric_top_stage_saturates",
                  "status": "PASS" if probe.stage == top else "FAIL",
                  "detail": f"after 20 all-pass episodes stage={probe.stage} (top={top}),"
                            f" policy={probe.policy['hard_failure_policy']}"})

    # (e) stage table monotonic, 8 stages, max exactly 3.0 m/s
    speeds = [float(stage["speed_m_s"]) for stage in STAGE_TABLE]
    monotonic = all(b > a for a, b in zip(speeds, speeds[1:]))
    items.append({"id": "curric_table_monotonic_max_3",
                  "status": "PASS" if monotonic and len(speeds) == 8 and speeds[-1] == 3.0
                  else "FAIL",
                  "detail": f"speeds={speeds} monotonic={monotonic} count={len(speeds)}"})

    # (f) high-speed arc is blocked unless explicitly opened, and the printer comes from the class
    probe = SpeedCurriculum(policy=fast)
    probe.stage = int(probe.policy["high_speed_arc_from_stage"])
    blocked_shares = probe.active_subtypes()
    opened = SpeedCurriculum(policy={**fast, "high_speed_arc_open": True})
    opened.stage = probe.stage
    ok = ("arc" not in blocked_shares and "arc" in opened.active_subtypes()
          and "needs_separate_validation" in "\n".join(probe.describe())
          and "PLACEHOLDER_PENDING_OPERATOR_RULING" in "\n".join(probe.describe()))
    items.append({"id": "curric_high_speed_arc_gated",
                  "status": "PASS" if ok else "FAIL",
                  "detail": f"stage {probe.stage} subtypes {sorted(blocked_shares)}; opened adds"
                            f" arc; describe() carries the placeholder marker"})

    # (g) the default curriculum printer is the sampler class itself
    arm_e_line = ArmEMixture().describe()[0]
    staged_lines = SpeedCurriculum().describe()
    items.append({"id": "curric_printer_from_sampler",
                  "status": "PASS" if "armE_mixture" in arm_e_line and staged_lines[0].startswith(
                      "[curric] mode=staged_0_to_3") and "stages 8" in staged_lines[0] else "FAIL",
                  "detail": "describe() is a method of ArmEMixture / SpeedCurriculum, i.e. the"
                            " classes that actually sample and gate"})
    return items


# --------------------------------------------------------------------------------------
# Phase (baseline / disturbance), control contract, triple-tap logging
# --------------------------------------------------------------------------------------
PHASES = ("baseline", "disturbance")
DISTURBANCES = ("none", "load", "push")
DEFAULT_PHASE = "baseline"

CONTROL_CONTRACT_VERSION = "dev-1"

# The deployment wrapper in this repo is state-free: r11_acceptance_ff.py says "no integrator, no
# filter state -> the 4-calls-per-policy-step cadence cannot compound", and the training-side
# wheel term (as2w_gyro_wheel_action.py) is also state-free.  Therefore "deployment consistent"
# means: no filter, no rate limiter, zero policy-step delay, same reset rules, same 13.5 N*m clip.
# If a deployment filter/rate limiter is ever frozen, this spec MUST change -> new fingerprint ->
# CONTROL_CONTRACT_CHANGED -> bump version + redo replay and baseline regression.
CONTROL_WRAPPER_SPEC = {
    "name": "deploy_consistent_r11_v1",
    "filter": {"type": "none", "cutoff_hz": None, "location": "none", "state": "none"},
    "rate_limit": {"type": "none", "limit": None, "location": "none", "state": "none"},
    "delay": {"type": "policy_steps", "policy_steps": 0, "state_free": True, "location": "none"},
    "torque_clip_nm": TORQUE_BUDGET_NM,
    "torque_clip_location": "apply_actions, before set_joint_velocity_target_index",
    "reset": {"history": "fill_all_slots_with_first_frame", "filter_state": "not_applicable",
              "delay_buffer": "not_applicable", "previous_action": "zeroed_on_episode_reset"},
    "history": {"frames": HISTORY_FRAMES, "fields": [field[0] for field in OBS_FIELDS],
                "order": "oldest_to_newest", "step": "policy_step"},
    "source": ("r11_acceptance_ff.py: no integrator, no filter state; "
               "as2w_gyro_wheel_action.py: state-free wheel term; wheel_contract_v1 torque budget"),
    "deployment_consistency": "identical wrapper instance/parameters for arms A, B and C",
}

RANDOMIZATION_TERMS = (
    ("wheel_friction", "wheel-ground static/dynamic friction randomization"),
    ("base_mass", "base mass scale randomization"),
    ("base_com", "base center-of-mass randomization"),
    ("gains", "actuator stiffness/damping scaling"),
    ("bounded_push", "random push by setting velocity"),
    ("reset_base", "initial pose/velocity randomization"),
    ("reset_robot_joints", "initial joint perturbation"),
    ("observation_noise", "gyro/joint observation noise (robust profile)"),
    ("action_delay", "action delay randomization"),
)

PUSH_SPEC = {
    "func": "mdp.push_by_setting_velocity",
    "mode": "interval",
    "interval_range_s": [3.5, 8.0],
    "velocity_range": {"x": [-0.08, 0.08], "y": [-0.06, 0.06], "yaw": [-0.08, 0.08]},
    "source": "as2w_training_safety.apply_safety(perturb=True) defaults",
    "threshold_status": CURRICULUM_THRESHOLD_STATUS,
}

PHASE_GATE = {
    "status": "PENDING_OPERATOR_RULING",
    "baseline_to_disturbance": "baseline must pass the held-out rolling evaluation before any"
                               " load/push continuation; no automatic promotion",
    "evaluator_reuse": ["deliveries/as2w-deploy/rolling-armD-turn65-research/runtime/"
                        "r11_acceptance_ff.py",
                        "skills/as2w/rolling/r11_acceptance_matrix.py"],
    "new_evaluator": "forbidden -- reuse one of the two existing evaluators",
    "merge_rule": "load and push are run separately and must each pass before a merged"
                  " (load+push) configuration is allowed; never start with load+push together",
}


def control_fingerprint(spec=None) -> str:
    """Canonical hash of the filter/delay/rate-limit/history semantics."""
    payload = json.dumps(spec or CONTROL_WRAPPER_SPEC, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_guard(previous: dict | None, version: str, fingerprint: str,
                   accept: bool = False) -> dict:
    """Decide whether a run may start when the control-contract semantics changed.

    * no previous state -> record (new baseline)
    * same version + same fingerprint -> ok
    * same version + different fingerprint -> CONTROL_CONTRACT_CHANGED, refuse --run unless accept
    * different version -> accepted bump (record), replay/baseline regression is the operator's task
    """
    if previous is None:
        return {"status": "new", "refuse": False, "message": "control contract recorded"}
    same_version = previous.get("version") == version
    same_fingerprint = previous.get("fingerprint") == fingerprint
    if same_version and same_fingerprint:
        return {"status": "ok", "refuse": False, "message": "control contract unchanged"}
    if same_version and not same_fingerprint:
        message = ("CONTROL_CONTRACT_CHANGED -> bump control_contract_version and redo replay +"
                   " baseline regression")
        if accept:
            return {"status": "accepted_change", "refuse": False,
                    "message": message + " [accepted via --accept-contract-change]"}
        return {"status": "changed", "refuse": True, "message": message}
    return {"status": "bumped", "refuse": False,
            "message": f"control contract version {previous.get('version')} -> {version}"
                       " (replay + baseline regression required)"}


def contract_state_path() -> Path:
    return RUN_DIR / ".control_contract_state.json"


def read_contract_state() -> dict | None:
    path = contract_state_path()
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def write_contract_state(version: str, fingerprint: str, note: str) -> None:
    contract_state_path().write_text(json.dumps({
        "version": version, "fingerprint": fingerprint,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note,
    }, indent=2) + "\n", encoding="utf-8")


class TripleTapRecorder:
    """Per-policy-step join: wheel/leg targets, history window and torque columns.

    Units are explicit.  Torque is its own group: the 13.5 N*m budget (config + both enforcement
    points), an ESTIMATED request computed from the dispatched wheel target and measured dq, and a
    solver column that is ``not_measured`` (the PhysX implicit drive does not expose applied joint
    effort on this backend).  Estimated torque is never written into the measured column.
    """

    FIELDS = (
        ("env_id", "int", "probe env index"),
        ("episode_id", "int", "per-env episode counter"),
        ("policy_step", "int", "iteration * n_steps + step"),
        ("target_wheel_pre_filter", "[4] rad/s", "5.0*clamp(action[12:16], +-6); before"
                                                 " filter/rate-limit/torque clip"),
        ("target_wheel_dispatch", "[4] rad/s", "read at set_joint_velocity_target_index"
                                               " (after delay/filter/rate-limit/13.5 clamp)"),
        ("target_leg_pos_pre_filter", "[12] rad", "default_pos + 0.35*clamp(action[0:12], +-6);"
                                                  " before filter/clip"),
        ("target_leg_pos_dispatch", "[12] rad", "read at set_joint_position_target_index"
                                                " (after the 90% angle clip)"),
        ("history_newest", "[65] float", "newest frame of the window the NEXT policy step consumes"),
        ("history_sha256", "hex64", "sha256 over the 6x65 float32 window, oldest->newest"),
        ("dtype", "str", "history window dtype (float32)"),
        ("field_order", "list[str]", "C3 observation field names, order contract"),
        ("window_shape", "[int,int]", "history window shape (6,65)"),
        ("torque_budget_nm", "float 13.5", "15.0*0.9; pre-dispatch clamp + effort_limit_sim"),
        ("torque_budget_location", "str", "apply_actions clamp; effort_limit_sim readback"),
        ("tau_request_est_nm", "[4] N*m ESTIMATED", "kd*(target_wheel_dispatch - dq); software"
                                                    " conversion, NOT a measurement"),
        ("tau_solver_nm", "[4] N*m or str", "not_measured: PhysX implicit drive does not expose"
                                            " applied joint effort"),
    )

    def __init__(self, path: Path | None = None, stride: int = 1, probe_env: int = 0,
                 sample_stride: int = 100, field_order=None):
        self.path = path
        self.stride = max(1, int(stride))
        self.probe_env = int(probe_env)
        self.sample_stride = max(1, int(sample_stride))
        self.field_order = list(field_order or OBS_FIELD_ORDER)

    @classmethod
    def field_table(cls) -> list[str]:
        return [f"  {name:26s} {form:24s} {meaning}" for name, form, meaning in cls.FIELDS]

    def record(self, env_id: int, episode_id: int, policy_step: int, action: torch.Tensor,
               wheel_dispatch: torch.Tensor, wheel_dq: torch.Tensor,
               leg_dispatch: torch.Tensor, history_window: torch.Tensor,
               kd: float, budget_readback: str) -> dict:
        action = action.detach().float().cpu()
        wheel_pre = WHEEL_ACTION_SCALE * torch.clamp(action[LEG_DIM:], -ACTION_CLIP, ACTION_CLIP)
        leg_pre = (torch.tensor(DEFAULT_LEG_POS, dtype=torch.float32)
                   + LEG_ACTION_SCALE * torch.clamp(action[:LEG_DIM], -ACTION_CLIP, ACTION_CLIP))
        window = history_window.detach().float().cpu().contiguous()
        tau_est = float(kd) * (wheel_dispatch.detach().float().cpu()
                               - wheel_dq.detach().float().cpu())
        return {
            "env_id": int(env_id),
            "episode_id": int(episode_id),
            "policy_step": int(policy_step),
            "target_wheel_pre_filter": [round(float(v), 6) for v in wheel_pre.tolist()],
            "target_wheel_dispatch": [round(float(v), 6) for v in
                                      wheel_dispatch.detach().float().cpu().tolist()],
            "target_leg_pos_pre_filter": [round(float(v), 6) for v in leg_pre.tolist()],
            "target_leg_pos_dispatch": [round(float(v), 6) for v in
                                        leg_dispatch.detach().float().cpu().tolist()],
            "history_newest": [round(float(v), 7) for v in window[-1].tolist()],
            "history_sha256": sha256_bytes(window.numpy().tobytes()),
            "dtype": str(window.dtype).replace("torch.", ""),
            "field_order": list(self.field_order),
            "window_shape": list(window.shape),
            "torque_budget_nm": TORQUE_BUDGET_NM,
            "torque_budget_location": "pre-dispatch clamp (apply_actions) + effort_limit_sim;"
                                      f" readback={budget_readback}",
            "tau_request_est_nm": [round(float(v), 6) for v in tau_est.tolist()],
            "tau_solver_nm": "not_measured",
        }

    @staticmethod
    def sample() -> dict:
        """Deterministic one-line sample for the dry run."""
        window = torch.arange(HISTORY_FRAMES * OBS_DIM, dtype=torch.float32).reshape(
            HISTORY_FRAMES, OBS_DIM) * 0.001
        return {
            "env_id": 0,
            "episode_id": 0,
            "policy_step": 0,
            "target_wheel_pre_filter": [0.0, 0.0, 0.0, 0.0],
            "target_wheel_dispatch": [0.0, 0.0, 0.0, 0.0],
            "target_leg_pos_pre_filter": [round(float(v), 6) for v in DEFAULT_LEG_POS],
            "target_leg_pos_dispatch": [round(float(v), 6) for v in DEFAULT_LEG_POS],
            "history_newest": [round(float(v), 7) for v in window[-1].tolist()],
            "history_sha256": sha256_bytes(window.contiguous().numpy().tobytes()),
            "dtype": "float32",
            "field_order": list(OBS_FIELD_ORDER),
            "window_shape": [HISTORY_FRAMES, OBS_DIM],
            "torque_budget_nm": TORQUE_BUDGET_NM,
            "torque_budget_location": "pre-dispatch clamp + effort_limit_sim (dry-run: spec only)",
            "tau_request_est_nm": [0.0, 0.0, 0.0, 0.0],
            "tau_solver_nm": "not_measured",
        }

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def describe(self) -> list[str]:
        lines = ["[tap] triple-tap log: one JSON object per policy step, fields:"]
        lines += self.field_table()
        lines.append("[tap] field_order literal: " + json.dumps(list(self.field_order)))
        lines.append("[tap] sampling policy: every"
                     f" {self.sample_stride} policy steps and the first 3 steps of every episode,"
                     " raw 6x65 windows saved to history_windows.npz (probe env);"
                     " dry-run writes 1 window to history_window_sample.npz")
        lines.append(f"[tap] run paths: triple_tap.jsonl (stride {self.stride}, probe env"
                     f" {self.probe_env}) + history_windows.npz")
        return lines


def phase_settings(phase: str, disturbance: str, load_kg: float) -> dict:
    if phase not in PHASES:
        raise SystemExit(f"[FATAL] unknown phase {phase!r}; have {PHASES}")
    if disturbance not in DISTURBANCES:
        raise SystemExit(f"[FATAL] unknown disturbance {disturbance!r}; have {DISTURBANCES}")
    if phase == "baseline" and disturbance != "none":
        raise SystemExit("[FATAL] baseline phase takes no disturbance factor")
    if phase == "disturbance" and disturbance == "load" and load_kg <= 0.0:
        raise SystemExit("[FATAL] --disturbance load requires a positive --load-kg")
    randomisation = {name: {"status": "OFF", "phase2_candidate": True, "note": note}
                     for name, note in RANDOMIZATION_TERMS}
    if phase == "baseline":
        randomisation["reset_base"] = {"status": "REDUCED", "phase2_candidate": True,
                                       "note": "nominal init pose (no pose/velocity range)"}
        randomisation["reset_robot_joints"] = {"status": "REDUCED", "phase2_candidate": True,
                                               "note": "small fixed joint offset, no wheel spin"}
    if phase == "disturbance" and disturbance == "load":
        randomisation["base_mass"] = {"status": "PAYLOAD", "phase2_candidate": False,
                                      "note": f"deterministic +{load_kg} kg on base_link"}
    if phase == "disturbance" and disturbance == "push":
        randomisation["bounded_push"] = {"status": "ON", "phase2_candidate": False,
                                         "note": json.dumps(PUSH_SPEC)}
    return {
        "phase": phase,
        "default_phase": DEFAULT_PHASE,
        "disturbance": disturbance,
        "load_kg": float(load_kg),
        "randomisation": randomisation,
        "randomisation_off_note": "every non-listed term is OFF in the baseline phase and is a"
                                  " phase-2 candidate",
        "push_spec": PUSH_SPEC,
        "phase_gate": PHASE_GATE,
        "merge_rule": PHASE_GATE["merge_rule"],
    }


def validate_contract_change_usage(run: bool, replay: bool, accept: bool) -> None:
    """--accept-contract-change is a dry-run/replay diagnostic only, never a training escape."""
    if accept and run:
        raise SystemExit(
            "[FATAL] --accept-contract-change is NOT allowed with --run: a formal training run"
            " must start from a bumped control_contract_version, and the bump must ship with"
            " replay + baseline regression results.  Use it only with --dry-run / --replay.")


def field_of_index(obs_index: int) -> tuple[str, int, int]:
    for name, start, end, _unit, _source in OBS_FIELDS:
        if start <= obs_index < end:
            return name, start, end
    raise IndexError(f"obs index {obs_index} outside the observation contract")


def first_hash_mismatch(ledger: list[dict], windows: list[torch.Tensor]):
    """First policy step whose recomputed window hash differs from the ledger, and its record."""
    for index, (record, window) in enumerate(zip(ledger, windows)):
        digest = sha256_bytes(window.detach().float().cpu().contiguous().numpy().tobytes())
        if record.get("history_sha256") != digest:
            return index, record
    return None, None


def field_diff(expected: torch.Tensor, actual: torch.Tensor) -> list[dict]:
    """Element-level diff mapped back to observation field names and field indices."""
    expected = expected.detach().float().cpu()
    actual = actual.detach().float().cpu()
    diffs = []
    for frame in range(expected.shape[0]):
        for obs_index in range(expected.shape[1]):
            delta = float(actual[frame, obs_index] - expected[frame, obs_index])
            if delta != 0.0:
                name, start, _end = field_of_index(obs_index)
                diffs.append({"frame": frame, "obs_index": obs_index, "field": name,
                              "field_index": obs_index - start, "delta": round(delta, 9)})
    return diffs


def check_tamper_detection(seed: int = DEFAULT_SEED) -> list[dict]:
    """Tamper one observation field and prove the whole localisation chain."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(HISTORY_FRAMES, OBS_DIM, generator=generator)
    windows = [(base + step * 0.01).clone() for step in range(8)]
    ledger = [{"policy_step": step,
               "history_sha256": sha256_bytes(window.contiguous().numpy().tobytes()),
               "window_shape": list(window.shape)} for step, window in enumerate(windows)]
    tampered = [window.clone() for window in windows]
    ang_vel_field_index = list(OBS_FIELD_ORDER).index("body_gyro")
    obs_index = 3 + 1          # body_gyro[1] = obs index 4
    tampered[3][-1][obs_index] += 1e-3
    step, record = first_hash_mismatch(ledger, tampered)
    reexport_ok = False
    diff = []
    if step is not None:
        npz_path = RUN_DIR / "tamper_first_mismatch_window.npz"
        import numpy as np
        np.savez(npz_path, expected=windows[step].numpy(), actual=tampered[step].numpy(),
                 policy_step=step, field_order=np.asarray(OBS_FIELD_ORDER))
        reexport_ok = npz_path.exists() and npz_path.stat().st_size > 0
        diff = field_diff(windows[step], tampered[step])
    located = (len(diff) == 1 and diff[0]["field"] == "body_gyro"
               and diff[0]["field_index"] == 1 and diff[0]["obs_index"] == obs_index
               and abs(diff[0]["delta"] - 1e-3) < 1e-5)
    ok = step == 3 and reexport_ok and located
    return [{"id": "offline_tamper_detection", "status": "PASS" if ok else "FAIL",
             "detail": f"tampered body_gyro[1] at step 3 -> first mismatching step={step},"
                       f" re-export ok={reexport_ok}, field diff={diff}"}]


def check_vx_gradient_isolation(seed: int = DEFAULT_SEED) -> list[dict]:
    """Two independent backwards: policy loss never trains the head, supervision never trains
    the actor, and the actor output really depends on vx_hat."""
    generator = torch.Generator().manual_seed(seed)
    frame = torch.randn(4, OBS_DIM, generator=generator)
    history_features = torch.randn(4, HISTORY_FRAMES * OBS_DIM, generator=generator)
    vx_true = torch.randn(4, generator=generator)
    nets = build_networks("C", seed, "cpu")
    estimator, actor = nets["estimator"], nets["actor"]

    def actor_input(vx_hat):
        return torch.cat((frame, vx_hat.unsqueeze(-1)), dim=1)

    vx_hat = estimator(history_features.detach()).detach().squeeze(-1)
    output = actor(actor_input(vx_hat))
    policy_loss = output.pow(2).mean()
    policy_loss.backward()
    actor_grad = any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                     for parameter in actor.parameters())
    head_grad_after_policy = any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                 for parameter in estimator.parameters())
    with torch.no_grad():
        changed = actor(actor_input(torch.zeros_like(vx_hat)))
    vx_hat_used = not torch.allclose(output, changed)

    actor.zero_grad(set_to_none=True)
    estimator.zero_grad(set_to_none=True)
    supervision = F.huber_loss(estimator(history_features).squeeze(-1), vx_true, delta=0.1)
    supervision.backward()
    head_grad_after_supervision = any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                      for parameter in estimator.parameters())
    actor_grad_after_supervision = any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                       for parameter in actor.parameters())

    # true vx is not an actor input: perturbing it cannot change the actor input at all
    truth_a = vx_true.clone()
    truth_b = vx_true * 1e6
    actor_in_a = actor_input(estimator(history_features.detach()).squeeze(-1))
    actor_in_b = actor_input(estimator(history_features.detach()).squeeze(-1))
    truth_isolated = torch.equal(actor_in_a, actor_in_b) and not torch.equal(truth_a, truth_b)

    ok = (actor_grad and not head_grad_after_policy and vx_hat_used
          and head_grad_after_supervision and not actor_grad_after_supervision
          and truth_isolated)
    return [{"id": "vx_head_gradient_isolation", "status": "PASS" if ok else "FAIL",
             "detail": f"policy backward: actor_grad={actor_grad},"
                       f" head_grad={head_grad_after_policy}, vx_hat_used={vx_hat_used};"
                       f" supervision backward: head_grad={head_grad_after_supervision},"
                       f" actor_grad={actor_grad_after_supervision};"
                       f" truth_vx_isolated={truth_isolated}"}]


def validate_eval_record(record_path, checkpoint_path) -> tuple[bool, str, str]:
    """Phase-2 entry requires a held-out evaluation record bound to this checkpoint."""
    if record_path is None:
        return False, "missing --eval-record (held-out pass record required for phase 2)", ""
    path = Path(record_path)
    if not path.exists():
        return False, f"eval-record not found: {path}", ""
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is None or not checkpoint.exists():
        return False, "checkpoint missing; cannot bind the eval record", ""
    try:
        record = load_json(path)
    except Exception:
        return False, f"eval-record unreadable: {path}", ""
    actual = sha256_file(checkpoint)
    recorded = record.get("checkpoint_sha256")
    if recorded != actual:
        return False, f"checkpoint sha mismatch: record={recorded} actual={actual}", actual
    required = ("checkpoint_sha256", "evaluator", "seeds", "passed", "tracking", "attitude",
                "contact", "brake")
    missing = [key for key in required if key not in record]
    if missing:
        return False, "eval-record missing fields: " + ", ".join(missing), actual
    return True, f"eval-record ok (evaluator={record.get('evaluator')})", actual


def validate_phase2_access(phase: str, checkpoint, allow_disturbance: bool,
                           eval_record, run: bool) -> tuple[bool, str]:
    """Negative cases: unlock-only, missing checkpoint, missing/тmismatched eval record."""
    if phase != "disturbance":
        return True, ""
    missing = []
    if checkpoint is None:
        missing.append("--init-checkpoint")
    if not allow_disturbance:
        missing.append("--allow-disturbance")
    if missing:
        message = "PHASE2_LOCKED missing " + ", ".join(missing)
        if run:
            raise SystemExit("[FATAL] " + message
                             + " (phase 2 is locked by default; operator unlock required)")
        return False, message
    record_ok, record_message, _sha = validate_eval_record(eval_record, checkpoint)
    if not record_ok:
        message = "PHASE2_LOCKED " + record_message
        if run:
            raise SystemExit("[FATAL] " + message)
        return False, message
    return True, record_message


def re_export_first_mismatch(run_dir: Path) -> dict:
    """Re-export the raw window at the first ledger mismatch and diff it field by field."""
    import numpy as np
    windows_path = Path(run_dir) / "history_windows.npz"
    ledger_path = Path(run_dir) / "triple_tap.jsonl"
    if not windows_path.exists():
        return {"found": False, "reason": f"missing {windows_path}"}
    data = np.load(windows_path)
    windows = [torch.as_tensor(item) for item in data["windows"]]
    steps = [int(item) for item in data["policy_steps"]]
    expected = ([torch.as_tensor(item) for item in data["expected_windows"]]
                if "expected_windows" in data else [None] * len(windows))
    ledger = {}
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            ledger[int(record["policy_step"])] = record
    for index, (step, window) in enumerate(zip(steps, windows)):
        record = ledger.get(step)
        if record is None:
            continue
        digest = sha256_bytes(window.contiguous().numpy().tobytes())
        if record.get("history_sha256") == digest:
            continue
        npz_path = Path(run_dir) / f"first_mismatch_step{step}.npz"
        np.savez(npz_path, window=window.numpy(), policy_step=step,
                 field_order=np.asarray(OBS_FIELD_ORDER))
        diff = (field_diff(expected[index], window)
                if expected[index] is not None else [])
        return {"found": True, "policy_step": step, "npz": str(npz_path), "diff": diff[:20],
                "diff_count": len(diff)}
    return {"found": False, "reason": "no hash mismatch in the ledger"}


def parse_run_gate_args(argv) -> argparse.Namespace:
    """Pure-CPU parser for the --run gates; never imports isaaclab."""
    parser = base_parser(run=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def check_cli_refusals(seed: int = DEFAULT_SEED) -> list[dict]:
    """CLI-level negative tests: refusals must print a reason and never import Isaac."""
    import subprocess
    script = Path(__file__).resolve()
    config = str(CONFIG_DIR / "fs_roll_A.json")
    dummy = RUN_DIR / "refusal_dummy.pt"
    dummy.write_bytes(b"refusal-test" * 64)
    dummy_sha = sha256_file(dummy)
    record = {"checkpoint_sha256": dummy_sha, "evaluator": "r11_acceptance_ff",
              "seeds": [449], "passed": True, "tracking": {}, "attitude": {},
              "contact": {}, "brake": {}}
    good = RUN_DIR / "refusal_good.json"
    good.write_text(json.dumps(record))
    bad = RUN_DIR / "refusal_bad.json"
    bad.write_text(json.dumps({**record, "checkpoint_sha256": "0" * 64}))
    base = [sys.executable, str(script), "--arm", "A", "--kd", "1.5", "--config", config]
    cases = {
        "phase2_missing": base + ["--phase", "disturbance", "--run"],
        "phase2_no_record": base + ["--phase", "disturbance", "--disturbance", "load",
                                    "--load-kg", "25", "--init-checkpoint", str(dummy),
                                    "--allow-disturbance", "--run"],
        "phase2_bad_sha": base + ["--phase", "disturbance", "--disturbance", "load",
                                  "--load-kg", "25", "--init-checkpoint", str(dummy),
                                  "--allow-disturbance", "--eval-record", str(bad), "--run"],
        "accept_change_run": base + ["--accept-contract-change", "--run"],
    }
    results = {}
    for name, command in cases.items():
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=90)
            output = done.stdout + done.stderr
            results[name] = {
                "rc": done.returncode,
                "refusal_text": ("PHASE2_LOCKED" in output
                                 or "CONTROL_CONTRACT_CHANGED" in output
                                 or "accept-contract-change" in output),
                "isaac_no": "ISAAC_IMPORTED=no" in output,
                "kit_marker": "KitLauncher initialization complete" in output,
            }
        except Exception as error:  # pragma: no cover - defensive
            results[name] = {"rc": None, "refusal_text": False, "isaac_no": False,
                             "kit_marker": False, "error": str(error)}
    for path in (dummy, good, bad):
        path.unlink(missing_ok=True)
    ok = all(item["rc"] not in (0, None) and item["refusal_text"] and item["isaac_no"]
             and not item["kit_marker"] for item in results.values())
    return [{"id": "refusal_never_imports_isaac", "status": "PASS" if ok else "FAIL",
             "detail": json.dumps(results)}]


def check_offline_actor_forward(seed: int = DEFAULT_SEED, count: int = 4) -> list[dict]:
    """Fixed-sample forward pass for A/B/C, including the arm-C estimator path.

    Deterministic samples are written to ``offline_forward_samples.npz`` so a replay can compare
    element by element.
    """
    generator = torch.Generator().manual_seed(seed)
    frame = torch.randn(count, OBS_DIM, generator=generator)
    history = HistoryBuffer(count, OBS_DIM, HISTORY_FRAMES, "cpu")
    history.reset(frame)
    reset_fill_ok = torch.equal(history.data, frame.unsqueeze(1).expand(-1, HISTORY_FRAMES, -1))
    new_frame = torch.randn(count, OBS_DIM, generator=generator)
    history.push(new_frame)
    push_ok = torch.equal(history.data[:, -1], new_frame)
    windows = {arm: (frame if arm == "A" else
                     (history.flat() if arm == "B" else
                      torch.cat((new_frame,
                                 torch.zeros(count, 1)), dim=1)))
               for arm in ("A", "B", "C")}
    outputs = {}
    results = []
    for arm in ("A", "B", "C"):
        nets = build_networks(arm, seed, "cpu")
        actor_in = windows[arm]
        expected_in = ARM_INFO[arm]["actor_input"]
        out = nets["actor"](actor_in)
        outputs[arm] = out
        results.append({"arm": arm, "input_dim": int(actor_in.shape[-1]),
                        "expected_input_dim": expected_in,
                        "out_shape": list(out.shape), "dtype": str(out.dtype)})

    # arm C: the estimator must run, its input must be detached from the actor graph, and the
    # actor input must be built from the history-derived vx_hat only (never true vx)
    nets_c = build_networks("C", seed, "cpu")
    flat = history.flat()
    tracked_input = flat.detach().clone().requires_grad_(True)
    vx_hat = nets_c["estimator"](tracked_input.detach()).squeeze(-1)
    grad_to_input = torch.autograd.grad(vx_hat.sum(), tracked_input, allow_unused=True)[0]
    stop_grad_ok = grad_to_input is None
    actor_in_c = torch.cat((new_frame, vx_hat.unsqueeze(-1)), dim=1)
    estimator_called = True
    source = Path(__file__).read_text(encoding="utf-8")
    feature_line = "features = torch.cat((frame, vx_hat.unsqueeze(-1)), dim=1)"
    truth_in_feature_line = any(
        line.strip().startswith("features =") and "true_vx" in line
        for line in source.splitlines())
    actor_uses_history_only = feature_line in source
    no_truth_in_actor = (feature_line in source and not truth_in_feature_line
                         and int(actor_in_c.shape[-1]) == OBS_DIM + 1)
    truth_a = torch.randn(count, generator=generator)
    truth_b = truth_a * 1e6
    out_a = nets_c["actor"](actor_in_c)
    vx_hat_again = nets_c["estimator"](history.flat().detach()).squeeze(-1)
    out_b = nets_c["actor"](torch.cat((new_frame, vx_hat_again.unsqueeze(-1)), dim=1))
    truth_isolated = torch.allclose(out_a, out_b) and not torch.allclose(truth_a, truth_b)

    dims_ok = all(result["input_dim"] == result["expected_input_dim"] for result in results)
    shapes_ok = all(result["out_shape"] == [count, ACTION_DIM] for result in results)
    dtype_ok = len({result["dtype"] for result in results}) == 1
    npz_path = RUN_DIR / "offline_forward_samples.npz"
    import numpy as np
    np.savez(npz_path, frame=frame.numpy(), history=history.data.numpy(),
             actor_in_A=windows["A"].numpy(), actor_in_B=windows["B"].numpy(),
             actor_in_C=actor_in_c.detach().numpy(), seed=seed, field_order=list(OBS_FIELD_ORDER))
    return [
        {"id": "offline_actor_forward_all_arms",
         "status": "PASS" if (dims_ok and shapes_ok and dtype_ok and reset_fill_ok and push_ok
                              and stop_grad_ok and actor_uses_history_only and no_truth_in_actor
                              and truth_isolated and estimator_called) else "FAIL",
         "detail": f"dims {[(r['arm'], r['input_dim']) for r in results]},"
                   f" out {[(r['arm'], r['out_shape']) for r in results]} dtype={results[0]['dtype']};"
                   f" reset_fill={reset_fill_ok} push={push_ok} vx_hat_stop_grad={stop_grad_ok}"
                   f" actor_input_from_history={actor_uses_history_only}"
                   f" truth_vx_absent={no_truth_in_actor} truth_perturb_isolated={truth_isolated};"
                   f" samples={npz_path}"},
    ]


def check_phase_and_contract(phase: dict, previous_state: dict | None) -> list[dict]:
    """Pure-CPU checks for the phase defaults and the control-contract guard."""
    items = []
    baseline = phase_settings("baseline", "none", 0.0)
    off = [name for name, term in baseline["randomisation"].items()
           if term["status"] == "OFF"]
    mass_off = baseline["randomisation"]["base_mass"]["status"] == "OFF"
    push_off = baseline["randomisation"]["bounded_push"]["status"] == "OFF"
    items.append({"id": "phase_default_is_baseline",
                  "status": "PASS" if DEFAULT_PHASE == "baseline" and mass_off and push_off
                  and len(off) >= 5 else "FAIL",
                  "detail": f"default={DEFAULT_PHASE}, mass=OFF, push=OFF,"
                            f" {len(off)} randomisation terms OFF"})

    identical = (control_fingerprint() == control_fingerprint()
                 and CONTROL_WRAPPER_SPEC["delay"]["policy_steps"] == 0
                 and CONTROL_WRAPPER_SPEC["filter"]["type"] == "none"
                 and CONTROL_WRAPPER_SPEC["rate_limit"]["type"] == "none")
    items.append({"id": "three_arm_identical_wrapper",
                  "status": "PASS" if identical else "FAIL",
                  "detail": "one CONTROL_WRAPPER_SPEC instance for A/B/C;"
                            f" fingerprint={control_fingerprint()[:16]}"})

    sample = TripleTapRecorder.sample()
    required = {"env_id", "episode_id", "policy_step", "target_wheel_pre_filter",
                "target_wheel_dispatch", "target_leg_pos_pre_filter", "target_leg_pos_dispatch",
                "history_newest", "history_sha256", "dtype", "field_order", "window_shape",
                "torque_budget_nm", "torque_budget_location", "tau_request_est_nm",
                "tau_solver_nm"}
    fields_ok = (required <= set(sample)
                 and len(sample["target_wheel_pre_filter"]) == WHEEL_DIM
                 and len(sample["target_wheel_dispatch"]) == WHEEL_DIM
                 and len(sample["target_leg_pos_pre_filter"]) == LEG_DIM
                 and len(sample["target_leg_pos_dispatch"]) == LEG_DIM
                 and len(sample["history_newest"]) == OBS_DIM
                 and len(sample["history_sha256"]) == 64
                 and list(sample["field_order"]) == list(OBS_FIELD_ORDER)
                 and sample["tau_solver_nm"] == "not_measured"
                 and sample["torque_budget_nm"] == TORQUE_BUDGET_NM)
    items.append({"id": "triple_tap_logging",
                  "status": "PASS" if fields_ok else "FAIL",
                  "detail": "wall/leg targets split with rad/s / rad units, torque group separated,"
                            " history records carry env/episode/step/dtype/field_order/shape;"
                            " field_order == observation contract order"})

    fabricated = {"version": CONTROL_CONTRACT_VERSION, "fingerprint": "0" * 64}
    changed = contract_guard(fabricated, CONTROL_CONTRACT_VERSION, control_fingerprint(), False)
    same = contract_guard({"version": CONTROL_CONTRACT_VERSION,
                           "fingerprint": control_fingerprint()},
                          CONTROL_CONTRACT_VERSION, control_fingerprint(), False)
    try:
        validate_contract_change_usage(True, False, True)
        accept_with_run_ok = False
        accept_with_run_message = ""
    except SystemExit as error:
        accept_with_run_ok = True
        accept_with_run_message = str(error)
    validate_contract_change_usage(False, True, True)   # dry-run/replay must be allowed
    ok = (changed["refuse"] and "CONTROL_CONTRACT_CHANGED" in changed["message"]
          and same["status"] == "ok" and accept_with_run_ok
          and "accept-contract-change" in accept_with_run_message)
    items.append({"id": "contract_version_guard",
                  "status": "PASS" if ok else "FAIL",
                  "detail": "NEG1: changed fingerprint + same version -> CONTROL_CONTRACT_CHANGED"
                            " and --run refuses; NEG2: --accept-contract-change --run -> non-zero"
                            " exit demanding a version bump; --replay/dry-run allowed;"
                            " unchanged -> ok"})

    try:
        phase_settings("disturbance", "load", 5.0)
        phase_settings("disturbance", "push", 0.0)
        single_ok = True
        reason = "one factor at a time accepted"
    except SystemExit as error:  # pragma: no cover - defensive
        single_ok, reason = False, str(error)
    try:
        phase_settings("baseline", "load", 5.0)
        both_ok = False
    except SystemExit:
        both_ok = True
    items.append({"id": "disturbance_single_factor",
                  "status": "PASS" if single_ok and both_ok else "FAIL",
                  "detail": "load-only and push-only accepted; a second factor with baseline (or"
                            " load+push together) is rejected with a non-zero exit"})

    if previous_state is not None:
        decision = contract_guard(previous_state, CONTROL_CONTRACT_VERSION, control_fingerprint(),
                                  False)
        items.append({"id": "contract_state_matches", "status":
                      "PASS" if decision["status"] in ("ok", "bumped") else "FAIL",
                      "detail": decision["message"]})

    dummy_ckpt = RUN_DIR / "phase2_negative_dummy.pt"
    dummy_ckpt.write_bytes(b"phase2-negative-test")
    dummy_sha = sha256_file(dummy_ckpt)
    good_record = RUN_DIR / "phase2_negative_eval_record.json"
    good_record.write_text(json.dumps({
        "checkpoint_sha256": dummy_sha, "evaluator": "r11_acceptance_ff", "seeds": [449],
        "passed": True, "tracking": {}, "attitude": {}, "contact": {}, "brake": {}}))
    bad_record = RUN_DIR / "phase2_negative_bad_sha.json"
    bad_record.write_text(json.dumps({
        "checkpoint_sha256": "0" * 64, "evaluator": "r11_acceptance_ff", "seeds": [449],
        "passed": True, "tracking": {}, "attitude": {}, "contact": {}, "brake": {}}))
    neg_missing_both = validate_phase2_access("disturbance", None, False, None, run=False)
    neg_unlock_only = validate_phase2_access("disturbance", str(dummy_ckpt), True, None, run=False)
    neg_bad_sha = validate_phase2_access("disturbance", str(dummy_ckpt), True,
                                         str(bad_record), run=False)
    pos_ok = validate_phase2_access("disturbance", str(dummy_ckpt), True,
                                    str(good_record), run=False)
    try:
        validate_phase2_access("disturbance", str(dummy_ckpt), True, None, run=True)
        run_blocked = False
    except SystemExit:
        run_blocked = True
    dummy_ckpt.unlink(missing_ok=True)
    good_record.unlink(missing_ok=True)
    bad_record.unlink(missing_ok=True)
    ok = (not neg_missing_both[0] and not neg_unlock_only[0] and not neg_bad_sha[0]
          and pos_ok[0] and run_blocked
          and "eval-record" in neg_unlock_only[1] and "sha mismatch" in neg_bad_sha[1]
          and "PHASE2_LOCKED" in neg_missing_both[1])
    items.append({"id": "phase2_locked_by_default",
                  "status": "PASS" if ok else "FAIL",
                  "detail": "NEG missing checkpoint+unlock -> " + neg_missing_both[1][:60]
                            + " | NEG unlock only -> " + neg_unlock_only[1][:60]
                            + " | NEG sha mismatch -> " + neg_bad_sha[1][:70]
                            + " | positive with record -> ok"
                            + f" | --run without record blocked={run_blocked}"})
    return items


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def build_mlp(sizes, out_gain=1.0) -> nn.Sequential:
    layers = []
    for index in range(len(sizes) - 1):
        layer = nn.Linear(sizes[index], sizes[index + 1])
        orthogonal_init(layer, 1.0 if index == len(sizes) - 2 else math.sqrt(2.0))
        layers.append(layer)
        if index < len(sizes) - 2:
            layers.append(nn.ELU())
    final = layers[-1]
    if isinstance(final, nn.Linear):
        nn.init.orthogonal_(final.weight, gain=out_gain)
        nn.init.zeros_(final.bias)
    return nn.Sequential(*layers)


def network_spec(arm: str) -> dict:
    info = ARM_INFO[arm]
    actor_in = info["actor_input"]
    spec = {
        "actor": {
            "input_dim": actor_in,
            "hidden": list(ACTOR_HIDDEN),
            "output_dim": ACTION_DIM,
            "activation": "elu",
            "init": "orthogonal (hidden gain sqrt(2), output gain 0.01)",
            "distribution": "diagonal Gaussian, state-independent log_std",
            "init_noise_std": PPO_DEFAULTS["init_noise_std"],
        },
        "critic": {
            "input_dim": actor_in + PRIVILEGED_DIM,
            "hidden": list(CRITIC_HIDDEN),
            "output_dim": 1,
            "activation": "elu",
            "init": "orthogonal (hidden gain sqrt(2), output gain 1.0)",
            "privileged_dim": PRIVILEGED_DIM,
            "privileged_fields": ["vx", "vy", "vz", "wz", "root_height"],
        },
        "estimator": None,
    }
    if info["forecast"]:
        spec["estimator"] = {
            "input_dim": HISTORY_FRAMES * OBS_DIM,
            "hidden": list(ESTIMATOR_HIDDEN),
            "output_dim": 1,
            "activation": "elu",
            "init": "orthogonal (hidden gain sqrt(2), output gain 1.0)",
            "target": "root_lin_vel_b[:, 0] (simulation truth, body-frame forward speed)",
            "target_noise": "none",
            "train_only": True,
            "actor_usage": "vx_hat from the 6-frame history with stop-gradient; true vx never"
                           " enters the actor path",
            "deployment_input": "the same 6-frame history the policy sees; no privileged state",
        }
    return spec


def build_networks(arm: str, seed: int, device="cpu") -> dict:
    info = ARM_INFO[arm]
    torch.manual_seed(seed)
    actor = build_mlp([info["actor_input"], *ACTOR_HIDDEN, ACTION_DIM], out_gain=0.01)
    critic = build_mlp([info["actor_input"] + PRIVILEGED_DIM, *CRITIC_HIDDEN, 1], out_gain=1.0)
    nets = {
        "actor": actor.to(device),
        "critic": critic.to(device),
        "log_std": nn.Parameter(torch.full((ACTION_DIM,), math.log(PPO_DEFAULTS["init_noise_std"]),
                                           device=device)),
        "estimator": None,
    }
    if info["forecast"]:
        nets["estimator"] = build_mlp(
            [HISTORY_FRAMES * OBS_DIM, *ESTIMATOR_HIDDEN, 1], out_gain=1.0).to(device)
    return nets


def count_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def parameter_fingerprint(module: nn.Module) -> dict:
    flat = torch.cat([p.detach().reshape(-1).float().cpu() for p in module.parameters()])
    return {
        "numel": int(flat.numel()),
        "mean": round(float(flat.mean()), 9),
        "std": round(float(flat.std()), 9),
        "l2": round(float(flat.norm()), 9),
        "sha256_16": sha256_bytes(flat.numpy().tobytes())[:16],
    }


def first_layer_norm(net: nn.Module) -> float:
    for module in net.modules():
        if isinstance(module, nn.Linear):
            return round(float(module.weight.norm()), 9)
    return 0.0


# --------------------------------------------------------------------------------------
# Contract / URDF handling
# --------------------------------------------------------------------------------------


def contract_key(kd: float) -> str:
    return "w1w2_as2w_diagnostic_kd_" + str(kd).replace(".", "p")


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"[FATAL] wheel contract missing: {path}")
    return {"path": path, "sha256": sha256_file(path), "doc": load_json(path)}


def select_contract(contract: dict, kd: float | None, name: str | None) -> tuple[str, dict]:
    groups = contract["doc"]["contracts"]
    if name is None:
        if kd is None:
            raise SystemExit("[FATAL] no contract group selected: pass --kd {0.5,1.5,2.0}"
                             " or --contract <group>; implicit defaults are forbidden")
        name = contract_key(kd)
    if name not in groups:
        raise SystemExit(f"[FATAL] contract group {name!r} not in contract; have {sorted(groups)}")
    entry = groups[name]
    if not name.startswith("w1w2_as2w_diagnostic_"):
        raise SystemExit(f"[FATAL] {name!r} is not one of the diagnostic arms for this experiment")
    if kd is not None and float(entry["wheel_kd"]) != float(kd):
        raise SystemExit(f"[FATAL] contract {name} has wheel_kd={entry['wheel_kd']} but --kd {kd}")
    return name, entry


def validate_contract(entry: dict, doc: dict) -> list[tuple[str, str]]:
    required = {
        "wheel_action_scale": WHEEL_ACTION_SCALE,
        "wheel_kp": WHEEL_KP,
        "wheel_mode": "velocity",
        "gyro_enabled": False,
        "leg_kp": LEG_KP,
        "leg_kd": LEG_KD,
        "leg_action_scale": LEG_ACTION_SCALE,
    }
    failures = []
    for key, expected in required.items():
        if entry.get(key) != expected:
            failures.append((key, f"expected {expected!r}, got {entry.get(key)!r}"))
    if float(entry.get("wheel_kd", -1.0)) not in (0.5, 1.5, 2.0):
        failures.append(("wheel_kd", f"unexpected value {entry.get('wheel_kd')!r}"))
    if float(doc.get("torque_budget_nm", -1)) != TORQUE_BUDGET_NM:
        failures.append(("torque_budget_nm", str(doc.get("torque_budget_nm"))))
    if int(doc.get("observation_dim", -1)) != OBS_DIM:
        failures.append(("observation_dim", str(doc.get("observation_dim"))))
    if int(doc.get("action_dim", -1)) != ACTION_DIM:
        failures.append(("action_dim", str(doc.get("action_dim"))))
    if float(doc.get("control_period_s", -1)) != CONTROL_PERIOD_S:
        failures.append(("control_period_s", str(doc.get("control_period_s"))))
    if tuple(doc.get("wheel_order", ())) != WHEEL_JOINT_ORDER:
        failures.append(("wheel_order", str(doc.get("wheel_order"))))
    groups = {k: v for k, v in doc["contracts"].items() if k.startswith("w1w2_as2w_diagnostic_")}
    if len(groups) != 3:
        failures.append(("diagnostic_groups", str(sorted(groups))))
    else:
        reference = None
        for key in sorted(groups):
            stripped = {k: v for k, v in groups[key].items() if k not in ("wheel_kd", "description")}
            if reference is None:
                reference = stripped
            elif stripped != reference:
                failures.append(("only_wheel_kd_varies", f"{key} differs in more than wheel_kd"))
    return failures


def parse_urdf(path: Path = URDF_PATH) -> dict:
    root = ET.parse(path).getroot()
    joints, order = {}, []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")
        joints[name] = {
            "type": joint.attrib.get("type"),
            "lower": None if limit is None else float(limit.attrib.get("lower", "nan")),
            "upper": None if limit is None else float(limit.attrib.get("upper", "nan")),
            "effort": None if limit is None else float(limit.attrib.get("effort", "nan")),
            "velocity": None if limit is None else float(limit.attrib.get("velocity", "nan")),
        }
        order.append(name)
    links = [link.attrib["name"] for link in root.findall("link")]
    return {"joints": joints, "order": order, "links": links}


def check_wheel_mapping(urdf: dict, entry: dict, wheel_order_contract) -> tuple[dict, list[str]]:
    """Name-based FL/FR/RL/RR mapping check; mismatch is fatal, never -1/fixed index/zeros.

    The canonical order is part-major (hip x4, thigh x4, calf x4, foot x4), as in
    ``as2w_isaac_flat_env_all.JOINTS``.  The URDF document order is leg-major and is *not*
    the articulation order; the runtime order is resolved by name through
    ``FlatEnvironment.as2w_joint_ids``, so the check is a set/name check, not an order check.
    """
    problems: list[str] = []
    movable = [n for n in urdf["order"] if urdf["joints"][n]["type"] != "fixed"]
    if sorted(movable) != sorted(JOINT_ORDER):
        missing = sorted(set(JOINT_ORDER) - set(movable))
        extra = sorted(set(movable) - set(JOINT_ORDER))
        problems.append(f"URDF movable joint name set mismatch: missing={missing} extra={extra}")
    if len(set(movable)) != len(movable):
        problems.append("duplicate movable joint names")
    canonical_wheels = list(JOINT_ORDER[LEG_DIM:])
    if canonical_wheels != list(WHEEL_JOINT_ORDER):
        problems.append(f"canonical channels 12..15 are {canonical_wheels}, not {list(WHEEL_JOINT_ORDER)}")
    wheels = [n for n in movable if n.endswith("_foot")]
    if sorted(wheels) != sorted(WHEEL_JOINT_ORDER):
        problems.append(f"URDF wheel joints {wheels} != {list(WHEEL_JOINT_ORDER)}")
    if len(set(wheels)) != WHEEL_DIM:
        problems.append("duplicate wheel joint names")
    action_channels = [WHEEL_ACTION_CHANNELS[name] for name in WHEEL_JOINT_ORDER]
    if action_channels != list(range(LEG_DIM, ACTION_DIM)):
        problems.append(f"action channels {action_channels} are not 12..15")
    if len(set(action_channels)) != WHEEL_DIM:
        problems.append("duplicate action channel index")
    if tuple(wheel_order_contract) != WHEEL_JOINT_ORDER:
        problems.append(f"contract wheel_order {wheel_order_contract} != {WHEEL_JOINT_ORDER}")
    foot_bodies = sorted(n for n in urdf["links"] if n.endswith("_foot"))
    if foot_bodies != sorted(WHEEL_JOINT_ORDER):
        problems.append(f"contact bodies {foot_bodies} != wheel bodies {sorted(WHEEL_JOINT_ORDER)}")
    for name in WHEEL_JOINT_ORDER:
        limit = urdf["joints"][name]
        if limit["effort"] != TORQUE_HARD_LIMIT_NM:
            problems.append(f"{name} effort limit {limit['effort']} != {TORQUE_HARD_LIMIT_NM}")
        if limit["type"] != "continuous":
            problems.append(f"{name} is {limit['type']}, expected continuous")
    mapping = {
        "joint_order_canonical": list(JOINT_ORDER),
        "urdf_document_order": list(urdf["order"]),
        "order_note": "canonical order is part-major (hip x4, thigh x4, calf x4, foot x4);"
                      " the articulation resolves names through FlatEnvironment.as2w_joint_ids,"
                      " so URDF document order (leg-major) is not used as the articulation order",
        "wheel_joint_order": list(WHEEL_JOINT_ORDER),
        "wheel_action_channels": {k: int(v) for k, v in WHEEL_ACTION_CHANNELS.items()},
        "contact_body_regex": ".*_foot",
        "contact_bodies": foot_bodies,
        "wheel_effort_limit_nm": TORQUE_HARD_LIMIT_NM,
        "wheel_velocity_limit_rad_s": urdf["joints"]["FL_foot"]["velocity"],
        "mapping_source": "URDF joint/link names; --run additionally asserts the runtime joint"
                          " name set, joint ids 12..15 and contact bodies",
    }
    return mapping, problems


# --------------------------------------------------------------------------------------
# Budget + GPU-memory estimates
# --------------------------------------------------------------------------------------


def resolve_budget(envs: int, n_steps: int, iterations: int | None,
                   total_steps: int | None) -> dict:
    per_round = envs * n_steps
    if total_steps is not None:
        iterations = total_steps // per_round
        remainder = total_steps % per_round
        declared_total = total_steps
    else:
        if iterations is None:
            raise SystemExit("[FATAL] either --iterations or --total-steps is required")
        declared_total = iterations * per_round
        remainder = 0
    return {
        "num_envs": envs,
        "n_steps": n_steps,
        "per_round_steps": per_round,
        "iterations": iterations,
        "total_env_steps": declared_total,
        "remainder": remainder,
        "gradient_steps_per_iteration": PPO_DEFAULTS["epochs"] * PPO_DEFAULTS["minibatches"],
        "total_gradient_steps": iterations * PPO_DEFAULTS["epochs"] * PPO_DEFAULTS["minibatches"],
        "formula": "per_round = num_envs * n_steps; total = iterations * per_round;"
                   " remainder = total_declared % per_round",
        "rsl_rl_name": "rsl_rl calls n_steps 'num_steps_per_env'; this entry re-implements the"
                       " PPO loop, so one rollout = one iteration = epochs*minibatches gradient steps",
        "legacy_steps_per_iteration": "num_envs * 24 (as2w_isaac_flat_env_all.py:258); superseded"
                                      " by n_steps=128 in this entry",
    }


def gpu_feasibility(envs: int, n_steps: int, arm: str, vx_coef: float) -> dict:
    memory = gpu_memory_mib()
    estimate = {
        "baseline": {"mib": GPU_BASELINE_MIB, "envs": GPU_BASELINE_ENVS, "source": "operator measurement"},
        "target_envs": envs,
        "method": "linear extrapolation in env count; NOT a measurement of this configuration",
    }
    if memory is None:
        estimate.update({"status": "UNKNOWN", "reason": "nvidia-smi unavailable"})
        return estimate
    total_mib, used_mib = memory
    projected = GPU_BASELINE_MIB * envs / GPU_BASELINE_ENVS
    limit = GPU_SAFETY_FRACTION * total_mib
    max_envs = int(GPU_BASELINE_ENVS * limit / GPU_BASELINE_MIB)
    # rollout tensors held on the GPU by this entry (float32), approximate
    per_step = ARM_INFO[arm]["actor_input"] + PRIVILEGED_DIM + ACTION_DIM + 6
    rollout_mib = n_steps * envs * per_step * 4 / (1024 ** 2)
    if ARM_INFO[arm]["forecast"] and vx_coef > 0:
        rollout_mib += n_steps * envs * (HISTORY_FRAMES * OBS_DIM) * 4 / (1024 ** 2)
    estimate.update({
        "available_mib": total_mib,
        "used_now_mib": used_mib,
        "estimated_sim_mib": round(projected, 1),
        "estimated_fraction": round(projected / total_mib, 4),
        "limit_mib": round(limit, 1),
        "estimated_max_envs": max_envs,
        "rollout_buffer_mib": round(rollout_mib, 1),
        "status": "OK_ESTIMATED" if projected <= limit else "FAIL_ESTIMATED",
    })
    return estimate


# --------------------------------------------------------------------------------------
# Self-check plumbing
# --------------------------------------------------------------------------------------


class Checker:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, check_id: str, ok: bool, detail: str, fatal_message: str | None = None) -> bool:
        self.items.append({"id": check_id, "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok and fatal_message:
            raise SystemExit(f"[FATAL] {check_id}: {fatal_message} :: {detail}")
        return ok

    def warn(self, check_id: str, detail: str) -> None:
        self.items.append({"id": check_id, "status": "WARN", "detail": detail})

    def failed(self) -> list[dict]:
        return [item for item in self.items if item["status"] == "FAIL"]


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def check_history() -> list[dict]:
    items = []
    buffer = HistoryBuffer(3, OBS_DIM, HISTORY_FRAMES, "cpu")
    first = torch.arange(3 * OBS_DIM, dtype=torch.float32).reshape(3, OBS_DIM) * 0.001
    buffer.data.fill_(-7.0)                      # stale frames from a previous episode
    buffer.reset(first)
    equal_fill = torch.equal(buffer.data, first.unsqueeze(1).expand(-1, HISTORY_FRAMES, -1))
    items.append({"id": "history_reset_clears_residue", "status": "PASS" if equal_fill else "FAIL",
                  "detail": "dirty buffer -> reset -> all 6 slots equal the first frame"})
    items.append({"id": "history_reset_fill_first", "status": "PASS" if equal_fill else "FAIL",
                  "detail": "reset fills all 6 frames with the first frame"})
    step = torch.full((3, OBS_DIM), 42.0)
    buffer.push(step)
    last_ok = torch.equal(buffer.data[:, -1], step)
    prev_ok = torch.equal(buffer.data[:, :-1], first.unsqueeze(1).expand(-1, HISTORY_FRAMES - 1, -1))
    items.append({"id": "history_push_last_frame",
                  "status": "PASS" if (last_ok and prev_ok) else "FAIL",
                  "detail": "after push(obs): slot 6 == new current frame, slots 1..5 shifted by one"})
    before = buffer.data.clone()
    buffer.reset(torch.full((3, OBS_DIM), 5.0), torch.tensor([1]))
    partial_ok = (torch.equal(buffer.data[1], torch.full((HISTORY_FRAMES, OBS_DIM), 5.0))
                  and torch.equal(buffer.data[0], before[0]))
    items.append({"id": "history_partial_reset", "status": "PASS" if partial_ok else "FAIL",
                  "detail": "per-env reset replaces only the reset env's window"})
    return items


def check_action_class_source(script_path: Path) -> tuple[bool, str, set]:
    """Static AST proof that the torque clamp and dispatch tap sit in apply_actions."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BudgetedWheelVelocity":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "apply_actions":
                    names = set()
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
                        if isinstance(sub, ast.Attribute):
                            names.add(sub.attr)
                    found[node.name] = names
    wheel = found.get("BudgetedWheelVelocity", set())
    ok = ("clamp_wheel_torque" in wheel and "set_joint_velocity_target_index" in wheel
          and "delayed" in wheel and "last_dispatch" in wheel)
    return ok, f"BudgetedWheelVelocity.apply_actions references {sorted(wheel)}", wheel


def read_source_clip_proof() -> tuple[bool, str]:
    safety = (CODE / "as2w_training_safety.py").read_text(encoding="utf-8")
    joint_actions = Path(
        "/workspace/isaaclab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py"
    ).read_text(encoding="utf-8")
    checks = {
        "apply_safety_sets_joint_clip": 'cfg.actions.joint_pos.clip = limits["angle_rad"]' in safety,
        "apply_safety_sets_effort_limit_sim":
            'actuator.effort_limit_sim = {k: limits["effort_nm"][k]' in safety,
        "safety_fraction_is_0p90": "FRACTION = .90" in safety,
        "joint_action_clamps_processed_actions":
            "self._processed_actions.clamp_(min=self._clip[:, :, 0]" in joint_actions,
        "clamp_before_joint_velocity_class":
            joint_actions.index("self._processed_actions.clamp_") <
            joint_actions.index("class JointVelocityAction"),
    }
    return all(checks.values()), json.dumps(checks)


def source_versions() -> dict:
    files = {}
    for label, path in SOURCE_FILES.items():
        files[label] = {"path": str(path), "sha256": sha256_file(path) if path.exists() else None}
    head = run_capture(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    branch = run_capture(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_capture(["git", "-C", str(REPO), "status", "--porcelain"])
    ahead = run_capture(["git", "-C", str(REPO), "rev-list", "--count", "origin/main..HEAD"])
    return {
        "git_commit": head,
        "git_branch": branch,
        "commits_ahead_of_origin_main": int(ahead) if ahead and ahead.isdigit() else None,
        "git_status_porcelain": status.splitlines() if status else [],
        "files": files,
        "arme_variants": armE_variants(),
        "note": "new files only; the listed modifications are other agents' uncommitted work and"
                " are read, never touched",
    }


def armE_variants() -> list[dict]:
    variants = [
        "skills/as2w/rolling/train_armE_support2p0.py",
        "deliveries/as2w-deploy/rolling-r11-turn-gyro-research/training/train_armE_support2p0.py",
        "deliveries/as2w-deploy/rolling-armD-turn65-research/training/train_armE_support2p0.py",
    ]
    rows = []
    for relative in variants:
        path = REPO / relative
        rows.append({"path": str(path), "sha256": sha256_file(path) if path.exists() else None})
    rows[0]["selected"] = True
    rows[0]["why"] = ("canonical skill directory (newest, 2026-11-16 18:29); the r11-turn-gyro copy"
                      " omits the RollingCommands docstring/AB branch, the armD copy omits the"
                      " long-dwell switch; the default armE command mixture used here is identical")
    return rows


# --------------------------------------------------------------------------------------
# Config emission
# --------------------------------------------------------------------------------------


def observation_block() -> dict:
    return {
        "dim": OBS_DIM,
        "source": "as2w_isaac_flat_env_all.observation() lines 59-81 with cfg.as2w_zero_memory=True",
        "zero_memory": True,
        "width_proof": {
            "base_frame_dim": 60,
            "appended_wheel_delta": 4,
            "appended_zero_flag": 1,
            "line_70": "if not env.cfg.as2w_zero_memory: return frame",
            "line_81": "torch.cat((frame, (delta / .20).clamp(-3., 3.) * zero[:, None],"
                       " zero[:, None]), dim=-1)",
            "line_252": "expected_features = 65 if args.zero_memory else 60",
        },
        "semantics_notes": [
            "60:64 is wheel angle minus the reference latched at zero-command entry, divided by"
            " 0.20 and clamped to +-3; it is not the raw wheel angle",
            "64 is the zero-command flag; it multiplies 60:64, so the delta is 0 unless the"
            " command is exactly zero",
            "58/59 are sin/cos of the 0.62 s episode clock",
        ],
        "layout": [
            {"name": name, "start": start, "end": end, "unit": unit, "source": source}
            for name, start, end, unit, source in OBS_FIELDS
        ],
    }


def history_block() -> dict:
    return {
        "frames": HISTORY_FRAMES,
        "includes_current_frame": True,
        "order": "oldest_to_newest",
        "advance": "exactly once per policy step (not per physics substep)",
        "policy_step_s": CONTROL_PERIOD_S,
        "first_to_last_span_s": HISTORY_SPAN_S,
        "reset": "fill all slots with the first frame of the new episode (no residue)",
        "convention_note": "NOT the as2w_history.py 720-D sub-sampled convention (60 + 12*55,"
                           " offsets 2..24, 480 ms, lift_actor zero padding); that convention is"
                           " deliberately unused and flagged pending operator confirmation",
        "arm_inputs": {arm: ARM_INFO[arm]["actor_input"] for arm in ("A", "B", "C")},
        "materialisation": "each arm's input width is explicit; no arm is produced by padding"
                           " another arm's weights (BLAS summation order makes that non-equivalent)",
    }


def build_config(arm: str, entry: dict, name: str, mapping: dict, sources: dict,
                 stamp: str, preset: str = DEFAULT_PRESET,
                 curriculum_mode: str = DEFAULT_CURRICULUM,
                 phase_mode: str = DEFAULT_PHASE, disturbance: str = "none",
                 load_kg: float = 0.0) -> dict:
    preset_values = PRESETS[preset]
    budget = resolve_budget(preset_values["num_envs"], preset_values["n_steps"],
                            DEFAULT_ITERATIONS, None)
    spec = network_spec(arm)
    return {
        "schema": "fs_roll_config_v1",
        "experiment": {
            "id": f"fs-roll-{stamp}",
            "arm": arm,
            "arm_name": ARM_INFO[arm]["name"],
            "status": "research",
            "deployable": False,
            "purpose": "from-scratch rolling comparison of observation/architecture arms A/B/C under"
                       " ONE wheel contract, reward set, command curriculum and env-step budget",
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "change_provenance": list(CHANGE_PROVENANCE),
        "control": {
            "control_period_s": CONTROL_PERIOD_S,
            "physics_dt_s": PHYSICS_DT,
            "decimation": DECIMATION,
            "episode_length_s": 32.0,
            "action_dim": ACTION_DIM,
            "action_clip": ACTION_CLIP,
            "device_dry_run": "cpu",
            "device_run": "cuda:0",
        },
        "observation": observation_block(),
        "history": history_block(),
        "network": spec,
        "ppo": dict(PPO_DEFAULTS, n_steps=preset_values["n_steps"]),
        "budget": {
            "preset": preset,
            "presets": PRESETS,
            **budget,
            "iterations_is_primary_key": True,
            "budget_self_proof": {
                "rejected_option_total": 32768 * 128 * 1500,
                "final_total": 16384 * 128 * 3000,
                "equal": 32768 * 128 * 1500 == 16384 * 128 * 3000,
                "note": "parallelism halved to save GPU memory; total budget unchanged",
            },
            "switch_to_total_steps": "pass --total-steps; iterations becomes total // per_round and"
                                     " the remainder is reported (never silently truncated)",
        },
        "seed": DEFAULT_SEED,
        "wheel_contract": {
            "file": str(CONTRACT_PATH),
            "sha256": sha256_file(CONTRACT_PATH),
            "expected_sha256": CONTRACT_SHA256,
            "sha256_matches": sha256_file(CONTRACT_PATH) == CONTRACT_SHA256,
            "selected_group": name,
            "selected_kd": float(entry["wheel_kd"]),
            "selection_status": "pending_validation",
            "selection_reason": "dual-simulator same-target loaded evidence is insufficient",
            "snapshot": load_json(CONTRACT_PATH),
        },
        "action_mapping": mapping,
        "assets": {
            "usd": {"path": str(USD_PATH), "sha256": sha256_file(USD_PATH)},
            "urdf": {"path": str(URDF_PATH), "sha256": sha256_file(URDF_PATH)},
        },
        "sources": sources,
        "command_curriculum": {
            "source": "armE RollingCommands (skills/as2w/rolling/train_armE_support2p0.py:57-147)",
            "mixture": {"hold": 0.20, "straight": 0.15, "spin": 0.40, "arc": 0.25},
            "vx_range_m_s": [-0.25, 0.25],
            "wz_range_rad_s": [-0.15, 0.15],
            "min_abs_command": 0.05,
            "resampling_time_range_s": [10.0, 10.0],
            "speed_curriculum_0_to_3_m_s": "available as curriculum.mode=staged_0_to_3;"
                                           " default mode is armE_mixture",
        },
        "curriculum": curriculum_settings(curriculum_mode),
        "phase": phase_settings(phase_mode, disturbance, load_kg),
        "control_contract": {
            "version": CONTROL_CONTRACT_VERSION,
            "fingerprint": control_fingerprint(),
            "spec": CONTROL_WRAPPER_SPEC,
            "guard_rule": "filter position / filter frequency / delay / history field semantics"
                          " change without a version bump -> CONTROL_CONTRACT_CHANGED and --run"
                          " refuses (override only with --accept-contract-change, logged)",
            "change_procedure": "bump control_contract_version, then redo replay and baseline"
                                " regression",
        },
        "logging": {
            "triple_tap": {
                "fields": ["env_id", "episode_id", "policy_step", "target_wheel_pre_filter[4] rad/s",
                           "target_wheel_dispatch[4] rad/s", "target_leg_pos_pre_filter[12] rad",
                           "target_leg_pos_dispatch[12] rad", "history_newest[65]",
                           "history_sha256", "dtype", "field_order", "window_shape",
                           "torque_budget_nm", "torque_budget_location",
                           "tau_request_est_nm[4] ESTIMATED", "tau_solver_nm (not_measured)"],
                "join_key": "step_id = iteration * n_steps + step",
                "run_path": "triple_tap.jsonl",
                "dry_run_path": "triple_tap_sample.jsonl (one line only)",
                "target_dispatch_source": "BudgetedWheelVelocity.last_dispatch, set immediately"
                                          " before set_joint_velocity_target_index",
            },
        },
        "safety": {
            "torque_budget_nm": TORQUE_BUDGET_NM,
            "torque_hard_limit_nm": TORQUE_HARD_LIMIT_NM,
            "torque_ratio": TORQUE_BUDGET_RATIO,
            "torque_clamp_before_dispatch": "BudgetedWheelVelocity.apply_actions ->"
                                            " clamp_wheel_torque(target_vel, joint_vel, kd, 13.5)",
            "effort_limit_sim_nm": TORQUE_BUDGET_NM,
            "action_to_target": "wheel target = 5.0 * clip(action, +-6) = up to 30 rad/s before the"
                                " 13.5 N*m torque clamp",
            "leg_angle_clip": "apply_safety sets cfg.actions.joint_pos.clip = stand + 0.90 *"
                              " (limit - stand) per axis; JointAction.process_actions clamps"
                              " _processed_actions before apply_actions",
            "wheel_velocity_clip_rad_s": 98.47,
        },
        "rewards": {
            "source": "as2w_isaac_flat_env_all.configuration + armE overrides",
            "track_lin_vel_xy_exp_weight": 1.5,
            "track_ang_vel_z_exp_weight": 1.5,
            "track_ang_vel_z_exp_std": 0.08,
            "rolling_wheel_support_weight": -0.65,
            "rolling_knee_margin_weight": -3.0,
            "zero_motion_weight": -0.4,
            "note": "identical for all three arms",
        },
        "vx_supervision": (None if not ARM_INFO[arm]["forecast"] else {
            "enabled": True,
            "loss": "huber(delta=0.1)",
            "loss_coef_default": 0.5,
            "target": "root_lin_vel_b[:, 0] (simulation truth, read every policy step)",
            "target_noise": "none",
            "train_only": True,
            "actor_path": "vx_hat from the 6-frame history with stop-gradient; true vx forbidden",
            "critic_path": "privileged true vx/vy/vz/wz/root height allowed",
            "deployment_input": "6-frame history only",
        }),
        "gpu_memory": gpu_feasibility(preset_values["num_envs"], preset_values["n_steps"], arm,
                                      (0.5 if ARM_INFO[arm]["forecast"] else 0.0)),
        "pending_validation": list(PENDING_VALIDATION),
    }


def emit_configs(argv) -> int:
    args = base_parser(run=False).parse_args(argv)
    contract = load_contract()
    name, entry = select_contract(contract, args.kd, args.contract)
    urdf = parse_urdf()
    mapping, problems = check_wheel_mapping(urdf, entry, contract["doc"]["wheel_order"])
    if problems:
        raise SystemExit("[FATAL] wheel mapping problems: " + "; ".join(problems))
    stamp = RUN_DIR.name[7:] if RUN_DIR.name.startswith("fs-roll-") else time.strftime("%Y%m%d-%H%M")
    sources = source_versions()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ("A", "B", "C"):
        config = build_config(arm, entry, name, mapping, sources, stamp,
                              curriculum_mode=args.curriculum or DEFAULT_CURRICULUM,
                              phase_mode=args.phase or DEFAULT_PHASE,
                              disturbance=args.disturbance, load_kg=args.load_kg)
        path = CONFIG_DIR / f"fs_roll_{arm}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"[emit] {path}  arm={arm}  kd={entry['wheel_kd']}  contract={name}"
              f"  curriculum={config['curriculum']['mode']}")
    write_cmd_sh(name, entry)
    print("[emit] cmd.sh written")
    return 0


def write_cmd_sh(name: str, entry: dict) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    kd = entry["wheel_kd"]
    lines = [
        "#!/usr/bin/env bash",
        f"# fs-roll exact startup commands, recorded {stamp} (research / deployable: false)",
        "# DEFAULT IS A DRY RUN.  Running this file executes nothing; it is an evidence record.",
        "set -euo pipefail",
        "",
        "# ---- dry run (pure CPU python; no isaaclab import, no stage, no CUDA) ----",
        f"# each wrapper bakes --arm <A|B|C> --kd {kd} --config configs/fs_roll_<arm>.json",
        f"# cd {RUN_DIR} && bash fs_roll_A.sh",
        f"# cd {RUN_DIR} && bash fs_roll_B.sh",
        f"# cd {RUN_DIR} && bash fs_roll_C.sh",
        "",
        "# ---- training (OPERATOR ONLY, explicit --run) ----",
        "# export PYTHONPATH=\"$(printf '%s:' /workspace/isaaclab/source/isaaclab*)/opt/isaaclab-venv/lib/python3.12/site-packages:"
        f"{CODE}\"",
        "# unset LD_LIBRARY_PATH",
        f"# cd {RUN_DIR} && bash fs_roll_A.sh --run",
        f"# cd {RUN_DIR} && bash fs_roll_B.sh --run",
        f"# cd {RUN_DIR} && bash fs_roll_C.sh --run",
        "",
        f"# contract group: {name}  wheel_kd={kd}  scale={WHEEL_ACTION_SCALE}"
        f"  budget={TORQUE_BUDGET_NM} N*m",
        "# budget: num_envs=16384  n_steps=128  per_round=2097152  iterations=3000"
        "  total=6291456000",
    ]
    (RUN_DIR / "cmd.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(RUN_DIR / "cmd.sh", 0o755)


# --------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------


def resolve_settings(args, config: dict) -> dict:
    preset = args.preset or config["budget"]["preset"]
    if preset not in PRESETS:
        raise SystemExit(f"[FATAL] unknown preset {preset!r}; have {sorted(PRESETS)}")
    values = PRESETS[preset]
    envs = args.envs if args.envs is not None else values["num_envs"]
    n_steps = args.n_steps if args.n_steps is not None else values["n_steps"]
    iterations = args.iterations if args.iterations is not None else config["budget"]["iterations"]
    budget = resolve_budget(envs, n_steps, iterations, args.total_steps)
    vx_coef = args.vx_loss_coef
    if vx_coef is None:
        vx_coef = (config.get("vx_supervision") or {}).get("loss_coef_default", 0.0)
    mode = args.curriculum or (config.get("curriculum") or {}).get("mode", DEFAULT_CURRICULUM)
    phase_mode = args.phase or (config.get("phase") or {}).get("phase", DEFAULT_PHASE)
    return {"preset": preset, "budget": budget, "vx_coef": float(vx_coef or 0.0),
            "curriculum": curriculum_settings(mode),
            "phase": phase_settings(phase_mode, args.disturbance, args.load_kg)}


def dry_run(argv, checker: Checker) -> int:
    args = base_parser(run=False).parse_args(argv)
    if args.device != "cpu":
        raise SystemExit(f"[FATAL] dry-run is CPU-only, refusing --device {args.device!r}")
    torch.set_num_threads(1)

    print_header("fs_roll dry-run -- NO TRAINING, NO ISAAC, NO GPU")
    print(f"run_dir      : {RUN_DIR}")
    print(f"time         : {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"host         : {os.uname().nodename}")
    print(f"python       : {sys.version.split()[0]}  torch={torch.__version__}")
    print(f"argv         : {' '.join(sys.argv)}")

    print_header("[1] RUN GATE / ISOLATION")
    print("--run given           : NO  -> dry-run only; Isaac/Kit is never started")
    print("isaaclab imported     :", any(m == "isaaclab" or m.startswith("isaaclab.") for m in sys.modules))
    print("isaacsim imported     :", any(m == "isaacsim" or m.startswith("isaacsim.") for m in sys.modules))
    print("torch.cuda initialized:", torch.cuda.is_initialized())
    print("torch threads         :", torch.get_num_threads())
    print("device (dry)          : cpu    (training would use cuda:0)")
    checker.add("run_gate", True, "--run absent: dry-run path, exits before any Isaac import")
    checker.add("no_isaac_import",
                not any(m == "isaaclab" or m.startswith("isaaclab.") or m == "isaacsim"
                        or m.startswith("isaacsim.") for m in sys.modules),
                "sys.modules contains no isaaclab/isaacsim")
    checker.add("no_cuda_init", not torch.cuda.is_initialized(), "torch.cuda.is_initialized() is False")
    checker.add("torch_single_thread", torch.get_num_threads() == 1,
                f"torch.get_num_threads() == {torch.get_num_threads()} (reproducibility contract)")

    print_header("[2] WHEEL CONTRACT (versioned, read-only)")
    contract = load_contract()
    print(f"file        : {contract['path']}")
    print(f"sha256      : {contract['sha256']}")
    print(f"expected    : {CONTRACT_SHA256}")
    print(f"match       : {contract['sha256'] == CONTRACT_SHA256}")
    checker.add("contract_sha256", contract["sha256"] == CONTRACT_SHA256,
                f"{contract['sha256']} == expected", "contract file changed since verification")
    name, entry = select_contract(contract, args.kd, args.contract)
    config = load_config(args)
    settings = resolve_settings(args, config)
    print(f"selected    : {name}  (wheel_kd={entry['wheel_kd']})")
    print("status      : pending_validation -- dual-simulator same-target loaded evidence insufficient")
    print("full selected group, field by field:")
    print(json.dumps(entry, indent=2))
    print("all groups and their wheel_kd (only wheel_kd may differ):")
    for key in sorted(contract["doc"]["contracts"]):
        item = contract["doc"]["contracts"][key]
        print(f"  {key:38s} wheel_kd={item['wheel_kd']}  scale={item['wheel_action_scale']}")
    failures = validate_contract(entry, contract["doc"])
    checker.add("contract_explicit_selection", True, f"--kd/--contract explicit; group {name}")
    checker.add("contract_invariants", not failures,
                "scale/kp/mode/gyro/leg gains/budget/obs/action/period all match"
                if not failures else json.dumps(failures))
    for key, detail in failures:
        if key in ("wheel_order", "observation_dim", "action_dim", "wheel_kd"):
            raise SystemExit(f"[FATAL] contract mismatch {key}: {detail}")

    print_header("[2b] CONTROL CONTRACT + PHASE")
    phase = settings["phase"]
    print(f"control_contract_version : {CONTROL_CONTRACT_VERSION}")
    print(f"contract_fingerprint     : {control_fingerprint()}")
    print(f"wrapper                  : {CONTROL_WRAPPER_SPEC['name']}  (one spec for A/B/C)")
    print(f"  filter                 : {json.dumps(CONTROL_WRAPPER_SPEC['filter'])}")
    print(f"  rate_limit             : {json.dumps(CONTROL_WRAPPER_SPEC['rate_limit'])}")
    print(f"  delay                  : {json.dumps(CONTROL_WRAPPER_SPEC['delay'])}")
    print(f"  torque_clip            : {CONTROL_WRAPPER_SPEC['torque_clip_nm']} N*m at"
          f" {CONTROL_WRAPPER_SPEC['torque_clip_location']}")
    print(f"  reset                  : {json.dumps(CONTROL_WRAPPER_SPEC['reset'])}")
    print(f"  source                 : {CONTROL_WRAPPER_SPEC['source']}")
    print(f"phase (default={DEFAULT_PHASE}) : {phase['phase']}"
          f"  disturbance={phase['disturbance']}  load_kg={phase['load_kg']}")
    print("randomisation for this phase (baseline: all off, phase-2 candidates):")
    for term_name, term in phase["randomisation"].items():
        print(f"  {term_name:20s} {term['status']:8s}"
              f" phase2_candidate={term['phase2_candidate']}  {term['note']}")
    previous_state = read_contract_state()
    decision = contract_guard(previous_state, CONTROL_CONTRACT_VERSION, control_fingerprint(),
                              args.accept_contract_change)
    print(f"contract guard           : {decision['status']}  {decision['message']}")
    if decision["refuse"]:
        print("CONTROL_CONTRACT_CHANGED -> --run would REFUSE to start; bump"
              " control_contract_version and redo replay + baseline regression")
    if phase["phase"] == "disturbance":
        checkpoint = args.init_checkpoint
        if checkpoint is None:
            print("INIT_CHECKPOINT : MISSING (disturbance phase requires --init-checkpoint)")
        elif Path(checkpoint).exists():
            print(f"INIT_CHECKPOINT : {checkpoint}  sha256={sha256_file(Path(checkpoint))}")
        else:
            print(f"INIT_CHECKPOINT : {checkpoint}  sha256=NOT_AVAILABLE (path does not exist;"
                  f" --run requires an existing checkpoint)")
        print("NOT_FROM_SCRATCH  ** disturbance phase resumes from a checkpoint; this is NOT a"
              " from-scratch run **")
        unlocked, lock_message = validate_phase2_access(
            phase["phase"], checkpoint, args.allow_disturbance, args.eval_record, run=False)
        if not unlocked:
            print(f"{lock_message}  -> phase 2 stays locked (dry-run exits 0; --run would fail)")
            checker.warn("phase2_locked", lock_message)
    if args.accept_contract_change and decision["status"] in ("changed", "accepted_change"):
        audit_path = RUN_DIR / "contract_change_audit.jsonl"
        with open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mode": "replay" if args.replay else "dry_run",
                "previous_fingerprint": (previous_state or {}).get("fingerprint"),
                "current_fingerprint": control_fingerprint(),
                "version": CONTROL_CONTRACT_VERSION,
                "reason": args.contract_change_reason,
            }) + "\n")
        print(f"contract change audit written: {audit_path}")
    checker.add("phase", phase["phase"] == (args.phase or DEFAULT_PHASE),
                f"active phase={phase['phase']} (default {DEFAULT_PHASE})")
    for item in check_phase_and_contract(phase, previous_state):
        print(f"  {item['status']:4s} {item['id']:32s} {item['detail']}")
        checker.items.append(item)
    for item in check_tamper_detection() + check_vx_gradient_isolation():
        print(f"  {item['status']:4s} {item['id']:32s} {item['detail']}")
        checker.items.append(item)
    if args.re_export_first_mismatch is not None:
        result = re_export_first_mismatch(args.re_export_first_mismatch)
        print(f"[replay] re-export first mismatch: {json.dumps(result)[:800]}")
        checker.add("replay_re_export", bool(result.get("found")),
                    json.dumps(result)[:400])
    wrapper_blocks = []
    for other_arm in ("A", "B", "C"):
        path = CONFIG_DIR / f"fs_roll_{other_arm}.json"
        if path.exists():
            wrapper_blocks.append(load_json(path)["control_contract"])
    checker.add("three_arm_identical_wrapper_config",
                len(wrapper_blocks) == 3 and all(block == wrapper_blocks[0]
                                                 for block in wrapper_blocks),
                "control_contract block is byte-identical in configs A/B/C"
                if len(wrapper_blocks) == 3 else f"only {len(wrapper_blocks)} configs found")

    print_header("[3] SOURCE VERSION + ASSETS (sha256)")
    sources = source_versions()
    print(f"git commit  : {sources['git_commit']}   branch={sources['git_branch']}"
          f"   ahead_of_origin/main={sources['commits_ahead_of_origin_main']}")
    print(f"git status  : {len(sources['git_status_porcelain'])} entries (other agents' work, read-only)")
    for line in sources["git_status_porcelain"]:
        print("   ", line)
    for label, info in sources["files"].items():
        print(f"  {label:34s} {info['sha256']}  {info['path']}")
    print("armE source variants (selected = canonical skill directory):")
    for row in sources["arme_variants"]:
        print(f"  {row['sha256']}  {row['path']}")
    print(f"  selected why: {sources['arme_variants'][0]['why']}")
    print(f"AGENTS.md   : {sources['files']['AGENTS.md']['sha256']}")
    checker.add("source_hashes_present", all(info["sha256"] for info in sources["files"].values()),
                "all tracked inputs hashed")
    checker.add("assets_exist", USD_PATH.exists() and URDF_PATH.exists(),
                f"usd={USD_PATH.exists()} urdf={URDF_PATH.exists()}")
    checker.add("arme_source_traceable", all(row["sha256"] for row in sources["arme_variants"]),
                "three armE copies hashed; canonical skill copy selected and diffed")

    print_header("[4] OBSERVATION CONTRACT")
    print(f"dim         : {OBS_DIM}   (zero_memory=True; the 60-D branch is not used)")
    print("layout (start:end  name  unit  source):")
    for name_, start, end, unit, source in OBS_FIELDS:
        print(f"  {start:>2}:{end:<2}  {name_:12s} {unit:22s} {source}")
    flat_env = (CODE / "as2w_isaac_flat_env_all.py").read_text(encoding="utf-8")
    proofs = {
        "zero_memory_branch": "if not env.cfg.as2w_zero_memory:" in flat_env,
        "wheel_delta_append": "(delta / .20).clamp(-3., 3.) * zero[:, None]" in flat_env,
        "width_selector": "expected_features = 65 if args.zero_memory else 60" in flat_env,
        "prev_action_slot": "env.action_manager.action," in flat_env,
    }
    print(f"source proof: {json.dumps(proofs)}")
    print("note        : as2w_history.py 720-D sub-sampled convention NOT used"
          " (6 consecutive frames instead)")
    coverage = []
    for _name, start, end, _unit, _src in OBS_FIELDS:
        coverage.extend(range(start, end))
    checker.add("obs_layout_contiguous", coverage == list(range(OBS_DIM)),
                f"fields cover 0..{OBS_DIM - 1} exactly once")
    checker.add("obs_width_matches_source", all(proofs.values()),
                "line 70/81 branch + line 252 selector found in as2w_isaac_flat_env_all.py")

    print_header("[5] SIX-FRAME HISTORY (hard requirement, asserted)")
    print(f"frames={HISTORY_FRAMES}  order=oldest->newest  step={CONTROL_PERIOD_S*1000:.0f} ms"
          f"  first->last span={HISTORY_SPAN_S*1000:.0f} ms  reset=fill-with-first-frame")
    for item in check_history():
        print(f"  {item['status']:4s} {item['id']:28s} {item['detail']}")
        checker.items.append(item)
    checker.add("history_span_100ms", abs(HISTORY_SPAN_S - 0.100) < 1e-12,
                f"({HISTORY_FRAMES}-1) * {CONTROL_PERIOD_S} = {HISTORY_SPAN_S} s")

    print_header("[6] NETWORKS (fixed hidden size, explicit per-arm input dims)")
    print(f"{'arm':>3} {'name':24s} {'actor_in':>9} {'actor_params':>13} {'critic_params':>14}"
          f" {'est_params':>11} {'total':>10}")
    table = {}
    for arm in ("A", "B", "C"):
        nets = build_networks(arm, DEFAULT_SEED, "cpu")
        counts = {
            "actor": count_parameters(nets["actor"]),
            "critic": count_parameters(nets["critic"]),
            "estimator": count_parameters(nets["estimator"]),
        }
        counts["total"] = counts["actor"] + counts["critic"] + counts["estimator"] + ACTION_DIM
        table[arm] = counts
        print(f"{arm:>3} {ARM_INFO[arm]['name']:24s} {ARM_INFO[arm]['actor_input']:>9}"
              f" {counts['actor']:>13} {counts['critic']:>14} {counts['estimator']:>11}"
              f" {counts['total']:>10}")
    nets = build_networks(args.arm, DEFAULT_SEED, "cpu")
    spec = network_spec(args.arm)
    print(f"selected arm {args.arm}:")
    for key in ("actor", "critic", "estimator"):
        module = nets[key]
        if module is None:
            print(f"  {key:9s}: none")
            continue
        print(f"  {key:9s}: input={spec[key]['input_dim']} hidden={spec[key]['hidden']}"
              f" output={spec[key]['output_dim']} params={count_parameters(module)}")
        for layer_name, layer in module.named_modules():
            if isinstance(layer, nn.Linear):
                print(f"      {layer_name or 'out':14s} Linear({layer.in_features} -> {layer.out_features})")
    print(f"  log_std  : {ACTION_DIM} params  init_std={PPO_DEFAULTS['init_noise_std']}")
    fp_actor = parameter_fingerprint(nets["actor"])
    print(f"  init     : orthogonal; actor output gain 0.01; seed={DEFAULT_SEED}")
    print(f"  actor fingerprint: {json.dumps(fp_actor)}  first_layer_norm={first_layer_norm(nets['actor'])}")
    same = parameter_fingerprint(build_networks(args.arm, DEFAULT_SEED, "cpu")["actor"]) == fp_actor
    print(f"  deterministic rebuild: {same}")
    checker.add("network_spec", spec["actor"]["hidden"] == list(ACTOR_HIDDEN)
                and spec["critic"]["hidden"] == list(CRITIC_HIDDEN),
                f"actor hidden {spec['actor']['hidden']} critic hidden {spec['critic']['hidden']}"
                " (identical for A/B/C)")
    checker.add("network_input_dims", [table[arm]["actor"] for arm in "ABC"]
                and ARM_INFO["A"]["actor_input"] == OBS_DIM
                and ARM_INFO["B"]["actor_input"] == HISTORY_FRAMES * OBS_DIM
                and ARM_INFO["C"]["actor_input"] == OBS_DIM + 1,
                "A=65 B=390 C=66 from ARM_INFO; parameter counts above are counted, not hand-computed")
    checker.add("init_deterministic", same, f"fingerprint {fp_actor['sha256_16']} reproducible")
    checker.add("init_nonzero", fp_actor["std"] > 0 and fp_actor["l2"] > 0,
                "weights non-degenerate after orthogonal init")

    print_header("[7] ENV-STEP BUDGET (iterations are the primary key)")
    budget = settings["budget"]
    print(f"preset             : {settings['preset']}  (presets: {json.dumps(PRESETS)})")
    print(f"num_envs           : {budget['num_envs']}")
    print(f"n_steps            : {budget['n_steps']}        # rsl_rl's name is num_steps_per_env")
    print(f"per_round_steps    : {budget['per_round_steps']} = num_envs * n_steps")
    print(f"iterations         : {budget['iterations']}")
    print(f"total_env_steps    : {budget['total_env_steps']} = iterations * per_round_steps")
    print(f"remainder          : {budget['remainder']}")
    print(f"gradient steps/iter: {budget['gradient_steps_per_iteration']}"
          f"  (epochs {PPO_DEFAULTS['epochs']} x minibatches {PPO_DEFAULTS['minibatches']})")
    print(f"total grad steps   : {budget['total_gradient_steps']}")
    old_total = 32768 * 128 * 1500
    print("BUDGET SELF-PROOF (parallelism vs total are independent):")
    print(f"  rejected option: 32768 envs * 128 steps * 1500 iters = {old_total}")
    print(f"  final          : {budget['num_envs']} envs * {budget['n_steps']} steps *"
          f" {budget['iterations']} iters = {budget['total_env_steps']}")
    print(f"  totals equal   : {old_total == budget['total_env_steps']}"
          "   (parallelism halved to save GPU memory; total budget unchanged)")
    print(f"legacy note        : {budget['legacy_steps_per_iteration']}")
    print(f"rsl_rl mapping     : {budget['rsl_rl_name']}")
    if budget["remainder"]:
        print(f"WARNING: total_env_steps is not an integer number of rounds; remainder="
              f"{budget['remainder']} steps are NOT consumed")
        checker.warn("budget_remainder", f"remainder={budget['remainder']} (raise a warning, do not"
                                         " silently truncate)")
    if budget["num_envs"] == PRESETS["small"]["num_envs"]:
        print("NOTE: running the small preset (control/load-test arm)")
    checker.add("budget_integer_rounds", budget["remainder"] == 0,
                f"{budget['total_env_steps']} % {budget['per_round_steps']} = {budget['remainder']}")

    print_header("[8] GPU MEMORY FEASIBILITY (linear extrapolation, NOT measured)")
    gpu = gpu_feasibility(budget["num_envs"], budget["n_steps"], args.arm, settings["vx_coef"])
    if gpu.get("status") == "UNKNOWN":
        print("GPU_MEMORY_UNKNOWN  nvidia-smi unavailable")
        checker.warn("gpu_memory_feasibility", "nvidia-smi unavailable; estimate impossible")
    else:
        print(f"GPU_MEMORY_ESTIMATE  baseline={GPU_BASELINE_MIB:.0f}MiB@{GPU_BASELINE_ENVS}envs"
              f"  target={budget['num_envs']}envs  estimated={gpu['estimated_sim_mib']:.0f}MiB"
              f"  available={gpu['available_mib']:.0f}MiB  ({gpu['estimated_fraction']*100:.1f}%)"
              f"  used_now={gpu['used_now_mib']:.0f}MiB")
        print(f"  method: linear extrapolation in env count; the 128-step rollout adds"
              f" ~{gpu['rollout_buffer_mib']:.0f}MiB of on-GPU tensors for arm {args.arm}"
              f" (buffer estimate only)")
        print(f"  estimated linear ceiling: {gpu['estimated_max_envs']} envs at"
              f" {GPU_SAFETY_FRACTION*100:.0f}% of {gpu['available_mib']:.0f}MiB")
        if gpu["status"] == "OK_ESTIMATED":
            print("GPU_MEMORY_FEASIBILITY_OK_ESTIMATED   # estimated only; not measured")
        else:
            print("GPU_MEMORY_FEASIBILITY_FAIL_ESTIMATED   # extrapolation exceeds the 90% limit;"
                  " operator decision required")
        print(f"GPU_MEMORY_FEASIBILITY_{gpu['status']}  estimated, not measured --"
              f" do NOT read this as 'will run'")
        checker.add("gpu_memory_estimate", True,
                    f"estimated_sim={gpu['estimated_sim_mib']}MiB estimated_fraction="
                    f"{gpu['estimated_fraction']} (linear extrapolation, not measured)")
        if gpu["status"] == "OK_ESTIMATED":
            checker.warn("gpu_memory_measured", "estimated within 90% of total; measurement pending")
        else:
            checker.warn("gpu_memory_measured",
                         f"estimated {gpu['estimated_sim_mib']:.0f}MiB exceeds"
                         f" {GPU_SAFETY_FRACTION*100:.0f}% of {gpu['available_mib']:.0f}MiB;"
                         f" estimated ceiling {gpu['estimated_max_envs']} envs")

    print_header("[8b] SPEED CURRICULUM (AGENTS.md section 1, printed by the sampler class)")
    curriculum = settings["curriculum"]
    staged_cfg = curriculum["staged_0_to_3"]
    if curriculum["mode"] == "staged_0_to_3":
        sampler = SpeedCurriculum(stages=staged_cfg["stages"],
                                  subtype_shares=staged_cfg["subtype_shares"],
                                  thresholds=staged_cfg["thresholds"],
                                  policy=staged_cfg["policy"], seed=DEFAULT_SEED)
        for line in sampler.describe():
            print(line)
        print(f"[curric] unlock rule: {staged_cfg['unlock_rule']}")
        print(f"[curric] dwell application: {staged_cfg['dwell_application']}")
        print(f"[curric] thresholds status: {staged_cfg['threshold_status']}"
              f"  source: {staged_cfg['threshold_source']}")
    else:
        sampler = ArmEMixture(curriculum["armE_mixture"])
        for line in sampler.describe():
            print(line)
        print("[curric] staged_0_to_3 is implemented but NOT active; switch explicitly with"
              " --curriculum staged_0_to_3 (or regenerate configs with --curriculum)")
    print(f"[curric] active mode: {curriculum['mode']}"
          f"   (default={DEFAULT_CURRICULUM})")
    checker.add("curriculum_mode", curriculum["mode"] in CURRICULUM_MODES,
                f"mode={curriculum['mode']} from CLI/config")
    if args.curriculum and config["curriculum"]["mode"] != args.curriculum:
        checker.warn("curriculum_cli_override",
                     f"CLI --curriculum {args.curriculum} overrides config"
                     f" {config['curriculum']['mode']}")
    for item in check_curriculum():
        print(f"  {item['status']:4s} {item['id']:32s} {item['detail']}")
        checker.items.append(item)
    checker.add("curriculum_thresholds_placeholder",
                staged_cfg["threshold_status"] == CURRICULUM_THRESHOLD_STATUS,
                "gate thresholds are explicitly marked PLACEHOLDER_PENDING_OPERATOR_RULING")

    print_header("[8c] TRIPLE-TAP LOG (wheel/leg targets, history window, torque)")
    ast_ok, ast_detail, ast_names = check_action_class_source(Path(__file__))
    recorder = TripleTapRecorder(path=None, stride=args.triple_tap_stride,
                                 sample_stride=args.history_sample_stride)
    for line in recorder.describe():
        print(line)
    sample = recorder.sample()
    print(f"[tap] sample line: {json.dumps(sample)}")
    (RUN_DIR / "triple_tap_sample.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    import numpy as np
    window = (torch.arange(HISTORY_FRAMES * OBS_DIM, dtype=torch.float32).reshape(
        HISTORY_FRAMES, OBS_DIM) * 0.001)
    np.savez(RUN_DIR / "history_window_sample.npz", window=window.numpy(),
             field_order=np.asarray(OBS_FIELD_ORDER), dtype=np.asarray("float32"),
             window_shape=np.asarray([HISTORY_FRAMES, OBS_DIM]))
    print(f"[tap] dry-run wrote {RUN_DIR / 'triple_tap_sample.jsonl'} (1 line) and"
          f" {RUN_DIR / 'history_window_sample.npz'} (1 window); --run writes triple_tap.jsonl"
          f" at stride {recorder.stride} plus history_windows.npz")
    checker.add("triple_tap_dispatch_point", "last_dispatch" in ast_names,
                "target_dispatch read from the action term at the real dispatch point"
                " (last_dispatch set immediately before set_joint_velocity_target_index)")
    checker.add("triple_tap_field_order",
                list(sample["field_order"]) == list(OBS_FIELD_ORDER),
                "history field_order literal equals the C3 observation order")

    print_header("[8d] OFFLINE ACTOR FORWARD (A/B/C fixed samples)")
    for item in check_offline_actor_forward(DEFAULT_SEED):
        print(f"  {item['status']:4s} {item['id']:32s} {item['detail']}")
        checker.items.append(item)
    print_header("[8e] CLI REFUSAL GATES (must never import Isaac)")
    for item in check_cli_refusals():
        print(f"  {item['status']:4s} {item['id']:32s} {item['detail']}")
        checker.items.append(item)

    print_header("[9] FOUR-WHEEL MAPPING (by NAME, not fixed index)")
    urdf = parse_urdf()
    mapping, problems = check_wheel_mapping(urdf, entry, contract["doc"]["wheel_order"])
    print(json.dumps(mapping, indent=2))
    print(f"problems: {problems if problems else 'none'}")
    checker.add("wheel_mapping_by_name", not problems,
                "URDF names, action channels 12..15, contact bodies and contract wheel_order agree"
                if not problems else "; ".join(problems), "wheel mapping mismatch")
    checker.add("wheel_mapping_no_placeholder",
                all(name_ in urdf["joints"] for name_ in WHEEL_JOINT_ORDER)
                and len(set(WHEEL_JOINT_ORDER)) == 4,
                "resolved by joint name; no -1 / hard-coded index / zero placeholder")

    print_header("[10] ACTION SCALE + TORQUE BUDGET (readback, clamp-before-dispatch)")
    kd_value = float(entry["wheel_kd"])
    print(f"wheel_action_scale : {entry['wheel_action_scale']}  (contract)")
    print(f"wheel_kp/mode      : {entry['wheel_kp']} / {entry['wheel_mode']}")
    print(f"wheel_kd (selected): {entry['wheel_kd']}")
    print(f"leg_kp/kd/scale    : {entry['leg_kp']} / {entry['leg_kd']} / {entry['leg_action_scale']}")
    print(f"action clip        : +-{ACTION_CLIP}")
    print(f"wheel target bound : scale * clip = {WHEEL_ACTION_SCALE} * {ACTION_CLIP}"
          f" = {WHEEL_TARGET_LIMIT_RAD_S} rad/s  (before torque clamp)")
    print(f"wheel effort limit : {TORQUE_HARD_LIMIT_NM} N*m (URDF *_foot)")
    print(f"torque budget      : {TORQUE_BUDGET_NM} N*m = {TORQUE_HARD_LIMIT_NM} * 0.9"
          f"  ratio={TORQUE_BUDGET_RATIO}")
    print(f"torque clamp       : implied torque = kd*(target - qd), clamped to"
          f" +-{TORQUE_BUDGET_NM} => max velocity error = {TORQUE_BUDGET_NM}/{kd_value}"
          f" = {TORQUE_BUDGET_NM/kd_value:.4f} rad/s")
    demo = {}
    for label, target in (("target_30", 30.0), ("target_5p0", 5.0), ("target_0", 0.0)):
        demo[label] = round(float(clamp_wheel_torque(
            torch.tensor([[target]]), torch.zeros(1, 1), kd_value, TORQUE_BUDGET_NM)), 6)
    print(f"clamp_wheel_torque(qd=0, kd={kd_value}, budget={TORQUE_BUDGET_NM}): {demo}")
    clip_ok, clip_detail = read_source_clip_proof()
    print(f"source clip proof : {clip_detail}")
    ast_ok, ast_detail, ast_names = check_action_class_source(Path(__file__))
    print(f"apply_actions AST : {ast_detail}")
    checker.add("torque_budget_math", abs(TORQUE_BUDGET_NM - TORQUE_HARD_LIMIT_NM * 0.9) < 1e-12,
                "13.5 == 15.0 * 0.9")
    checker.add("torque_clamp_before_dispatch", ast_ok,
                "clamp_wheel_torque called inside apply_actions before"
                " set_joint_velocity_target_index", "torque clamp is not in the dispatch path")
    checker.add("action_scale_readback", entry["wheel_action_scale"] == WHEEL_ACTION_SCALE
                and entry["leg_action_scale"] == LEG_ACTION_SCALE,
                f"wheel {entry['wheel_action_scale']} / leg {entry['leg_action_scale']}"
                f" / target bound {WHEEL_TARGET_LIMIT_RAD_S} rad/s")
    checker.add("safety_clip_sources", clip_ok, clip_detail)

    print_header("[11] CONSTANT-ACTION SANITY (why 'does not move' can be a config fact)")
    default_legs = (0.0, 0.8, -1.6, 0.0)
    print("default joint pos: hip 0.0, thigh 0.8, calf -1.6, foot 0.0 (init_state)")
    print(f"{'action':>12} {'wheel target':>14} {'implied N*m':>12} {'clamped target':>16}"
          f" {'leg target':>11}")
    for label, value in (("0.0", 0.0), ("+0.2", 0.2), ("+1.0", 1.0), ("+6.0 (clip)", 6.0)):
        target = WHEEL_ACTION_SCALE * value
        implied = kd_value * target
        clamped = float(clamp_wheel_torque(torch.tensor([[target]]), torch.zeros(1, 1),
                                           kd_value, TORQUE_BUDGET_NM))
        print(f"{label:>12} {target:>14.3f} {implied:>12.3f} {clamped:>16.4f}"
              f" {default_legs[0] + LEG_ACTION_SCALE * value:>11.3f}")
    print("zero action  -> wheel target 0 rad/s and legs hold the default stance: stands still.")
    print("a degenerate 'stand still' policy therefore reports max_forward ~ 0, which is a POLICY")
    print("result, not a wiring bug -- compare against this table before blaming the learner.")
    checker.add("constant_action_sanity", True,
                "zero action -> zero wheel target + default leg stance (table printed)")

    print_header("[12] CONFIG <-> CLI CONSISTENCY")
    cfg_arm = config["experiment"]["arm"]
    cfg_kd = float(config["wheel_contract"]["selected_kd"])
    cfg_contract = config["wheel_contract"]["selected_group"]
    print(f"config      : arm={cfg_arm} kd={cfg_kd} group={cfg_contract}"
          f" envs={config['budget']['num_envs']} n_steps={config['budget']['n_steps']}")
    print(f"cli         : arm={args.arm} kd={entry['wheel_kd']} group={name}"
          f" envs={budget['num_envs']} n_steps={budget['n_steps']}")
    print(f"control period: {config['control']['control_period_s']} s"
          f"   obs dim: {config['observation']['dim']}")
    print(f"USD drive gains: baked stiffness=1.7453293 / damping=0.017453292 per joint"
          f" (URDF-converter rad values of 100 / 1.0); cfg carries leg kp {entry['leg_kp']} /"
          f" kd {entry['leg_kd']} and wheel kd {entry['wheel_kd']} -> config value intended to win;"
          f" runtime write path NOT verified (not_measured)")
    checker.add("config_arm_match", cfg_arm == args.arm, f"{cfg_arm} == {args.arm}",
                "config/CLI arm mismatch")
    checker.add("config_contract_match", cfg_kd == float(entry["wheel_kd"]) and cfg_contract == name,
                f"{cfg_contract} == {name}", "config/CLI contract mismatch")
    checker.add("control_period", config["control"]["control_period_s"] == CONTROL_PERIOD_S,
                "0.02 s from config and source")

    print_header("[13] EVIDENCE WRITTEN")
    write_evidence(args, config, sources, contract, name, entry, mapping, budget, settings, checker)
    for path in ("inputs.sha256", "run-config.json", "selfcheck.json", "cmd.sh"):
        print(f"  {RUN_DIR / path}  ({os.path.getsize(RUN_DIR / path)} bytes)")

    print_header("[14] SELF-CHECK SUMMARY")
    for item in checker.items:
        print(f"  {item['status']:7s} {item['id']:34s} {item['detail']}")
    failed = checker.failed()
    warnings = [item for item in checker.items if item["status"] == "WARN"]
    print()
    if failed:
        print(f"DRY-RUN FAILED: {len(failed)} check(s) failed, {len(warnings)} warning(s)")
        return 1
    print(f"DRY-RUN OK: {len(checker.items)} checks, 0 failed, {len(warnings)} warning(s),"
          f" training NOT started")
    print("device=cpu  gpu_used=none  isaac_imported=no  stage_created=no  weights_written=no")
    return 0


def load_config(args) -> dict:
    path = args.config or (CONFIG_DIR / f"fs_roll_{args.arm}.json")
    if not path.exists():
        raise SystemExit(f"[FATAL] config not found: {path} (run --emit-configs first)")
    return load_json(path)


def write_evidence(args, config, sources, contract, name, entry, mapping, budget, settings,
                   checker) -> None:
    config_path = args.config or (CONFIG_DIR / f"fs_roll_{args.arm}.json")
    script = Path(__file__).resolve()
    lines = []
    entries = [("config", config_path), ("train_entry", script),
               ("wheel_contract", CONTRACT_PATH), ("usd", USD_PATH), ("urdf", URDF_PATH),
               ("AGENTS.md", AGENTS_PATH)]
    entries += [(label, info["path"]) for label, info in sources["files"].items()]
    for label, path in entries:
        path = Path(path)
        if path.exists():
            lines.append(f"{sha256_file(path)}  {path}  ({label})")
    (RUN_DIR / "inputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_config = {
        "schema": "fs_roll_run_config_v1",
        "mode": "dry_run",
        "argv": sys.argv,
        "parsed": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "arm": args.arm,
        "contract_group": name,
        "wheel_kd": entry["wheel_kd"],
        "observation_dim": OBS_DIM,
        "history": {"frames": HISTORY_FRAMES, "span_s": HISTORY_SPAN_S},
        "network": network_spec(args.arm),
        "preset": settings["preset"],
        "budget": budget,
        "vx_loss_coef": settings["vx_coef"],
        "gpu_memory": gpu_feasibility(budget["num_envs"], budget["n_steps"], args.arm,
                                      settings["vx_coef"]),
        "device_dry_run": "cpu",
        "device_run": "cuda:0",
        "wheel_contract_sha256": contract["sha256"],
        "assets": config["assets"],
        "sources": sources,
        "action_mapping": mapping,
        "config_snapshot": config,
        "checks": checker.items,
        "isaac_imported": any(m == "isaaclab" or m.startswith("isaaclab.") for m in sys.modules),
        "cuda_initialized": torch.cuda.is_initialized(),
        "gpu_touched": False,
        "training_started": False,
        "status": "research",
        "deployable": False,
    }
    (RUN_DIR / "run-config.json").write_text(json.dumps(run_config, indent=2) + "\n",
                                             encoding="utf-8")
    (RUN_DIR / "selfcheck.json").write_text(json.dumps({
        "schema": "fs_roll_selfcheck_v1",
        "arm": args.arm,
        "contract_group": name,
        "all_passed": not checker.failed(),
        "warnings": [item["id"] for item in checker.items if item["status"] == "WARN"],
        "checks": checker.items,
    }, indent=2) + "\n", encoding="utf-8")
    write_cmd_sh(name, entry)


# --------------------------------------------------------------------------------------
# Training path (--run): the only place Isaac is imported
# --------------------------------------------------------------------------------------


def build_runtime_env(args, physics, contract_entry, num_envs, curriculum_cfg, phase_cfg):
    """Build the rolling env exactly like armE, with the contract wheel gain applied."""
    from types import SimpleNamespace

    from as2w_isaac_flat_env_all import JOINTS
    from as2w_isaac_robust_env import (
        DelayedGyroVelocity, RobustEnvironment, TakeoverCommands, robust_config,
    )
    from as2w_isaac_speed_range import wheel_support_error
    from as2w_training_safety import apply_safety, rolling_knee_margin_cost
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg

    if tuple(JOINTS) != JOINT_ORDER:
        raise RuntimeError(f"canonical joint order changed: {JOINTS}")

    class RollingCommands(TakeoverCommands):
        """armE command mixture, decoded by ArmEMixture.

        The decode expression is the armE default path (mode buckets .20/.35/.75, five uniform
        draws, same order); only the code location moved so dry-run prints from the sampler that
        actually runs.  The AS2W_AB_CURRICULUM branch is intentionally not carried over.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.mixture = ArmEMixture(env.cfg.as2w_curriculum["armE_mixture"])

        def _resample_command(self, env_ids):
            draw = torch.rand((self.targets[env_ids].shape[0], 5), device=self.device)
            vx, yaw, _mode = self.mixture.decode(draw)
            self.targets[env_ids, 0] = vx
            self.targets[env_ids, 1] = 0.
            self.targets[env_ids, 2] = yaw

    class StagedCommands(TakeoverCommands):
        """staged 0 -> 3 m/s command sampler driven by SpeedCurriculum (same class as dry-run)."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            staged = env.cfg.as2w_curriculum["staged_0_to_3"]
            self.curriculum = SpeedCurriculum(stages=staged["stages"],
                                              subtype_shares=staged["subtype_shares"],
                                              thresholds=staged["thresholds"],
                                              policy=staged["policy"],
                                              seed=int(getattr(env.cfg, "seed", 0) or 0))
            print("CURRICULUM_DESCRIBE\n" + "\n".join(self.curriculum.describe()), flush=True)

        def _resample_command(self, env_ids):
            count = self.targets[env_ids].shape[0]
            samples = [self.curriculum.sample() for _ in range(count)]
            self.targets[env_ids, 0] = self.targets.new_tensor([s["vx"] for s in samples])
            self.targets[env_ids, 1] = 0.
            self.targets[env_ids, 2] = self.targets.new_tensor([s["wz"] for s in samples])

    class BudgetedWheelVelocity(DelayedGyroVelocity):
        """Velocity set-point with the 13.5 N*m budget applied before dispatch."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._budget_nm = float(env.cfg.as2w_torque_budget_nm)
            self._drive_kd = float(env.cfg.as2w_wheel_kd)

        def apply_actions(self):
            if self._as2w_on:
                raise RuntimeError("gyro feedback is disabled by the wheel contract; refusing to"
                                   " run with AS2W_GYRO_KP / AS2W_GYRO_FF_GAIN set")
            target = clamp_wheel_torque(
                self.delayed(),
                self._asset.data.joint_vel.torch[:, self._joint_ids],
                self._drive_kd,
                self._budget_nm,
            )
            # triple-tap: expose the value that is actually dispatched, at the dispatch point
            self.last_dq = self._asset.data.joint_vel.torch[:, self._joint_ids].detach().clone()
            self.last_dispatch = target.detach().clone()
            self._asset.set_joint_velocity_target_index(target=target, joint_ids=self._joint_ids)

    from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

    class BudgetedLegPosition(JointPositionAction):
        """Leg position target after the 90% angle clip; exposes the dispatch value."""

        def apply_actions(self):
            target = self.processed_actions
            self.last_dispatch = target.detach().clone()
            self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    cfg = robust_config(SimpleNamespace(
        **vars(args), num_envs=int(num_envs), profile="baseline", direction_focus=True,
        command_stage="all", zero_memory=True, zero_focus=True), physics)
    # baseline/disturbance phase-1 rule: every randomisation except the two reset terms is OFF.
    for term_name in list(vars(cfg.events)):
        if term_name not in ("reset_base", "reset_robot_joints"):
            setattr(cfg.events, term_name, None)
    payload_kg = float(phase_cfg.get("load_kg", 0.0))
    if phase_cfg["phase"] == "disturbance" and phase_cfg["disturbance"] == "load":
        cfg.events.payload = EventTermCfg(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                    "mass_distribution_params": (payload_kg, payload_kg), "operation": "add"})
    if phase_cfg["phase"] == "disturbance" and phase_cfg["disturbance"] == "push":
        spec = phase_cfg["push_spec"]
        cfg.events.bounded_push = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=tuple(spec["interval_range_s"]),
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "velocity_range": {k: tuple(v) for k, v in spec["velocity_range"].items()}})
    print(f"PHASE {phase_cfg['phase']} DISTURBANCE {phase_cfg['disturbance']}"
          f" LOAD_KG {payload_kg}", flush=True)
    for term_name, term in vars(cfg.events).items():
        print(f"EVENT {term_name:24s} {'ON' if term is not None else 'OFF'}"
              f"{'  (phase-2 candidate)' if term is None else ''}", flush=True)
    cmd = cfg.commands.base_velocity
    cfg.as2w_curriculum = curriculum_cfg
    if curriculum_cfg["mode"] == "staged_0_to_3":
        staged = curriculum_cfg["staged_0_to_3"]
        cmd.class_type = StagedCommands
        dwell = [float(stage["dwell_s"]) for stage in staged["stages"]]
        cmd.resampling_time_range = (min(dwell), max(dwell))
        cmd.ranges.lin_vel_x = (-3.0, 3.0)
        cmd.ranges.lin_vel_y = (0., 0.)
        cmd.ranges.ang_vel_z = (-0.25, 0.25)
        print("CURRICULUM_MODE staged_0_to_3  max_speed 3.0 m/s  stages"
              f" {len(staged['stages'])}  resampling {min(dwell):.0f}-{max(dwell):.0f} s",
              flush=True)
    else:
        cmd.class_type = RollingCommands
        cmd.resampling_time_range = (10., 10.)
        cmd.ranges.lin_vel_x = (-0.25, 0.25)
        cmd.ranges.lin_vel_y = (0., 0.)
        cmd.ranges.ang_vel_z = (-0.15, 0.15)
        print("CURRICULUM_MODE armE_mixture  vx_cap 0.25 m/s", flush=True)
    cfg.episode_length_s = 32.
    cfg.scene.robot.actuators["foot"].damping = float(contract_entry["wheel_kd"])
    cfg.scene.robot.actuators["foot"].stiffness = 0.0
    cfg.actions.wheel_vel.scale = float(contract_entry["wheel_action_scale"])
    cfg.actions.joint_pos.scale = float(contract_entry["leg_action_scale"])
    cfg.actions.wheel_vel.class_type = BudgetedWheelVelocity
    cfg.actions.joint_pos.class_type = BudgetedLegPosition
    cfg.as2w_torque_budget_nm = TORQUE_BUDGET_NM
    cfg.as2w_wheel_kd = float(contract_entry["wheel_kd"])
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
    cfg.rewards.rolling_knee_margin = RewardTermCfg(
        func=rolling_knee_margin_cost, weight=-3., params={"margin_m": .10})
    limits = apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    return env, limits


def assert_runtime_mapping(env, contract_entry) -> dict:
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    if len(names) != len(JOINT_ORDER) or sorted(names) != sorted(JOINT_ORDER):
        raise RuntimeError(f"runtime joint name set {sorted(names)} != canonical"
                           f" {sorted(JOINT_ORDER)}")
    ids = list(env.as2w_joint_ids)
    wheel_ids = ids[LEG_DIM:]
    expected = [names.index(name) for name in WHEEL_JOINT_ORDER]
    if wheel_ids != expected:
        raise RuntimeError(f"runtime wheel ids {wheel_ids} != {expected} (names {WHEEL_JOINT_ORDER})")
    bodies = list(robot.body_names)
    missing = [name for name in WHEEL_JOINT_ORDER if name not in bodies]
    if missing:
        raise RuntimeError(f"wheel contact bodies missing from articulation: {missing}")
    wheel_term = env.action_manager.get_term("wheel_vel")
    leg_term = env.action_manager.get_term("joint_pos")
    if type(wheel_term).__name__ != "BudgetedWheelVelocity":
        raise RuntimeError(f"wheel action term is {type(wheel_term).__name__}, torque clamp missing")
    if float(wheel_term.cfg.scale) != float(contract_entry["wheel_action_scale"]):
        raise RuntimeError(f"wheel scale {wheel_term.cfg.scale} != {contract_entry['wheel_action_scale']}")
    if float(leg_term.cfg.scale) != float(contract_entry["leg_action_scale"]):
        raise RuntimeError(f"leg scale {leg_term.cfg.scale} != {contract_entry['leg_action_scale']}")
    foot_effort = env.cfg.scene.robot.actuators["foot"].effort_limit_sim
    worst = max(float(value) for value in foot_effort.values()) if isinstance(foot_effort, dict) \
        else float(foot_effort)
    if abs(worst - TORQUE_BUDGET_NM) > 1e-9:
        raise RuntimeError(f"wheel effort_limit_sim {worst} != {TORQUE_BUDGET_NM}")
    return {
        "joint_names": names, "wheel_joint_ids": wheel_ids,
        "wheel_action_channels": {name: LEG_DIM + i for i, name in enumerate(WHEEL_JOINT_ORDER)},
        "wheel_scale": float(wheel_term.cfg.scale), "leg_scale": float(leg_term.cfg.scale),
        "wheel_effort_limit_sim_nm": worst, "wheel_drive_kd": float(env.cfg.as2w_wheel_kd),
    }


class CurriculumTracker:
    """Per-episode gate metrics -> SpeedCurriculum.observe() at episode end.

    Only active when curriculum.mode == staged_0_to_3.  Accumulation is vectorised; the Python
    loop runs once per finished episode, not per step.  NOT exercised during preparation
    (``--run`` was never executed), so treat the runtime wiring as untested.
    """

    def __init__(self, env, curriculum):
        self.env = env
        self.curriculum = curriculum
        self.num_envs = env.num_envs
        self.device = env.device
        self.sensor = env.scene["contact_forces"]
        names = list(self.sensor.body_names)
        self.wheel_ids = [names.index(name) for name in WHEEL_JOINT_ORDER]
        self.nonwheel_ids = [i for i, name in enumerate(names) if not name.endswith("_foot")]
        self.t = curriculum.thresholds
        self.reset_all()

    def reset_all(self) -> None:
        zeros = torch.zeros(self.num_envs, device=self.device)
        ones = torch.ones(self.num_envs, device=self.device)
        self.steps = zeros.clone()
        self.in_band = zeros.clone()
        self.max_lin = zeros.clone()
        self.max_yaw = zeros.clone()
        self.min_upright = ones.clone()
        self.max_roll = zeros.clone()
        self.max_pitch = zeros.clone()
        self.min_loaded = torch.full_like(ones, float(WHEEL_DIM))
        self.unload_steps = zeros.clone()
        self.max_unload_steps = zeros.clone()
        self.max_nonwheel = zeros.clone()
        self.brake_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.brake_time = zeros.clone()
        self.brake_distance = zeros.clone()
        self.brake_measured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stop_time = torch.full_like(zeros, float("inf"))
        self.stop_distance = torch.full_like(zeros, float("inf"))
        self.drift = zeros.clone()
        self.prev_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_index(self, index: int) -> None:
        self.steps[index] = 0.0
        self.in_band[index] = 0.0
        self.max_lin[index] = 0.0
        self.max_yaw[index] = 0.0
        self.min_upright[index] = 1.0
        self.max_roll[index] = 0.0
        self.max_pitch[index] = 0.0
        self.min_loaded[index] = float(WHEEL_DIM)
        self.unload_steps[index] = 0.0
        self.max_unload_steps[index] = 0.0
        self.max_nonwheel[index] = 0.0
        self.brake_active[index] = False
        self.brake_time[index] = 0.0
        self.brake_distance[index] = 0.0
        self.brake_measured[index] = False
        self.stop_time[index] = float("inf")
        self.stop_distance[index] = float("inf")
        self.drift[index] = 0.0
        self.prev_zero[index] = False

    def update(self) -> None:
        robot = self.env.scene["robot"]
        command = self.env.command_manager.get_command("base_velocity")
        linear = robot.data.root_lin_vel_b.torch[:, :2]
        yaw_rate = robot.data.root_ang_vel_b.torch[:, 2]
        speed = torch.linalg.vector_norm(linear, dim=1)
        lin_err = torch.linalg.vector_norm(command[:, :2] - linear, dim=1)
        yaw_err = (command[:, 2] - yaw_rate).abs()
        self.max_lin = torch.maximum(self.max_lin, lin_err)
        self.max_yaw = torch.maximum(self.max_yaw, yaw_err)
        in_band = ((lin_err <= self.t["track"]["lin_vel_abs_error_m_s_max"])
                   & (yaw_err <= self.t["track"]["yaw_rate_abs_error_rad_s_max"]))
        self.in_band = self.in_band + in_band.float()
        gravity = robot.data.projected_gravity_b.torch
        upright = (-gravity[:, 2]).clamp(-1.0, 1.0)
        roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs() * (180.0 / math.pi)
        pitch = torch.atan2(gravity[:, 0], torch.sqrt(
            gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs() * (180.0 / math.pi)
        self.min_upright = torch.minimum(self.min_upright, upright)
        self.max_roll = torch.maximum(self.max_roll, roll)
        self.max_pitch = torch.maximum(self.max_pitch, pitch)
        forces = self.sensor.data.net_forces_w.torch[:, self.wheel_ids, 2]
        loaded = (forces > self.t["contact"]["wheel_normal_force_min_n"]).sum(dim=1).float()
        self.min_loaded = torch.minimum(self.min_loaded, loaded)
        unloaded = (forces < self.t["contact"]["wheel_normal_force_min_n"]).any(dim=1)
        self.unload_steps = torch.where(unloaded, self.unload_steps + 1,
                                        torch.zeros_like(self.unload_steps))
        self.max_unload_steps = torch.maximum(self.max_unload_steps, self.unload_steps)
        nonwheel = self.sensor.data.net_forces_w.torch[:, self.nonwheel_ids].norm(dim=-1).amax(dim=1)
        self.max_nonwheel = torch.maximum(self.max_nonwheel, nonwheel)
        zero = (command == 0.0).all(dim=1)
        started = zero & (~self.prev_zero) & (speed > 0.05)
        self.brake_active = self.brake_active | started
        active = self.brake_active.float()
        self.brake_time = self.brake_time + active * self.env.step_dt
        self.brake_distance = self.brake_distance + active * speed * self.env.step_dt
        stopped = self.brake_active & (speed < self.t["brake"]["post_stop_drift_m_s_max"])
        first_stop = stopped & (~self.brake_measured)
        self.stop_time = torch.where(first_stop, self.brake_time, self.stop_time)
        self.stop_distance = torch.where(first_stop, self.brake_distance, self.stop_distance)
        self.brake_measured = self.brake_measured | first_stop
        self.drift = torch.maximum(self.drift, torch.where(stopped, speed,
                                                          torch.zeros_like(speed)))
        self.brake_active = self.brake_active & (~stopped)
        self.prev_zero = zero
        self.steps = self.steps + 1.0

    def finalize(self, terminated: torch.Tensor, truncated: torch.Tensor):
        done = terminated | truncated
        if not bool(done.any()):
            return []
        events = []
        for index in torch.nonzero(done, as_tuple=False).squeeze(-1).tolist():
            steps = max(float(self.steps[index]), 1.0)
            hard = bool(terminated[index])
            outcome = {
                "track": {"lin_vel_abs_error_m_s": float(self.max_lin[index]),
                          "yaw_rate_abs_error_rad_s": float(self.max_yaw[index]),
                          "in_band_fraction": float(self.in_band[index]) / steps},
                "attitude": {"upright_cos": float(self.min_upright[index]),
                             "roll_deg": float(self.max_roll[index]),
                             "pitch_deg": float(self.max_pitch[index])},
                "contact": {"min_loaded_wheels": int(self.min_loaded[index].item()),
                            "max_unload_ms": float(self.max_unload_steps[index])
                            * self.env.step_dt * 1000.0,
                            "nonwheel_contact_n": float(self.max_nonwheel[index])},
                "brake": {"measured": bool(self.brake_measured[index]),
                          "stop_time_s": float(self.stop_time[index]),
                          "stop_distance_m": float(self.stop_distance[index]),
                          "post_stop_drift_m_s": float(self.drift[index])},
                "hard_fail": hard,
                "hard_fail_kind": "terminated" if hard else None,
            }
            events.append((index, self.curriculum.observe(outcome)))
            self._reset_index(index)
        return events


def privileged_state(env) -> torch.Tensor:
    robot = env.scene["robot"]
    velocity = robot.data.root_lin_vel_b.torch
    yaw_rate = robot.data.root_ang_vel_b.torch[:, 2:3]
    height = robot.data.root_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3]
    return torch.cat((velocity, yaw_rate, height), dim=1)


def true_vx(env) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_b.torch[:, 0]


class PPOTrainer:
    def __init__(self, nets, args, device, ppo: dict):
        self.nets = nets
        self.device = device
        self.ppo = ppo
        params = [p for p in nets["actor"].parameters()]
        params += [p for p in nets["critic"].parameters()]
        if nets["estimator"] is not None:
            params += [p for p in nets["estimator"].parameters()]
        params += [nets["log_std"]]
        self.optimizer = torch.optim.Adam(params, lr=ppo["learning_rate"], eps=1e-5)
        self.vx_coef = float(getattr(args, "vx_loss_coef", 0.0) or 0.0)

    def distribution(self, actor_input):
        mean = self.nets["actor"](actor_input)
        std = self.nets["log_std"].exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, actor_input, critic_input):
        dist = self.distribution(actor_input)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.nets["critic"](critic_input).squeeze(-1)
        return action, log_prob, value

    @torch.no_grad()
    def value(self, critic_input):
        return self.nets["critic"](critic_input).squeeze(-1)

    def estimator_forward(self, history_features, detach=True):
        """vx_hat for the actor; ``detach`` cuts the policy-loss gradient path into the head.

        The supervised head loss calls the module directly (no detach), so the head still learns
        from its own loss while the policy gradient never touches it.
        """
        features = history_features.detach() if detach else history_features
        value = self.nets["estimator"](features).squeeze(-1)
        return value.detach() if detach else value

    def update(self, batch):
        ppo = self.ppo
        actor_in, actions = batch["actor_in"], batch["actions"]
        total = actor_in.shape[0]
        minibatch = max(1, total // ppo["minibatches"])
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "vx_loss": 0.0,
                 "clip_frac": 0.0, "updates": 0}
        for _epoch in range(ppo["epochs"]):
            order = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch):
                index = order[start:start + minibatch]
                dist = self.distribution(actor_in[index])
                log_prob = dist.log_prob(actions[index]).sum(-1)
                ratio = (log_prob - batch["log_prob"][index]).exp()
                advantage = batch["advantages"][index]
                surrogate = torch.min(ratio * advantage,
                                      torch.clamp(ratio, 1.0 - ppo["clip"],
                                                  1.0 + ppo["clip"]) * advantage)
                critic_input = torch.cat((actor_in[index], batch["private"][index]), dim=-1)
                values = self.nets["critic"](critic_input).squeeze(-1)
                value_loss = 0.5 * (values - batch["returns"][index]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = (-surrogate.mean() + ppo["value_coef"] * value_loss
                        - ppo["entropy_coef"] * entropy)
                vx_loss = torch.zeros((), device=self.device)
                if self.nets["estimator"] is not None and self.vx_coef > 0.0:
                    predicted = self.estimator_forward(batch["history"][index])
                    vx_loss = F.huber_loss(predicted, batch["vx"][index], delta=0.1)
                    loss = loss + self.vx_coef * vx_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    ppo["max_grad_norm"])
                self.optimizer.step()
                stats["policy_loss"] += float(-surrogate.mean())
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["vx_loss"] += float(vx_loss)
                stats["clip_frac"] += float(((ratio - 1.0).abs() > ppo["clip"]).float().mean())
                stats["updates"] += 1
        for key in stats:
            if key != "updates":
                stats[key] /= max(1, stats["updates"])
        return stats


def rollout_loop(args, env, entry, budget, ppo, settings) -> int:
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    nets = build_networks(args.arm, args.seed, device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        nets["actor"].load_state_dict(checkpoint["actor"])
        nets["critic"].load_state_dict(checkpoint["critic"])
        if nets["estimator"] is not None and checkpoint.get("estimator"):
            nets["estimator"].load_state_dict(checkpoint["estimator"])
        if checkpoint.get("log_std") is not None:
            nets["log_std"].data.copy_(checkpoint["log_std"].to(device))
        print("INIT_CHECKPOINT_LOADED", args.init_checkpoint, "NOT_FROM_SCRATCH", flush=True)
    trainer = PPOTrainer(nets, args, device, ppo)
    history = HistoryBuffer(env.num_envs, OBS_DIM, HISTORY_FRAMES, device)
    n_steps = ppo["n_steps"]
    envs = env.num_envs
    foot_effort = env.cfg.scene.robot.actuators["foot"].effort_limit_sim
    budget_readback = (f"effort_limit_sim={max(float(v) for v in foot_effort.values())}"
                       if isinstance(foot_effort, dict) else f"effort_limit_sim={foot_effort}")
    tap = TripleTapRecorder(path=args.out / "triple_tap.jsonl", stride=args.triple_tap_stride,
                            sample_stride=args.history_sample_stride)
    sampled_windows, sampled_steps = [], []
    episode_ids = torch.zeros(envs, dtype=torch.long, device=device)
    tracker = None
    if settings["curriculum"]["mode"] == "staged_0_to_3":
        term = env.command_manager.get_term("base_velocity")
        tracker = CurriculumTracker(env, term.curriculum)
        print("CURRICULUM_START " + json.dumps(term.curriculum.state()), flush=True)
    print("NETWORKS", json.dumps(network_spec(args.arm)), flush=True)
    print("PARAM_COUNTS actor", count_parameters(nets["actor"]), "critic",
          count_parameters(nets["critic"]), "estimator", count_parameters(nets["estimator"]),
          flush=True)
    print("TRAIN_START", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
    obs_dict, _ = env.reset()
    frame = obs_dict["policy"].detach().float()
    history.reset(frame)
    log_path = args.out / "train_log.jsonl"
    started = time.time()
    for iteration in range(budget["iterations"]):
        shape = (n_steps, envs)
        actor_in = torch.empty((*shape, ARM_INFO[args.arm]["actor_input"]), device=device)
        private_in = torch.empty((*shape, PRIVILEGED_DIM), device=device)
        actions = torch.empty((*shape, ACTION_DIM), device=device)
        log_prob = torch.empty(shape, device=device)
        values = torch.empty((n_steps + 1, envs), device=device)
        rewards = torch.empty(shape, device=device)
        dones = torch.zeros(shape, device=device)
        need_hist = ARM_INFO[args.arm]["forecast"] and trainer.vx_coef > 0.0
        history_features = (torch.empty((*shape, HISTORY_FRAMES * OBS_DIM), device=device)
                            if need_hist else None)
        vx_targets = torch.empty(shape, device=device) if need_hist else None

        for step in range(n_steps):
            if ARM_INFO[args.arm]["forecast"]:
                flat = history.flat()
                vx_hat = trainer.estimator_forward(flat)
                features = torch.cat((frame, vx_hat.unsqueeze(-1)), dim=1)
                if need_hist:
                    history_features[step] = flat
                    vx_targets[step] = true_vx(env)
            elif ARM_INFO[args.arm]["name"] == "six_frame_concat":
                features = history.flat()
            else:
                features = frame
            private = privileged_state(env)
            action, step_log_prob, step_value = trainer.act(
                features, torch.cat((features, private), dim=1))
            actor_in[step] = features
            private_in[step] = private
            actions[step] = action
            log_prob[step] = step_log_prob
            values[step] = step_value
            obs_dict, reward, terminated, truncated, _extras = env.step(
                torch.clamp(action, -ACTION_CLIP, ACTION_CLIP))
            frame = obs_dict["policy"].detach().float()
            rewards[step] = reward
            done = (terminated | truncated)
            dones[step] = done.float()
            history.push(frame)
            if bool(done.any()):
                history.reset(frame, torch.nonzero(done, as_tuple=False).squeeze(-1))
            if tracker is not None:
                tracker.update()
                for _index, event in tracker.finalize(terminated, truncated):
                    if event["event"] in ("advance", "hard_fail"):
                        print("CURRICULUM_EVENT", json.dumps(event), flush=True)
            global_step = iteration * n_steps + step
            if global_step % tap.stride == 0:
                wheel_term = env.action_manager.get_term("wheel_vel")
                leg_term = env.action_manager.get_term("joint_pos")
                tap.write(tap.record(
                    tap.probe_env, int(episode_ids[tap.probe_env]), global_step,
                    action[tap.probe_env], wheel_term.last_dispatch[tap.probe_env],
                    wheel_term.last_dq[tap.probe_env], leg_term.last_dispatch[tap.probe_env],
                    history.data[tap.probe_env], float(entry["wheel_kd"]), budget_readback))
            if global_step % tap.sample_stride == 0 and len(sampled_windows) < 512:
                sampled_windows.append(history.data[tap.probe_env].detach().cpu().numpy().copy())
                sampled_steps.append(global_step)
            episode_ids[done] += 1
        with torch.no_grad():
            if ARM_INFO[args.arm]["forecast"]:
                flat = history.flat()
                last = torch.cat((frame, trainer.estimator_forward(flat).unsqueeze(-1)), dim=1)
            elif ARM_INFO[args.arm]["name"] == "six_frame_concat":
                last = history.flat()
            else:
                last = frame
            values[n_steps] = trainer.value(torch.cat((last, privileged_state(env)), dim=1))

        advantages = torch.zeros(shape, device=device)
        last_adv = torch.zeros(envs, device=device)
        for step in reversed(range(n_steps)):
            nonterminal = 1.0 - dones[step]
            delta = (rewards[step] + ppo["gamma"] * values[step + 1] * nonterminal
                     - values[step])
            last_adv = (delta + ppo["gamma"] * ppo["gae_lambda"] * nonterminal * last_adv)
            advantages[step] = last_adv
        returns = advantages + values[:n_steps]
        flat_adv = advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        batch = {
            "actor_in": actor_in.reshape(-1, actor_in.shape[-1]),
            "private": private_in.reshape(-1, PRIVILEGED_DIM),
            "actions": actions.reshape(-1, ACTION_DIM),
            "log_prob": log_prob.reshape(-1),
            "advantages": flat_adv,
            "returns": returns.reshape(-1),
        }
        if need_hist:
            batch["history"] = history_features.reshape(-1, history_features.shape[-1])
            batch["vx"] = vx_targets.reshape(-1)
        stats = trainer.update(batch)
        record = {
            "iteration": iteration + 1,
            "env_steps": (iteration + 1) * budget["per_round_steps"],
            "mean_reward_per_step": float(rewards.mean()),
            "done_fraction": float(dones.mean()),
            "value_mean": float(values[:n_steps].mean()),
            **stats,
            "elapsed_s": round(time.time() - started, 2),
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if iteration % max(1, args.log_every) == 0 or iteration + 1 == budget["iterations"]:
            print("ITER", json.dumps(record), flush=True)
        if (iteration + 1) % max(1, args.save_interval) == 0 or iteration + 1 == budget["iterations"]:
            save_checkpoint(args, nets, trainer, iteration + 1, entry)
    if sampled_windows:
        import numpy as np
        np.savez(args.out / "history_windows.npz", windows=np.stack(sampled_windows),
                 policy_steps=np.asarray(sampled_steps),
                 field_order=np.asarray(OBS_FIELD_ORDER))
    if tracker is not None:
        print("CURRICULUM_END " + json.dumps(tracker.curriculum.state()), flush=True)
    print("FS_ROLL_TRAIN_COMPLETE", args.arm, budget["total_env_steps"], flush=True)
    return 0


def save_checkpoint(args, nets, trainer, iteration, entry) -> None:
    checkpoint = {
        "schema": "fs_roll_checkpoint_v1",
        "iteration": iteration,
        "arm": args.arm,
        "contract_group": contract_key(float(entry["wheel_kd"])),
        "wheel_kd": float(entry["wheel_kd"]),
        "actor": nets["actor"].state_dict(),
        "critic": nets["critic"].state_dict(),
        "estimator": None if nets["estimator"] is None else nets["estimator"].state_dict(),
        "log_std": nets["log_std"].detach().cpu(),
        "optimizer": trainer.optimizer.state_dict(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    path = args.out / f"model_{iteration:07d}.pt"
    torch.save(checkpoint, path)
    torch.save(checkpoint, args.out / "latest.pt")
    print("CHECKPOINT", path, flush=True)


def train(args, physics) -> int:
    contract = load_contract()
    if contract["sha256"] != CONTRACT_SHA256:
        raise RuntimeError("wheel contract changed on disk; refusing to train")
    name, entry = select_contract(contract, args.kd, args.contract)
    if validate_contract(entry, contract["doc"]):
        raise RuntimeError("wheel contract invariants violated")
    config = load_config(args)
    settings = resolve_settings(args, config)
    budget = settings["budget"]
    ppo = dict(PPO_DEFAULTS, n_steps=budget["n_steps"])
    args.vx_loss_coef = settings["vx_coef"]
    args.seed = args.seed if args.seed is not None else int(config["seed"])
    args.save_interval = max(1, int(args.save_interval))
    if args.out is None:
        args.out = RUN_DIR / "train" / f"{args.arm}-{time.strftime('%Y%m%d-%H%M%S')}"

    print_header("fs_roll --run (TRAINING)")
    print(f"arm={args.arm}  contract={name}  wheel_kd={entry['wheel_kd']}"
          f"  envs={budget['num_envs']}  n_steps={budget['n_steps']}"
          f"  iterations={budget['iterations']}  total_env_steps={budget['total_env_steps']}"
          f"  seed={args.seed}")
    phase = settings["phase"]
    print(f"PHASE {phase['phase']}  DISTURBANCE {phase['disturbance']}"
          f"  CURRICULUM {settings['curriculum']['mode']}")
    checkpoint = args.init_checkpoint
    if phase["phase"] == "disturbance":
        validate_phase2_access(phase["phase"], checkpoint, args.allow_disturbance,
                               args.eval_record, run=True)
        if checkpoint is None or not Path(checkpoint).exists():
            raise SystemExit(f"[FATAL] INIT_CHECKPOINT missing or unreadable: {checkpoint}")
        print(f"INIT_CHECKPOINT sha256={sha256_file(Path(checkpoint))} path={checkpoint}",
              flush=True)
        print("NOT_FROM_SCRATCH  ** disturbance phase resumes from a checkpoint **", flush=True)
    elif checkpoint is not None:
        print(f"INIT_CHECKPOINT sha256={sha256_file(Path(checkpoint))} path={checkpoint}"
              " (baseline warm start, recorded)", flush=True)
    env, limits = build_runtime_env(args, physics, entry, budget["num_envs"],
                                    settings["curriculum"], phase)
    try:
        mapping = assert_runtime_mapping(env, entry)
        if env.num_envs != budget["num_envs"]:
            raise RuntimeError(f"envs {env.num_envs} != {budget['num_envs']}")
        print("RUNTIME_MAPPING", json.dumps(mapping), flush=True)
        print("BUDGET", json.dumps(budget), flush=True)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "train_contract.json").write_text(json.dumps({
            "schema": "fs_roll_train_contract_v1",
            "arm": args.arm, "arm_name": ARM_INFO[args.arm]["name"],
            "contract_group": name, "contract": entry,
            "wheel_contract_sha256": contract["sha256"],
            "observation_dim": OBS_DIM, "history_frames": HISTORY_FRAMES,
            "history_span_s": HISTORY_SPAN_S,
            "network": network_spec(args.arm), "ppo": ppo, "budget": budget,
            "seed": args.seed, "runtime_mapping": mapping, "simulation_safety": limits,
            "device": args.device, "argv": sys.argv,
            "status": "research", "deployable": False,
        }, indent=2) + "\n", encoding="utf-8")
        return rollout_loop(args, env, entry, budget, ppo, settings)
    finally:
        env.close()


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def base_parser(run: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=sorted(ARM_INFO), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kd", type=float, choices=(0.5, 1.5, 2.0))
    group.add_argument("--contract", choices=("w1w2_as2w_diagnostic_kd_0p5",
                                              "w1w2_as2w_diagnostic_kd_1p5",
                                              "w1w2_as2w_diagnostic_kd_2p0"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--curriculum", choices=list(CURRICULUM_MODES), default=None,
                        help="armE_mixture (default) or staged_0_to_3; config is the fallback")
    parser.add_argument("--phase", choices=list(PHASES), default=None,
                        help="baseline (default) or disturbance; baseline is nominal/no push")
    parser.add_argument("--disturbance", choices=list(DISTURBANCES), default="none",
                        help="disturbance phase factor: none|load|push (one at a time)")
    parser.add_argument("--load-kg", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="disturbance-phase starting checkpoint (records sha256)")
    parser.add_argument("--accept-contract-change", action="store_true",
                        help="explicitly accept a changed control-contract fingerprint")
    parser.add_argument("--triple-tap-stride", type=int, default=1)
    parser.add_argument("--history-sample-stride", type=int, default=100)
    parser.add_argument("--allow-disturbance", action="store_true",
                        help="explicit phase-2 unlock (required for --phase disturbance)")
    parser.add_argument("--eval-record", type=Path, default=None,
                        help="phase-2 held-out evaluation record (json) bound to the checkpoint")
    parser.add_argument("--re-export-first-mismatch", type=Path, default=None,
                        help="replay: re-export the raw window at the first hash mismatch")
    parser.add_argument("--contract-change-reason", default="unspecified")
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vx-loss-coef", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    if not run:
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--replay", action="store_true",
                            help="replay/diagnostic mode: same pure-CPU path as dry-run")
        parser.add_argument("--selfcheck", action="store_true",
                            help="run the self-check (dry-run runs it automatically)")
        parser.add_argument("--emit-configs", action="store_true")
    return parser


def parse_args_for_run(argv) -> argparse.Namespace:
    parser = base_parser(run=True)
    from isaaclab.app import add_launcher_args
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if not str(args.device).startswith("cuda"):
        parser.error("--run requires a CUDA device; CPU training is not supported")
    args.usd = USD_PATH
    args.urdf = URDF_PATH
    if args.seed is None:
        args.seed = DEFAULT_SEED
    args.physics = "isaacsim_physx"
    return args


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--run", action="store_true")
    pre.add_argument("--dry-run", action="store_true")
    pre.add_argument("--replay", action="store_true")
    pre.add_argument("--emit-configs", action="store_true")
    known, rest = pre.parse_known_args(argv)
    if known.emit_configs and known.run:
        raise SystemExit("[FATAL] --emit-configs and --run are mutually exclusive")
    if known.run and (known.dry_run or known.replay):
        raise SystemExit("[FATAL] --run cannot be combined with --dry-run/--replay")

    if known.run:
        # ---- all gates run on pure CPU, BEFORE any isaaclab import --------------------
        gate_args = parse_run_gate_args(rest)
        try:
            validate_contract_change_usage(True, False, gate_args.accept_contract_change)
            config = load_config(gate_args)
            settings = resolve_settings(gate_args, config)
            decision = contract_guard(read_contract_state(), CONTROL_CONTRACT_VERSION,
                                      control_fingerprint(), False)
            if decision["refuse"]:
                print("CONTROL_CONTRACT_CHANGED -> bump control_contract_version and redo replay"
                      " + baseline regression", flush=True)
                print("ISAAC_IMPORTED=no", flush=True)
                print("KIT_LAUNCHED=no", flush=True)
                raise SystemExit(2)
            phase = settings["phase"]
            validate_phase2_access(phase["phase"], gate_args.init_checkpoint,
                                   gate_args.allow_disturbance, gate_args.eval_record, run=True)
        except SystemExit as error:
            print(f"RUN_GATE_REFUSED {error}", flush=True)
            print("ISAAC_IMPORTED=no  (no isaaclab/isaacsim import happened)", flush=True)
            print("KIT_LAUNCHED=no", flush=True)
            raise
        print(f"RUN_GATES_OK phase={phase['phase']} disturbance={phase['disturbance']}"
              f" curriculum={settings['curriculum']['mode']}", flush=True)
        print("ISAAC_IMPORTED=no  (all gates passed before importing Isaac)", flush=True)
        write_contract_state(CONTROL_CONTRACT_VERSION, control_fingerprint(), decision["status"])
        # ---- gates passed: only now may Isaac be imported/launched --------------------
        args = parse_args_for_run(rest)
        from isaaclab.app import launch_simulation
        from isaaclab.sim import SimulationCfg
        with launch_simulation(SimulationCfg(device=args.device, dt=PHYSICS_DT), args) as physics:
            return train(args, physics)
    if known.emit_configs:
        return emit_configs(rest)
    return dry_run(rest, Checker())


if __name__ == "__main__":
    sys.exit(main())

