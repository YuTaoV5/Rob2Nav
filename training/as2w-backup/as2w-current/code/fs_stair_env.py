"""FS stair line: from-scratch blind stair environment with stage curriculum.

This module is the *only* place that defines the FS stair control contract,
reward, termination, milestones and stage geometry. It is deliberately
self-contained and does not import ``train_stair_curriculum`` (the repo copy is
being edited by another agent); the pinned reference copy is hashed and its
literal control scales are re-read by ``--selfcheck``.

Provenance (sha256 verified at start-up before anything else runs)
-----------------------------------------------------------------
* control / reward / termination contract
  ``/root/gpufree-data/as2w-runs/w1w2-stair-scratch.kuZUnA/train_stair_curriculum.py``
  sha256 a08b8d76f357c93e09352c3e66bac43136e744a5f127e60795f6bfb3bd1991dd
  (identical to the repo copy at prep time; pinned because the repo copy is
  owned by the other stair agent).
* stair model builder + sustained-load milestone accounting (parity-gated copy)
  ``/root/gpufree-data/as2w-runs/w1w2-stair-ab.tTar7N/w1w2_stair_env.py``
  sha256 8b470b84ea6b195dab465bcb45fb2eefcc60eb362ad8a92446128a179e8676db
* MuJoCo model / limits / dynamics stress: r152 delivery runtime
  ``.../omni-blind-stairs-r152/runtime/mujoco_screen.py`` sha256 0aa9f80b...
* assets ``assets/r237-train-root`` (MJCF, URDF, fall-recovery contract).

Deliberate differences from the pinned training env (all printed in --dry-run)
-----------------------------------------------------------------------------
1. Stage geometry (levels / rise / tread) is explicit and read back from the
   loaded model, then asserted. The pinned env sampled tread {0.31,0.35,0.39}
   from the scene seed; the FS line fixes tread 0.31 so the known geometry
   (0.70-2.70 / 0.70-3.01 / 0.70-3.94) can be asserted.
2. ``flat_approach`` (0 stair levels) and ``step1``/``step2`` exist as real,
   separately runnable stages; promotion is stage-count based instead of a
   riser-height sweep.
3. Torque is clipped to the 90 % budget (13.5 Nm wheels, 54/81 Nm legs) read
   from the wheel contract / AGENTS.md rule instead of the raw MJCF ctrlrange
   (+-15 / +-60 / +-90). ``torque_clip="mujoco_ctrlrange"`` reproduces the
   pinned env exactly and is recorded as such.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np

import fs_stair_contract as contract_mod
import mujoco_screen as sim

# ---------------------------------------------------------------- pinned consts
WHEEL_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
WHEEL_LABELS = ("FL", "FR", "RL", "RR")
MILESTONE_WEIGHTS = np.array((.25, .25, .50, .50))
SUSTAINED_LOAD_N = 8.
TOP_HOLD_S = .5
MILESTONE_DWELL_S = .06
TRAIN_TERM_IMPACT_N = 90.
TRAIN_TERM_LATERAL_M = .65
NONWHEEL_CONTACT_N = 1.
ACCEPT_PEAK_IMPACT_N = 50.
ACCEPT_TOP_HOLD_S = .5
FALL_UPRIGHT = .80
STABLE_UPRIGHT = .94
LEG_ACTION_SCALE = .35
WHEEL_ACTION_SCALE = 16.
ACTION_CLIP = 6.
LEG_KP = 55.
JOINT_KD = 2.
STAIR_EPISODE_S = 9.
FLAT_EPISODE_S = 6.
OBS_DIM = 60
ACTION_DIM = 16
GAIT_CLOCK_PERIOD_S = .62
COMMAND = np.array((.4, 0., 0.))

# Phase-1 initial-state distribution (shared verbatim by S/H0/H6 and part of
# contract_fingerprint).  Values mirror the pinned training env's reset.
INITIAL_STATE = {
    "base_x_uniform_m": [-0.12, 0.12],
    "base_y_uniform_m": [-0.15, 0.15],
    "yaw_uniform_rad": [-0.08, 0.08],
    "joint_offset_uniform_rad": [-0.12, 0.12],
    "base_height_m": 0.37,
    "start_back_flat_m": 0.35,
    "scene_seed_range": [100000, 1000000000],
}

# Dispatch semantics recorded next to the numbers so units can never be confused.
DISPATCH = {
    "wheel_target_unit": "rad/s",
    "dispatch_unit": "Nm at data.ctrl; data.actuator_force is the measured solver force",
    "previous_action_semantics": "executed (delayed, clipped to +-action_clip) action; equals "
                                 "the raw policy action when |a| <= action_clip and delay = 0",
}

# Default identity control wrapper (phase 1 baseline).  Every field is explicit
# and participates in contract_fingerprint.
DEFAULT_CONTROL = {
    "action_clip": ACTION_CLIP,
    "leg_action_scale": LEG_ACTION_SCALE,
    "wheel_action_scale": WHEEL_ACTION_SCALE,
    "wheel_target_filter_alpha": 1.0,
    "wheel_target_rate_limit_per_s": None,
    "leg_target_filter_alpha": 1.0,
    "action_delay_frames": 0,
    "reset_rule": "clear_filter_delay_and_history_on_episode_reset",
    "history_reset_fill": "repeat_first_frame",
    "torque_clip_mode": "90pct",
    "torque_budget_nm": 13.5,
}

SOURCE_PINS = {
    "pinned_training_copy": {
        "path": "/root/gpufree-data/as2w-runs/w1w2-stair-scratch.kuZUnA/train_stair_curriculum.py",
        "sha256": "a08b8d76f357c93e09352c3e66bac43136e744a5f127e60795f6bfb3bd1991dd",
        "role": "control / reward / termination contract, re-read by --selfcheck",
    },
    "w1w2_stair_env": {
        "path": "/root/gpufree-data/as2w-runs/w1w2-stair-ab.tTar7N/w1w2_stair_env.py",
        "sha256": "8b470b84ea6b195dab465bcb45fb2eefcc60eb362ad8a92446128a179e8676db",
        "role": "milestone accounting / stage builder reference (parity gated)",
    },
    "mujoco_screen": {
        "path": "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-r284-release/"
                "deliveries/as2w-deploy/omni-blind-stairs-r152/runtime/mujoco_screen.py",
        "sha256": "0aa9f80ba19e8c57fd36c499ee99e0a79857c526894388fba23f0ba75bde2bfd",
        "role": "MuJoCo model loading, target limits, dynamics stress",
    },
    "as2w_flat_xml": {
        "path": "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/assets/"
                "r237-train-root/local-step-diagnostic/as2w_flat.xml",
        "sha256": "bd86aae1d8047a7be67aef8340c1303950f136af5fb5506dfeb451a1c94d0ce2",
        "role": "robot + ground MJCF used for every stage",
    },
    "as2w_urdf": {
        "path": "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/assets/"
                "r237-train-root/official-model/robots/as2w_description/as2w.urdf",
        "sha256": "696aec2c0915ec1e06bcc91a63104d8234c51c0eb94331fa5496bf426c51e926",
        "role": "joint limits (load_target_limits)",
    },
    "fall_recovery_contract": {
        "path": "/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current/assets/"
                "r237-train-root/fall-baseline/fall-recovery-contract.json",
        "sha256": "1e4e3fb76806e83dbe3541979fa0cb40f4b3150f5bf4de298224bf9baa9b0490",
        "role": "joint order / default pose / control_dt / action scales",
    },
}


class ContractError(RuntimeError):
    """Raised for any naming, index or contract inconsistency. Never caught."""


# ---------------------------------------------------------------- thresholds
@dataclass(frozen=True)
class Thresholds:
    """The three threshold sets are kept separate on purpose.

    * ``train_termination``  -- when the *training / replay* episode stops.
    * ``acceptance``         -- when a *formal stair acceptance* run passes.
    * ``nonwheel_contact``   -- when a contact counts as a non-wheel touch.

    They are never merged: a run can be safe for training (impact 60 N does not
    terminate) while still failing acceptance (peak impact must stay < 50 N).
    """

    train_termination_impact_n: float = TRAIN_TERM_IMPACT_N
    train_termination_lateral_m: float = TRAIN_TERM_LATERAL_M
    train_fall_upright: float = FALL_UPRIGHT
    accept_peak_impact_n: float = ACCEPT_PEAK_IMPACT_N
    accept_top_hold_s: float = ACCEPT_TOP_HOLD_S
    accept_all_levels: int = 5
    nonwheel_contact_n: float = NONWHEEL_CONTACT_N

    def as_dict(self) -> dict:
        return {
            "train_termination": {
                "source": "train_stair_curriculum.py:207",
                "impact_n_gt": self.train_termination_impact_n,
                "abs_y_m_gt": self.train_termination_lateral_m,
                "fall_upright_lt": self.train_fall_upright,
            },
            "acceptance": {
                "source": "mujoco_screen.py:295",
                "levels_ordered": self.accept_all_levels,
                "top_stable_s_ge": self.accept_top_hold_s,
                "peak_nonwheel_impact_n_lt": self.accept_peak_impact_n,
            },
            "nonwheel_contact": {
                "source": "mujoco_screen.py:254",
                "contact_force_n_ge": self.nonwheel_contact_n,
                "note": "a contact at or above this force counts as a non-wheel touch",
            },
        }


# ---------------------------------------------------------------- stage model
@dataclass
class Stage:
    name: str
    levels: int
    rise_m: float | None
    tread_m: float | None
    episode_s: float
    first_riser_x_m: float
    start_back_m: float = 0.
    approach_line_m: float | None = None
    description: str = ""
    known_geometry: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict) -> "Stage":
        known = row.get("known_geometry") or {}
        return cls(
            name=str(row["name"]),
            levels=int(row["levels"]),
            rise_m=None if row.get("rise_m") is None else float(row["rise_m"]),
            tread_m=None if row.get("tread_m") is None else float(row["tread_m"]),
            episode_s=float(row["episode_s"]),
            first_riser_x_m=float(row["first_riser_x_m"]),
            start_back_m=float(row.get("start_back_m", 0.)),
            approach_line_m=None if row.get("approach_line_m") is None else float(row["approach_line_m"]),
            description=str(row.get("description", "")),
            known_geometry=dict(known),
        )

    def geometry_from_config(self) -> list:
        """Expected geometry derived from the configured stage parameters.

        The derivation mirrors the MJCF builder formula, so comparing it with
        the *loaded* model catches silent drift between config and asset.
        """
        rows = []
        previous_top = 0.
        for level in range(1, self.levels + 1):
            tread = float(self.tread_m)
            start = .7 + (level - 1) * tread
            length = tread if level < self.levels else 2.
            top = level * float(self.rise_m)
            rows.append({"level": level, "x_front_m": start, "x_back_m": start + length,
                         "tread_top_m": top, "rise_m": top - previous_top,
                         "tread_depth_m": length})
            previous_top = top
        return rows


def build_stair_model(assets: Path, seed: int, n_levels: int,
                      rise: float | None, tread: float | None):
    """Byte-parity copy of the stair branch of ``mujoco_screen.load_flat_model``.

    ``n_levels=5, rise=None, tread=None`` must reproduce the delivery model
    exactly; ``--selfcheck`` proves that against the runtime implementation.
    ``n_levels=0`` keeps the robot and the ground contact configuration but
    builds no riser, which is what ``flat_approach`` needs.
    """
    path = Path(assets) / "local-step-diagnostic/as2w_flat.xml"
    xml = ET.parse(path).getroot()
    for asset in xml.findall("./asset/mesh"):
        asset.set("file", (path.parent / asset.attrib["file"]).resolve().as_posix())
    ground_geom = xml.find("./worldbody/geom[@name='ground']")
    ground_geom.set("group", "5")
    if n_levels:
        rise = (.06, .08, .10)[seed % 3] if rise is None else float(rise)
        tread = (.31, .35, .39)[(seed // 3) % 3] if tread is None else float(tread)
        for level in range(1, n_levels + 1):
            start = .7 + (level - 1) * tread
            length = tread if level < n_levels else 2.
            top = level * rise
            ET.SubElement(xml.find("worldbody"), "geom", name=f"stair_{level}", type="box",
                          size=f"{length / 2} 1.25 {top / 2}",
                          pos=f"{start + length / 2} 0 {top / 2}", group="5",
                          friction=ground_geom.attrib["friction"])
    model = sim.mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    ground = model.geom("ground").id
    surface_ids = [ground]
    surface_ids.extend(model.geom(f"stair_{level}").id for level in range(1, n_levels + 1))
    model.geom_priority[surface_ids] = int(model.geom_priority.max()) + 1
    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = 0
    model.geom_contype[surface_ids] = 0
    model.geom_conaffinity[surface_ids] = 1
    return model


def stair_geometry(model, n_levels: int) -> list:
    """Read the effective stair geometry back out of the loaded model."""
    rows = []
    previous_top = 0.
    for level in range(1, n_levels + 1):
        geom = model.geom(f"stair_{level}")
        size = np.array(geom.size, dtype=float)
        pos = np.array(geom.pos, dtype=float)
        x0 = float(pos[0] - size[0])
        x1 = float(pos[0] + size[0])
        top = float(pos[2] + size[2])
        rows.append({"level": level, "x_front_m": x0, "x_back_m": x1,
                     "tread_top_m": top, "rise_m": top - previous_top,
                     "tread_depth_m": x1 - x0})
        previous_top = top
    return rows


def geometry_x_range(rows: list) -> list:
    if not rows:
        return [None, None]
    return [min(row["x_front_m"] for row in rows), max(row["x_back_m"] for row in rows)]


def assert_stage_geometry(stage: Stage, rows: list, x_range: list, tol: float = 1e-6) -> None:
    """Fail fast when the loaded model disagrees with the configured stage."""
    expected = stage.geometry_from_config()
    if len(rows) != stage.levels:
        raise ContractError(f"{stage.name}: read {len(rows)} levels, expected {stage.levels}")
    for want, got in zip(expected, rows):
        for key in ("x_front_m", "x_back_m", "tread_top_m", "rise_m", "tread_depth_m"):
            if abs(float(want[key]) - float(got[key])) > tol:
                raise ContractError(
                    f"{stage.name}: level {want['level']} {key} read back {got[key]!r} "
                    f"!= configured {want[key]!r}")
    if rows:
        if abs(x_range[0] - rows[0]["x_front_m"]) > tol:
            raise ContractError(f"{stage.name}: x_range front mismatch")
        if abs(x_range[1] - rows[-1]["x_back_m"]) > tol:
            raise ContractError(f"{stage.name}: x_range back mismatch")
        if abs(stage.known_geometry.get("x_front_m", rows[0]["x_front_m"])
               - rows[0]["x_front_m"]) > tol:
            raise ContractError(f"{stage.name}: x front differs from the known geometry")
        if abs(stage.known_geometry.get("x_back_m", rows[-1]["x_back_m"])
               - rows[-1]["x_back_m"]) > tol:
            raise ContractError(f"{stage.name}: x back differs from the known geometry")


# ---------------------------------------------------------------- contract
def load_contract(assets: Path) -> dict:
    return json.loads((Path(assets) / "fall-baseline/fall-recovery-contract.json").read_text())


def validate_contract(model, contract: dict) -> dict:
    """Name / index / order validation. Any inconsistency raises ContractError."""
    names = list(contract["joint_names"])
    if len(names) != ACTION_DIM:
        raise ContractError(f"contract has {len(names)} joints, expected {ACTION_DIM}")
    if len(set(names)) != len(names):
        raise ContractError("contract joint names are not unique")
    action_order = [name for term in contract["action_terms"] for name in term["joints"]]
    if action_order != names:
        raise ContractError("action_terms order differs from joint_names order")
    missing = [name for name in names if model.joint(name).id < 0] if False else [
        name for name in names if _joint_id(model, name) is None]
    if missing:
        raise ContractError(f"joint names missing from the model: {missing}")
    if tuple(names[12:]) != WHEEL_NAMES:
        raise ContractError(f"unexpected wheel order {names[12:]}, expected {WHEEL_NAMES}")
    actuator_names = [model.actuator(name).name for name in names]
    if len(set(actuator_names)) != len(names):
        raise ContractError("duplicate actuator names in the model")
    body_ids = [int(model.joint(name).bodyid[0]) for name in names[12:]]
    if len(set(body_ids)) != 4:
        raise ContractError(f"wheel joints do not map to four distinct bodies: {body_ids}")
    scales = {term["name"]: float(term["scale"]) for term in contract["action_terms"]}
    if abs(scales["joint_pos"] - LEG_ACTION_SCALE) > 1e-12:
        raise ContractError(
            f"contract leg action scale {scales['joint_pos']} != pinned {LEG_ACTION_SCALE}")
    return {"joint_names": names, "actuator_names": actuator_names,
            "wheel_body_ids": body_ids, "contract_scales": scales}


def _joint_id(model, name: str):
    try:
        return int(model.joint(name).id)
    except Exception:  # noqa: BLE001 - mujoco raises KeyError for unknown names
        return None


# ---------------------------------------------------------------- milestones
class MilestoneTracker:
    """Sustained-load wheel milestones (W1W2 semantics, kept identical).

    A wheel is credited with tread ``level`` only when

    * the contact normal is vertical (``|frame[2]| >= .7``), so a riser face
      contact can never be credited, and
    * the wheel carries at least ``load_n`` newtons on that tread for at least
      ``dwell_s`` seconds, and
    * the robot is stable and at least three wheels are loaded (the pinned
      ``supported_stair_progress`` gate).

    Credited levels are monotone per wheel (``np.maximum``), so stepping on the
    same tread again can never award the milestone twice.
    """

    def __init__(self, n_levels: int, dt: float, *, weights=MILESTONE_WEIGHTS,
                 load_n: float = SUSTAINED_LOAD_N, dwell_s: float = MILESTONE_DWELL_S):
        self.n_levels = int(n_levels)
        self.dt = float(dt)
        self.weights = np.asarray(weights, dtype=float)
        self.load_n = float(load_n)
        self.dwell_s = float(dwell_s)
        self.reset()

    def reset(self) -> None:
        self.levels = np.zeros(4, dtype=int)
        self.candidate = np.zeros(4, dtype=int)
        self.streak = np.zeros(4, dtype=int)
        self.credit_events = []

    def update(self, tread_loads: np.ndarray, stable: bool, frame: int = 0):
        supported = np.asarray(tread_loads, dtype=float) >= self.load_n
        if stable and np.count_nonzero(supported.any(axis=1)) >= 3:
            candidates = (supported * np.arange(supported.shape[1])).max(axis=1)
        else:
            candidates = np.zeros(4, dtype=int)
        self.streak = np.where(
            candidates > 0, np.where(candidates == self.candidate, self.streak + 1, 1), 0)
        self.candidate = candidates
        contacts = np.where(self.streak * self.dt >= self.dwell_s - 1e-9, candidates, 0)
        previous = self.levels.copy()
        self.levels = np.maximum(self.levels, contacts)
        credited = np.nonzero(self.levels > previous)[0]
        for wheel in credited:
            self.credit_events.append({"wheel": WHEEL_LABELS[int(wheel)],
                                       "wheel_name": WHEEL_NAMES[int(wheel)],
                                       "level": int(self.levels[wheel]), "frame": int(frame),
                                       "t_s": round(frame * self.dt, 4)})
        newly_reached = float((self.levels - previous) @ self.weights)
        return contacts, newly_reached


# ---------------------------------------------------------------- curriculum
PROMOTION_SIGNALS = ("top_stable_hold", "all_wheels_top", "longest_all4_top_s")


def promotion_complete(signal: str, *, top_stable: bool, all_wheels_top: bool,
                       longest_all4_top_s: float, min_top_hold_s: float,
                       min_all4_top_s: float, flat_stage: bool = False) -> bool:
    """Evaluate the configured promotion signal for one episode.

    ``top_stable_hold`` is the task-defined criterion (top stable hold >= 0.5 s).
    The diagnostic line measured it at 0/6 on 1-, 2- and 5-step terrain for
    three arms, so it is *expected* to stall the ladder at stage 0; the
    alternatives exist only so the operator can switch signals explicitly.
    ``flat_approach`` has no wheel-top milestone, so it always uses its own
    approach-line + stable-hold completion.
    """
    if flat_stage:
        return bool(top_stable)
    if signal == "top_stable_hold":
        return bool(top_stable)
    if signal == "all_wheels_top":
        return bool(all_wheels_top)
    if signal == "longest_all4_top_s":
        return float(longest_all4_top_s) >= float(min_all4_top_s) - 1e-9
    raise ContractError(f"unknown promotion signal {signal!r}; expected one of {PROMOTION_SIGNALS}")


class DegenerationMonitor:
    """Fires ``PROMOTION_SIGNAL_DEGENERATE`` when a stage never completes.

    Counts consecutive from-bottom episodes whose promotion signal is False.
    Once the counter reaches ``after`` for a stage the warning is emitted once;
    a single success resets the counter for that stage.
    """

    def __init__(self, after: int = 6):
        self.after = max(1, int(after))
        self.count = 0
        self.fired = set()
        self.by_stage = {}

    def update(self, stage: str, signal: bool) -> bool:
        self.by_stage.setdefault(stage, {"episodes": 0, "completed": 0})
        self.by_stage[stage]["episodes"] += 1
        self.by_stage[stage]["completed"] += int(bool(signal))
        if signal:
            self.count = 0
            return False
        self.count += 1
        if self.count >= self.after and stage not in self.fired:
            self.fired.add(stage)
            return True
        return False

    def summary(self) -> dict:
        return {"consecutive_without_signal": self.count, "stop_after": self.after,
                "fired_for_stages": sorted(self.fired), "by_stage": self.by_stage}


class StairCurriculum:
    """Promotion / demotion over the stage ladder (pinned logic, stage index).

    Promotion counts only episodes that (a) started from the bottom, (b) were
    not manually placed or reset mid-course, (c) completed the *configured*
    promotion signal and (d) never tripped a hard protection. Manual top
    placement and mid-course resets are logged separately and never feed the
    counters.

    Demotion counts from-bottom episodes: any hard protection demotes
    immediately, and ``demote_after`` consecutive zero-progress episodes demote.
    """

    def __init__(self, stages: list, start_index: int = 0,
                 promote_after: int = 2, demote_after: int = 2):
        if not stages:
            raise ContractError("stage ladder is empty")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ContractError(f"duplicate stage names in the ladder: {names}")
        if not 0 <= start_index < len(stages):
            raise ContractError(f"start stage {start_index} outside the ladder")
        self.stages = list(stages)
        self.index = int(start_index)
        self.promote_after = int(promote_after)
        self.demote_after = int(demote_after)
        self.successes = 0
        self.no_progress = 0
        self.history = []

    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    def record(self, outcome: dict) -> dict:
        full_start = bool(outcome.get("started_from_bottom")) and not bool(outcome.get("manual"))
        complete = bool(outcome.get("complete"))
        hard = bool(outcome.get("hard_protection"))
        zero = bool(outcome.get("zero_progress"))
        eligible = full_start and complete and not hard
        row = {"stage_from": self.stage.name, "full_start": full_start,
               "eligible_for_promotion": eligible, "hard_protection": hard,
               "zero_progress": zero, "promoted": False, "demoted": False}
        if not full_start:
            self.history.append(row)
            return row
        if complete:
            self.successes += 1
        else:
            self.successes = 0
        if hard or zero:
            self.no_progress = self.no_progress + 1 if zero else self.no_progress
        if hard or self.no_progress >= self.demote_after:
            if self.index > 0:
                self.index -= 1
                row["demoted"] = True
            self.successes = 0
            self.no_progress = 0
        elif eligible and self.successes >= self.promote_after:
            if self.index < len(self.stages) - 1:
                self.index += 1
                row["promoted"] = True
            self.successes = 0
            self.no_progress = 0
        row["stage_to"] = self.stage.name
        self.history.append(row)
        return row


class ControlChain:
    """The one action -> target pipeline shared by training and evaluation.

    Deployment-consistent wrapper (phase 1): clip the action to +-6, optional
    action delay, then the wheel velocity target (16 * a[12:], limited to the
    URDF wheel speed), optional rate limit and low-pass alpha, and the leg
    position target (default + 0.35 * a[:12], limited to the URDF range).
    With the shipped defaults (alpha=1.0, rate=None, delay=0) every optional
    stage is the identity, so the chain reproduces the pinned training env
    exactly; the parameters are still explicit and fingerprinted, so adding a
    filter later is a contract change instead of a silent drift.
    """

    def __init__(self, control: dict | None = None):
        self.cfg = dict(DEFAULT_CONTROL)
        self.cfg.update(control or {})
        self.dt = None
        self.wheel_limit = None
        self.leg_low = None
        self.leg_high = None
        self.defaults = None
        self.reset()

    def bind(self, dt, wheel_limit, leg_low, leg_high, defaults):
        self.dt = float(dt)
        self.wheel_limit = np.asarray(wheel_limit, dtype=float)
        self.leg_low = np.asarray(leg_low, dtype=float)
        self.leg_high = np.asarray(leg_high, dtype=float)
        self.defaults = np.asarray(defaults, dtype=float)

    def reset(self):
        delay = int(self.cfg["action_delay_frames"])
        self._delay = deque([np.zeros(16, dtype=np.float32) for _ in range(delay + 1)],
                            maxlen=delay + 1)
        self._prev_wheel_limited = None
        self._prev_wheel_filtered = None
        self._prev_leg_filtered = None

    def targets(self, action) -> dict:
        if self.defaults is None:
            raise ContractError("ControlChain.targets called before bind()")
        clip = float(self.cfg["action_clip"])
        executed = np.clip(np.asarray(action), -clip, clip)
        if int(self.cfg["action_delay_frames"]):
            self._delay.append(executed)
            executed = self._delay[0]
        wheel_pre = float(self.cfg["wheel_action_scale"]) * executed[12:]
        wheel_limited = np.clip(wheel_pre, -self.wheel_limit, self.wheel_limit)
        wheel = wheel_limited
        if self.cfg["wheel_target_rate_limit_per_s"] is not None and \
                self._prev_wheel_limited is not None:
            step_limit = float(self.cfg["wheel_target_rate_limit_per_s"]) * self.dt
            wheel = self._prev_wheel_limited + np.clip(
                wheel - self._prev_wheel_limited, -step_limit, step_limit)
        alpha = float(self.cfg["wheel_target_filter_alpha"])
        if alpha < 1.0 and self._prev_wheel_filtered is not None:
            wheel = alpha * wheel + (1. - alpha) * self._prev_wheel_filtered
        self._prev_wheel_limited = wheel_limited.copy()
        self._prev_wheel_filtered = wheel.copy()
        leg_pre = self.defaults[:12] + float(self.cfg["leg_action_scale"]) * executed[:12]
        if float(self.cfg["leg_target_filter_alpha"]) < 1.0 and \
                self._prev_leg_filtered is not None:
            beta = float(self.cfg["leg_target_filter_alpha"])
            leg_pre = beta * leg_pre + (1. - beta) * self._prev_leg_filtered
        self._prev_leg_filtered = leg_pre.copy()
        leg = np.clip(leg_pre, self.leg_low, self.leg_high)
        return {"executed": executed, "wheel_pre": wheel_pre,
                "wheel_limited": wheel_limited, "wheel": wheel,
                "leg_pre": leg_pre, "leg": leg}


# ---------------------------------------------------------------- observation
class _TapLogMixin:
    """Per-policy-step three-segment logging + sampled raw windows.

    One JSONL row per (strided) policy step holds, for the same ``step_id``:

    * ``target_wheel_pre_filter``  [4]  rad/s -- 16*action[12:] before clip/filter
    * ``target_wheel_clipped``     [4]  rad/s -- after the wheel velocity limit
    * ``target_wheel_dispatch``    [4]  rad/s -- captured at the data.ctrl assignment point
    * ``target_leg_pos_pre_filter``[12] rad   -- default + 0.35*action[:12], pre-clip
    * ``target_leg_pos_dispatch``  [12] rad   -- after the leg limit (leg_low/leg_high)
    * ``tau_request_est_nm``       [16] Nm    -- ESTIMATED PD request
    * ``tau_applied_measured_nm``  [16] Nm    -- MEASURED data.actuator_force
    * ``torque_budget_nm`` / ``torque_clip_mode`` / ``wheel_limit_rad_s``
    * ``history_newest``           [60]       -- the newest frame the next policy step sees
    * ``history_sha256``                       -- sha256 of the whole float32 window
    * ``env_id, episode_id, policy_step, dtype, field_order, window_shape``

    Windows are sampled to ``.npz`` (periodic, episode first/last, and +/-3 steps
    around the first failure of each kind) so a replay can locate the first
    divergent step and field.
    """

    def _init_tap(self, tap):
        tap = dict(tap or {})
        self.tap_config = tap
        self.tap_enabled = bool(tap.get("enabled", False))
        self.tap_stride = max(1, int(tap.get("stride", 1)))
        self.tap_max_rows = int(tap.get("max_rows", 50000))
        self.tap_dir = Path(tap["dir"]) if tap.get("dir") else None
        self.tap_worker = int(tap.get("worker_id", 0))
        self.env_id = f"w{self.tap_worker}"
        self.tap_rows = 0
        self.tap_truncated = False
        self.window_every = max(1, int(tap.get("window_every_steps", 100)))
        self.windows_max = int(tap.get("max_windows", 200))
        self.window_dir = Path(tap["window_dir"]) if tap.get("window_dir") else None
        self._window_count = 0
        self._window_buffer = deque(maxlen=6)
        self._seen_failures = set()
        self._failure_post = 0
        self._log_path = None
        if self.tap_enabled and self.tap_dir is not None:
            self.tap_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = self.tap_dir / f"triple_tap.worker{self.tap_worker}.jsonl"
        if self.window_dir is not None:
            self.window_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def _window_sha(self, stacked) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(stacked, dtype="<f4").tobytes()).hexdigest()

    def _save_window(self, stacked, row, reason: str) -> None:
        if self.window_dir is None or self._window_count >= self.windows_max:
            return
        self._window_count += 1
        path = self.window_dir / (f"w{self.tap_worker}_step{row['step_id']:07d}_{reason}.npz")
        np.savez_compressed(
            path,
            obs=np.asarray(stacked, dtype="<f4").reshape(self.frames, self.base_dim),
            history_sha256=np.array([row["history_sha256"]]),
            field_order=np.asarray(contract_mod.FIELD_ORDER),
            step_id=np.array([row["step_id"]]),
            episode_id=np.array([row["episode_id"]]),
            policy_step=np.array([row["policy_step"]]),
            scene_seed=np.array([row["scene_seed"]]),
            reason=np.array([reason]),
            target_wheel_dispatch=np.asarray(row["target_wheel_dispatch"], dtype="<f8"),
            target_wheel_pre_filter=np.asarray(row["target_wheel_pre_filter"], dtype="<f8"),
        )
        with (self.window_dir / "index.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "path": path.name, "reason": reason, "step_id": row["step_id"],
                "episode_id": row["episode_id"], "policy_step": row["policy_step"],
                "history_sha256": row["history_sha256"],
                "obs_shape": list(np.asarray(stacked).shape),
            }, sort_keys=True) + "\n")

    def _tap_after_step(self, stacked, newest, terminated: bool, truncated: bool) -> None:
        env = self.env.unwrapped
        tap = dict(env.last_tap)
        row = {
            "env_id": self.env_id,
            "episode_id": int(tap.get("episode_id", 0)),
            "policy_step": int(tap.get("tick", 0)),
            "scene_seed": int(tap.get("scene_seed", 0)),
            "stage": tap.get("stage"),
            "dtype": "float32",
            "field_order": list(contract_mod.FIELD_ORDER),
            "window_shape": [int(self.frames), int(self.base_dim)],
            "history_newest": [round(float(value), 6) for value in newest],
            "history_sha256": self._window_sha(stacked),
        }
        for key, value in tap.items():
            if key not in row:
                row[key] = value
        row["window_shape"] = [int(self.frames), int(self.base_dim)]
        if (self.tap_enabled and self._log_path is not None
                and row["policy_step"] % self.tap_stride == 0):
            if self.tap_rows < self.tap_max_rows:
                with self._log_path.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                self.tap_rows += 1
            elif not self.tap_truncated:
                self.tap_truncated = True
                with self._log_path.open("a") as handle:
                    handle.write(json.dumps({"kind": "triple_tap_truncated",
                                             "max_rows": self.tap_max_rows}) + "\n")
        # ---- sampled windows: periodic, episode edges, first-failure +/-3
        if self.window_dir is not None:
            self._window_buffer.append((np.array(stacked, copy=True), dict(row)))
            failures = set(env.first_fails) if hasattr(env, "first_fails") else set()
            new_failures = failures - self._seen_failures
            if new_failures:
                self._seen_failures |= new_failures
                for buffered, buffered_row in self._window_buffer:
                    self._save_window(buffered, buffered_row, "failure_pre")
                self._failure_post = 3
            if self._failure_post > 0:
                self._save_window(stacked, row, "failure_post")
                self._failure_post -= 1
            if row["policy_step"] <= 2:
                self._save_window(stacked, row, "episode_start")
            elif row["policy_step"] % self.window_every == 0:
                self._save_window(stacked, row, "periodic")
            if terminated or truncated:
                for buffered, buffered_row in list(self._window_buffer)[-2:]:
                    self._save_window(buffered, buffered_row, "episode_end")
                self._seen_failures = set()


class ObservationHistory(_TapLogMixin, gym.Wrapper):
    """Six-frame stack, oldest -> newest, one frame per policy step.

    Definition (also asserted by ``--selfcheck``):

    * the stack contains the *current* frame plus ``frames-1`` older frames;
    * consecutive frames are one policy step apart (20 ms), so six frames span
      5 * 20 ms = 100 ms between the oldest and the newest frame;
    * the stack is advanced once per policy step (not per physics substep);
    * ``reset`` clears the stack and fills it with copies of the first frame,
      so no stale frames from the previous episode survive.

    ``frames=1`` degenerates to the plain single-frame observation.
    """

    def __init__(self, env, frames: int = 1, tap: dict | None = None):
        super().__init__(env)
        if int(frames) < 1:
            raise ContractError(f"history frames must be >= 1, got {frames}")
        self.frames = int(frames)
        self.base_dim = int(np.prod(env.observation_space.shape))
        self.history = deque(maxlen=self.frames)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.frames * self.base_dim,), dtype=np.float32)
        self._init_tap(tap)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.extend(np.array(obs, dtype=np.float32, copy=True) for _ in range(self.frames))
        self._window_buffer.clear()
        self._seen_failures = set()
        self._failure_post = 0
        return np.concatenate(self.history), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(np.array(obs, dtype=np.float32, copy=True))
        stacked = np.concatenate(self.history)
        self._tap_after_step(stacked, obs, terminated, truncated)
        return stacked, reward, terminated, truncated, info


class ZeroHistory(_TapLogMixin, gym.Wrapper):
    """H0 control arm: 360-dim input whose historical slots are pinned to zero.

    It exists to separate the *shape / parameter-count* cost from the *history
    information* benefit: H0 and H6 have identical observation shape and actor
    parameters, so H0 -> H6 measures only the value of the history content.
    The newest slot always carries the current frame.
    """

    def __init__(self, env, frames: int = 6, tap: dict | None = None):
        super().__init__(env)
        if int(frames) < 2:
            raise ContractError(f"zero_history needs >= 2 frames, got {frames}")
        self.frames = int(frames)
        self.base_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.frames * self.base_dim,), dtype=np.float32)
        self._init_tap(tap)

    def _stack(self, obs):
        current = np.asarray(obs, dtype=np.float32)
        return np.concatenate(
            [np.zeros((self.frames - 1) * self.base_dim, dtype=np.float32), current])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._window_buffer.clear()
        self._seen_failures = set()
        self._failure_post = 0
        return self._stack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        stacked = self._stack(obs)
        self._tap_after_step(stacked, obs, terminated, truncated)
        return stacked, reward, terminated, truncated, info


# ---------------------------------------------------------------- environment
def read_transmission(model, names, motors) -> dict:
    """Read the actuator transmission back out of the loaded model.

    ``actuator_force`` is an actuator-coordinate force; it only equals the joint
    torque when the actuator is a direct joint motor with gear 1 and the joint
    has exactly one actuator.  Everything else must be read from
    ``qfrc_actuator`` (dof space) and labelled as a simulated joint torque.
    """
    actuators = []
    actuators_per_joint = {}
    joint_index_by_actuator = {}
    for index, (name, motor) in enumerate(zip(names, motors)):
        motor = int(motor)
        dyntype = int(model.actuator_dyntype[motor])
        gaintype = int(model.actuator_gaintype[motor])
        biastype = int(model.actuator_biastype[motor])
        trntype = int(model.actuator_trntype[motor])
        gear = float(model.actuator_gear[motor, 0])
        joint_id = int(model.actuator_trnid[motor, 0])
        joint_name = model.joint(joint_id).name
        if joint_name != name:
            raise ContractError(
                f"actuator {index} drives joint {joint_name!r}, contract expects {name!r}")
        kind = "motor" if (dyntype == 0 and gaintype == 0 and biastype == 0) else "general"
        actuators.append({"name": name, "actuator_id": motor, "kind": kind,
                          "trntype": trntype, "dyntype": dyntype, "gaintype": gaintype,
                          "biastype": biastype, "gear": gear, "joint_id": joint_id,
                          "joint_name": joint_name})
        joint_index_by_actuator[index] = index
        actuators_per_joint.setdefault(joint_id, []).append(name)
    one_per_joint = all(len(value) == 1 for value in actuators_per_joint.values())
    direct = all(row["kind"] == "motor" and abs(row["gear"] - 1.) < 1e-12
                 for row in actuators)
    quantity = "actuator_force" if (direct and one_per_joint) else "qfrc_actuator"
    if quantity == "actuator_force":
        label = ("MEASURED joint torque: every actuator is a direct joint motor with "
                 "gear=1 and one actuator per joint")
        actuator_label = "MEASURED joint torque (= data.actuator_force)"
    else:
        label = ("simulated joint actuator torque: the transmission is not 1:1 direct, "
                 "so data.qfrc_actuator (dof space) is used instead of actuator_force")
        actuator_label = "actuator-coordinate force (not necessarily joint torque)"
    return {"actuators": actuators,
            "actuators_per_joint": {str(key): value
                                    for key, value in actuators_per_joint.items()},
            "direct_motor_gear1": bool(direct),
            "one_actuator_per_joint": bool(one_per_joint),
            "torque_quantity": quantity, "label": label,
            "actuator_force_label": actuator_label,
            "joint_index_by_actuator": joint_index_by_actuator}


class StairEnv(gym.Env):
    """Blind AS2-W stair episode on a stage-specific stair model."""

    metadata = {"render_modes": []}

    def __init__(self, assets, stage: Stage, *, curriculum: StairCurriculum | None = None,
                 torque_clip: str = "contract_90pct", torque_budget_nm: float = 13.5,
                 trace: bool = False, nonwheel_contact_n: float = NONWHEEL_CONTACT_N,
                 promotion: dict | None = None, control: dict | None = None,
                 randomization: dict | None = None):
        super().__init__()
        self.assets = Path(assets)
        self.stage = stage
        self.curriculum = curriculum
        self.torque_clip = torque_clip
        self.torque_budget_nm = float(torque_budget_nm)
        self.trace_enabled = bool(trace)
        self.nonwheel_contact_n = float(nonwheel_contact_n)
        self.promotion = dict(promotion or {})
        self.promotion.setdefault("signal", "top_stable_hold")
        self.promotion.setdefault("min_top_hold_s", TOP_HOLD_S)
        self.promotion.setdefault("min_all4_top_s", .26)
        if self.promotion["signal"] not in PROMOTION_SIGNALS:
            raise ContractError(f"unknown promotion signal {self.promotion['signal']!r}")
        self.control = dict(control or {})
        self.control.setdefault("action_clip", ACTION_CLIP)
        self.control.setdefault("leg_action_scale", LEG_ACTION_SCALE)
        self.control.setdefault("wheel_action_scale", WHEEL_ACTION_SCALE)
        self.control.setdefault("wheel_target_filter_alpha", 1.0)
        self.control.setdefault("wheel_target_rate_limit_per_s", None)
        self.control.setdefault("leg_target_filter_alpha", 1.0)
        self.control.setdefault("action_delay_frames", 0)
        self.control.setdefault("reset_rule", "clear_filter_delay_and_history_on_episode_reset")
        self.randomization = dict(randomization or {})
        self.randomization.setdefault("load", False)
        self.randomization.setdefault("friction", False)
        self.randomization.setdefault("gain", False)
        self.randomization.setdefault("push", False)
        self.randomization.setdefault("com_offset", False)
        self.randomization.setdefault("observation_noise", False)
        if self.control["action_delay_frames"] < 0:
            raise ContractError("action_delay_frames must be >= 0")
        self.thresholds = Thresholds()
        sim.ROOT = self.assets
        self.contract = load_contract(self.assets)
        self.limits = sim.load_target_limits(self.contract)
        self.defaults = np.asarray(self.contract["default_joint_pos"], dtype=float)
        self.dt = float(self.contract["control_dt"])
        self.n_levels = int(stage.levels)
        self.top_index = self.n_levels
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -ACTION_CLIP, ACTION_CLIP, (ACTION_DIM,), dtype=np.float32)
        self.rows = []
        self._model_cache = None
        self.episode_id = 0
        self.tap_step = 0
        self.last_tap = {}
        self.randomization_applied = {}

    # ------------------------------------------------------------- properties
    @property
    def command(self):
        return COMMAND

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scene_seed = options.get("scene_seed")
        if scene_seed is None:
            scene_seed = int(self.np_random.integers(100_000, 1_000_000_000))
        scene_seed = int(scene_seed)
        manual_top = bool(options.get("manual_top", False))
        manual_mid = bool(options.get("manual_mid", False))
        self.episode_id += 1
        self.scene_seed = scene_seed
        self.manual = manual_top or manual_mid
        self.start_kind = "manual_top" if manual_top else ("manual_mid" if manual_mid else "bottom")
        rng = np.random.default_rng(scene_seed)
        model = build_stair_model(self.assets, scene_seed, self.n_levels,
                                  self.stage.rise_m, self.stage.tread_m)
        self.geometry = stair_geometry(model, self.n_levels)
        x_range = geometry_x_range(self.geometry)
        assert_stage_geometry(self.stage, self.geometry, x_range)
        self.model = model
        data = self.data = sim.mj.MjData(model)
        self.spec = validate_contract(model, self.contract)
        names = self.spec["joint_names"]
        joints = [model.joint(name) for name in names]
        self.qa = [joint.qposadr[0] for joint in joints]
        self.va = [joint.dofadr[0] for joint in joints]
        self.motors = [model.actuator(name).id for name in names]
        self.base = model.body("base_link").id
        self.wheel_index = {int(joint.bodyid[0]): i for i, joint in enumerate(joints[12:])}
        self.body_names = {i: model.body(i).name for i in range(model.nbody)}
        ground = model.geom("ground").id
        self.surfaces = {ground: 0}
        for level in range(1, self.n_levels + 1):
            self.surfaces[model.geom(f"stair_{level}").id] = level
        self.kp = np.r_[np.full(12, LEG_KP), np.zeros(4)]
        self.kd = np.full(16, JOINT_KD)
        flags = self.randomization
        for unsupported in ("com_offset", "observation_noise"):
            if flags.get(unsupported):
                raise ContractError(
                    f"randomization.{unsupported} is not implemented; enabling it would be a "
                    f"silent no-op. Implement it (and bump the control contract) first.")
        need_stress = any(bool(flags.get(key)) for key in ("load", "friction", "gain", "push"))
        self.stress = (sim.sample_dynamics_stress(scene_seed, model.nbody - 1, 16)
                       if need_stress else None)
        if self.stress is not None:
            if flags.get("load"):
                model.body_mass[1:] *= self.stress.mass_scale
                model.body_inertia[1:] *= self.stress.mass_scale[:, None]
            if flags.get("gain"):
                self.kp *= self.stress.gain_scale[0]
                self.kd *= self.stress.gain_scale[1]
        sim.mj.mj_setConst(model, data)
        if self.stress is not None and flags.get("friction"):
            model.geom_friction[list(self.surfaces), 0] = self.stress.friction
        self.randomization_applied = {key: bool(flags.get(key)) for key in
                                      ("load", "friction", "gain", "push")}
        # The nominal friction draw is consumed unconditionally: it is an RNG-stream
        # parity draw with the benchmark, not a randomization.
        rng.uniform(.55, .85)
        data.qpos[:2] = rng.uniform((-.12, -.15), (.12, .15))
        data.qpos[0] -= self.stage.start_back_m
        yaw = rng.uniform(-.08, .08)
        data.qpos[3:7] = [np.cos(yaw / 2.), 0., 0., np.sin(yaw / 2.)]
        data.qpos[self.qa] = self.defaults + rng.uniform(-.12, .12, size=16)
        data.qpos[2] = .37
        if manual_top and self.n_levels:
            top = self.geometry[-1]
            data.qpos[0] = .5 * (top["x_front_m"] + top["x_back_m"])
            data.qpos[2] = top["tread_top_m"] + .37
        sim.mj.mj_forward(model, data)
        self.tick = 0
        self.previous = np.zeros(16)
        self.force = np.zeros(6)
        self.start_x = float(data.qpos[0])
        self.start_y = float(data.qpos[1])
        self.max_x = self.start_x
        self.substeps = round(self.dt / model.opt.timestep)
        self.leg_low, self.leg_high = self.limits[:12, :2].T
        self.wheel_limit = self.limits[12:, 2]
        raw_low, raw_high = model.actuator_ctrlrange[self.motors].T
        mode = {"90pct": "90pct", "contract_90pct": "90pct",
                "mjcf_ctrlrange": "mjcf_ctrlrange",
                "mujoco_ctrlrange": "mjcf_ctrlrange"}.get(self.torque_clip)
        if mode is None:
            raise ContractError(f"unknown torque_clip mode {self.torque_clip!r}")
        self.torque_clip = mode
        if mode == "mjcf_ctrlrange":
            self.torque_low, self.torque_high = raw_low.copy(), raw_high.copy()
        else:
            budget = np.where(np.arange(16) >= 12, self.torque_budget_nm,
                              .9 * np.maximum(np.abs(raw_low), np.abs(raw_high)))
            self.torque_low = -np.minimum(np.abs(raw_low), budget)
            self.torque_high = np.minimum(np.abs(raw_high), budget)
        self.raw_torque_low, self.raw_torque_high = raw_low, raw_high
        # ---- control wrapper (shared by training and evaluation)
        self.control_chain = ControlChain(self.control)
        self.control_chain.bind(self.dt, self.wheel_limit, self.leg_low, self.leg_high,
                                self.defaults)
        # ---- transmission readback: what unit is actuator_force in?
        self.transmission = read_transmission(model, self.spec["joint_names"], self.motors)
        self.actuator_gear = np.array([row["gear"] for row in self.transmission["actuators"]])
        self.actuator_joint_index = np.array(
            [self.transmission["joint_index_by_actuator"][index] for index in range(16)])
        self.transmission_max_error = 0.
        self.tau_request_peak_nm = 0.
        self.tau_joint_peak_nm = 0.
        self._tau_actuator = np.zeros(16)
        self._tau_joint = np.zeros(16)
        self._dispatch_wheel = np.zeros(4)
        self._dispatch_leg = np.zeros(12)
        self._tau_request = np.zeros(16)
        self.last_tap = {}
        self.tracker = MilestoneTracker(self.n_levels, self.dt)
        self.top_streak = 0
        self.course_completed = False
        self.ever_levels = np.zeros(4, dtype=int)
        self.current_levels = np.zeros(4, dtype=int)
        self.all4_streak = np.zeros(self.n_levels + 1, dtype=int)
        self.longest_all4 = np.zeros(self.n_levels + 1, dtype=int)
        self.max_top_hold_s = 0.
        self.last_tread_loads = np.zeros((4, max(self.n_levels, 1) + 1))
        self.impact_peak = 0.
        self.impact_body = ""
        self.riser_peak = 0.
        self.riser_frames = 0
        self.max_lateral = 0.
        self.min_upright = 1.
        self.approach_min_gap = None
        self.first_fails = {}
        self.term_terms = {"fall": False, "impact": False, "lateral": False}
        self.term_totals = {}
        self.cumulative_reward = 0.
        self.push_frame = (round(self.stress.push_time / self.dt)
                           if (self.stress is not None and flags.get("push")) else None)
        self.leg_saturated_substeps = 0
        self.wheel_saturated_substeps = 0
        self.substep_clip_count = 0
        self.milestone_without_load = 0
        # ---- control-wrapper state (reset rule: cleared on every episode reset)
        self._delay_frames = int(self.control["action_delay_frames"])
        self._wheel_alpha = float(self.control["wheel_target_filter_alpha"])
        self._leg_alpha = float(self.control["leg_target_filter_alpha"])
        self.last_tap = {}
        self.tap_step = 0
        self.rows = []
        self.milestones = []
        self.started_from_bottom = not self.manual
        return self.observe(), {"scene_seed": scene_seed, "stage": self.stage.name,
                                "manual": self.manual, "start_kind": self.start_kind}

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        data = self.data
        q = data.qpos[self.qa].copy()
        q[12:] = (q[12:] + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
        clock = self.gait_clock(self.tick * self.dt)
        obs = np.r_[self.command - (.4, 0., 0.), data.qvel[3:6],
                    -data.xmat[self.base, 6:9], .4, q - self.defaults,
                    data.qvel[self.va], self.previous, clock]
        if obs.shape != (OBS_DIM,):
            raise ContractError(f"observation has {obs.shape[0]} features, expected {OBS_DIM}")
        return obs.astype(np.float32)

    @staticmethod
    def gait_clock(elapsed: float):
        angle = 2. * np.pi * elapsed / GAIT_CLOCK_PERIOD_S
        return (np.sin(angle), np.cos(angle))

    # ----------------------------------------------------------------- reward
    def _reward(self, upright: float):
        data = self.data
        contacts = np.zeros(4, dtype=int)
        tread_loads = np.zeros((4, self.n_levels + 1))
        impact = 0.
        impact_body = ""
        riser = 0.
        riser_contacts = 0
        for index, contact in enumerate(data.contact):
            surface, robot_geom = contact.geom
            if surface not in self.surfaces:
                surface, robot_geom = robot_geom, surface
            if surface not in self.surfaces or robot_geom in self.surfaces:
                continue
            sim.mj.mj_contactForce(self.model, data, index, self.force)
            wheel = self.wheel_index.get(int(self.model.geom_bodyid[robot_geom]))
            if wheel is None:
                if self.force[0] > impact:
                    impact = float(self.force[0])
                    impact_body = self.body_names.get(int(self.model.geom_bodyid[robot_geom]), "?")
            elif abs(contact.frame[2]) >= .7:
                # Vertical normal: a tread contact. Only this can credit a milestone.
                level = self.surfaces[surface]
                tread_loads[wheel, level] += self.force[0] * abs(contact.frame[2])
            else:
                # Horizontal normal: a riser face. A real collision, never a milestone.
                riser = max(riser, float(self.force[0]))
                riser_contacts += 1
        top_loads = tread_loads[:, self.top_index]
        stable = upright >= STABLE_UPRIGHT and impact < self.nonwheel_contact_n
        contacts, newly_reached = self.tracker.update(tread_loads, stable, frame=self.tick)
        for wheel in np.nonzero(contacts)[0]:
            if tread_loads[int(wheel)].max() < SUSTAINED_LOAD_N:
                # Structural invariant: a credit must come from tread load. Riser
                # face contacts and repeated same-tread steps can never satisfy it.
                self.milestone_without_load += 1
        for event in self.tracker.credit_events:
            if event not in self.milestones:
                self.milestones.append(event)
        self.ever_levels = np.maximum(self.ever_levels, self.tracker.levels)
        capped_x = data.qpos[0]
        if self.stage.approach_line_m is not None:
            capped_x = min(capped_x, self.stage.approach_line_m)
        progress = max(0., capped_x - self.max_x)
        self.max_x = max(self.max_x, data.qpos[0])
        if self.geometry:
            gap = float(self.geometry[0]["x_front_m"]) - float(data.qpos[0])
            self.approach_min_gap = gap if self.approach_min_gap is None else min(
                self.approach_min_gap, gap)
        forward_speed = float(data.xmat[self.base, [0, 3, 6]] @ data.qvel[:3])
        tracking = float(np.exp(-((forward_speed - .40) / .15) ** 2))
        stalled = bool(self.tick * self.dt >= 1. and self.max_x - self.start_x < .12)
        gate_ok = (self.stage.approach_line_m is None
                   or data.qpos[0] >= self.stage.approach_line_m)
        top = bool(stable and top_loads.min() >= SUSTAINED_LOAD_N and gate_ok)
        self.top_streak = self.top_streak + 1 if top else 0
        self.max_top_hold_s = max(self.max_top_hold_s, self.top_streak * self.dt)
        self.course_completed |= bool(self.top_streak * self.dt >= TOP_HOLD_S - 1e-9)
        terms = {
            "progress": 8. * progress,
            "tracking": .02 * tracking,
            "newly_reached": 2. * newly_reached,
            "top_hold": .20 * top,
            "top_completion": 4. * float(self.top_streak == round(TOP_HOLD_S / self.dt)),
            "lateral": -.10 * abs(data.qpos[1]),
            "nonwheel_impact": -.010 * max(impact - 15., 0.),
            "stalled": -.03 * stalled,
            "excess_speed": 0.,
        }
        self.impact_peak = max(self.impact_peak, float(impact))
        if impact >= self.impact_peak:
            self.impact_body = impact_body
        self.riser_peak = max(self.riser_peak, float(riser))
        if riser_contacts:
            self.riser_frames += 1
        self.last_tread_loads = tread_loads
        self.current_levels = np.zeros(4, dtype=int)
        for wheel in range(4):
            loaded = np.nonzero(tread_loads[wheel] >= SUSTAINED_LOAD_N)[0]
            self.current_levels[wheel] = int(loaded.max()) if len(loaded) else 0
        for level in range(1, self.n_levels + 1):
            if int(self.current_levels.min()) >= level:
                self.all4_streak[level] += 1
                self.longest_all4[level] = max(self.longest_all4[level],
                                               int(self.all4_streak[level]))
            else:
                self.all4_streak[level] = 0
        unsafe = impact > TRAIN_TERM_IMPACT_N or abs(data.qpos[1]) > TRAIN_TERM_LATERAL_M
        return (terms, contacts, tread_loads, float(impact), impact_body, float(riser),
                riser_contacts, unsafe)

    # ------------------------------------------------------------------- step
    def step(self, action):
        # Keep the caller dtype: the pinned env multiplies a float32 action by a
        # Python float and NumPy 2 keeps that in float32; widening changes the
        # contact-rich trajectory. Parity requirement.
        action = np.asarray(action)
        if action.shape != (ACTION_DIM,):
            raise ContractError(f"action shape {action.shape} != ({ACTION_DIM},)")
        model, data = self.model, self.data
        targets = self.control_chain.targets(action)
        executed = targets["executed"]
        self.previous = executed
        wheel_pre = targets["wheel_pre"]
        wheel_limited = targets["wheel_limited"]
        wheel = targets["wheel"]
        leg_pre = targets["leg_pre"]
        leg = targets["leg"]
        for _ in range(self.substeps):
            q, dq = data.qpos[self.qa], data.qvel[self.va]
            torque = np.r_[self.kp[:12] * (leg - q[:12]) - self.kd[:12] * dq[:12],
                           self.kd[12:] * (wheel - dq[12:])]
            bounded = np.clip(torque, self.torque_low, self.torque_high)
            clipped = np.abs(torque - bounded) > 1e-9
            if clipped.any():
                self.substep_clip_count += int(np.count_nonzero(clipped))
                if clipped[:12].any():
                    self.leg_saturated_substeps += 1
                if clipped[12:].any():
                    self.wheel_saturated_substeps += 1
            data.ctrl[self.motors] = bounded
            self._dispatch_wheel = wheel.copy()
            self._dispatch_leg = leg.copy()
            self._tau_request = torque.copy()
            sim.mj.mj_step(model, data)
        # ---- applied torque: actuator coordinate + DOF space (both logged)
        self._tau_actuator = data.actuator_force[self.motors].copy()
        self._tau_joint = data.qfrc_actuator[self.va].copy()
        self.tau_request_peak_nm = max(self.tau_request_peak_nm,
                                       float(np.max(np.abs(self._tau_request))))
        self.tau_joint_peak_nm = max(self.tau_joint_peak_nm,
                                     float(np.max(np.abs(self._tau_joint))))
        expected = np.zeros(16)
        for index, motor in enumerate(self.motors):
            joint_index = self.actuator_joint_index[index]
            expected[joint_index] += self.actuator_gear[index] * self._tau_actuator[index]
        error = float(np.max(np.abs(self._tau_joint - expected)))
        self.transmission_max_error = max(self.transmission_max_error, error)
        tolerance = 1e-9 * max(1., float(np.max(np.abs(self._tau_joint))))
        if error > tolerance:
            raise ContractError(
                f"transmission inconsistency: max|qfrc_actuator - sum(gear*actuator_force)| = "
                f"{error:.3e} > {tolerance:.3e}")
        rotation = data.xmat[self.base].reshape(3, 3)
        upright = float(rotation[2, 2])
        fallen = bool(upright < FALL_UPRIGHT)
        (terms, contacts, tread_loads, impact, impact_body, riser, riser_contacts,
         unsafe) = self._reward(upright)
        lateral = abs(float(data.qpos[1]))
        self.max_lateral = max(self.max_lateral, lateral)
        self.min_upright = min(self.min_upright, upright)
        terminated = bool(fallen or unsafe)
        if fallen:
            self.term_terms["fall"] = True
            self._note_fail("fall", self.tick, None, upright, FALL_UPRIGHT, "upright")
        if impact > TRAIN_TERM_IMPACT_N:
            self.term_terms["impact"] = True
            self._note_fail("nonwheel_impact", self.tick, impact_body, impact,
                            TRAIN_TERM_IMPACT_N, "impact_n")
        if lateral > TRAIN_TERM_LATERAL_M:
            self.term_terms["lateral"] = True
            self._note_fail("lateral_out", self.tick, None, lateral,
                            TRAIN_TERM_LATERAL_M, "abs_y_m")
        self.tick += 1
        truncated = bool(self.tick * self.dt >= self.stage.episode_s)
        pushed = False
        if self.tick == self.push_frame:
            data.qvel[:2] += self.stress.push_delta_xy
            sim.mj.mj_forward(model, data)
            pushed = True
        total = float(sum(terms.values()) - 2. * terminated)
        self.cumulative_reward += total
        for key, value in terms.items():
            self.term_totals[key] = self.term_totals.get(key, 0.) + value
        if self.trace_enabled:
            self.rows.append({
                "frame": self.tick, "t": round(self.tick * self.dt, 4),
                "x": round(float(data.qpos[0]), 5), "y": round(float(data.qpos[1]), 5),
                "z": round(float(data.qpos[2]), 5), "upright": round(upright, 5),
                "vx_body": round(float(data.xmat[self.base, [0, 3, 6]] @ data.qvel[:3]), 5),
                "levels": [int(v) for v in self.tracker.levels],
                "current": [int(v) for v in self.current_levels],
                "top_streak": int(self.top_streak), "impact_n": round(impact, 3),
                "impact_body": impact_body, "riser_n": round(riser, 2),
                "riser_contacts": riser_contacts, "term": terminated, "trunc": truncated,
                "terms": {key: round(float(value), 6) for key, value in terms.items()},
                "reward": round(total, 6),
            })
        info = {}
        self.tap_step += 1
        self.last_tap = {
            "step_id": int(self.tap_step),
            "episode_id": int(self.episode_id),
            "tick": int(self.tick),
            "scene_seed": int(self.scene_seed),
            "stage": self.stage.name,
            "wheel_target_unit": "rad/s",
            "leg_target_unit": "rad",
            "torque_unit": "Nm",
            "target_wheel_pre_filter": [round(float(value), 6) for value in wheel_pre],
            "target_wheel_clipped": [round(float(value), 6) for value in wheel_limited],
            "target_wheel_dispatch": [round(float(value), 6)
                                      for value in self._dispatch_wheel],
            "target_leg_pos_pre_filter": [round(float(value), 6) for value in leg_pre],
            "target_leg_pos_dispatch": [round(float(value), 6) for value in self._dispatch_leg],
            "tau_request_est_nm": [round(float(value), 6) for value in self._tau_request],
            "tau_request_label": "ESTIMATED",
            "tau_actuator_coord_nm": [round(float(value), 6)
                                      for value in self._tau_actuator],
            "tau_actuator_coord_label": self.transmission["actuator_force_label"],
            "tau_joint_qfrc_nm": [round(float(value), 6) for value in self._tau_joint],
            "tau_joint_qfrc_label": "simulated joint actuator torque (dof space)",
            "torque_quantity": self.transmission["torque_quantity"],
            "torque_budget_nm": float(self.torque_budget_nm),
            "torque_clip_mode": self.torque_clip,
            "wheel_limit_rad_s": [round(float(value), 6) for value in self.wheel_limit],
            "action_clip": float(self.control["action_clip"]),
            "delay_frames": int(self._delay_frames),
        }
        if terminated or truncated:
            record = self.episode_record(terminated=terminated, truncated=truncated)
            if self.curriculum is not None:
                record["curriculum"] = self.curriculum.record(record)
                self.stage = self.curriculum.stage
                self.n_levels = self.stage.levels
                self.top_index = self.n_levels
            info["fs_stair"] = record
        return self.observe(), total, terminated, truncated, info

    # ------------------------------------------------------- failure accounting
    def _note_fail(self, kind: str, frame: int, wheel, value: float, threshold: float,
                   quantity: str):
        if kind in self.first_fails:
            return
        self.first_fails[kind] = {
            "kind": kind, "scene_seed": getattr(self, "scene_seed", None),
            "stage": self.stage.name, "frame": int(frame),
            "t_s": round(frame * self.dt, 4), "wheel": wheel, "quantity": quantity,
            "value": float(value), "threshold": float(threshold),
            "x_m": round(float(self.data.qpos[0]), 5),
            "y_m": round(float(self.data.qpos[1]), 5),
            "levels": [int(v) for v in getattr(self, "ever_levels", np.zeros(4))],
        }

    def failure_kinds(self) -> list:
        kinds = [kind for kind in self.first_fails
                 if kind in ("fall", "nonwheel_impact", "lateral_out")]
        progress = self.max_x - self.start_x
        if not kinds:
            if not self.stage.levels:
                if not self.course_completed:
                    kinds.append("approach_fail" if progress < .12 else "approach_incomplete")
            elif int(self.ever_levels.max()) == 0:
                if progress < .12:
                    kinds.append("approach_fail")
                else:
                    kinds.append("riser_block")
            elif (int(self.ever_levels[:2].max()) >= 1
                  and int(self.ever_levels[2:].min()) == 0):
                kinds.append("rear_wheel_follow_fail")
            elif int(self.ever_levels.min()) >= self.stage.levels and not self.course_completed:
                kinds.append("top_hold_fail")
            elif not self.course_completed:
                kinds.append("partial_progress")
        return kinds

    def episode_record(self, *, terminated: bool, truncated: bool) -> dict:
        kinds = self.failure_kinds()
        if "approach_fail" in kinds:
            self._note_fail("approach_fail", self.tick, None, self.max_x - self.start_x,
                            .12, "progress_m")
        if "riser_block" in kinds:
            self._note_fail("riser_block", self.tick, "FL/FR", self.riser_peak,
                            self.nonwheel_contact_n, "riser_contact_n")
        if "rear_wheel_follow_fail" in kinds:
            rear = int(np.argmin(self.ever_levels[2:])) + 2
            self._note_fail("rear_wheel_follow_fail", self.tick, WHEEL_LABELS[rear],
                            float(self.ever_levels[rear]), 1., "ever_level")
        if "top_hold_fail" in kinds:
            self._note_fail("top_hold_fail", self.tick, "all", self.max_top_hold_s,
                            TOP_HOLD_S, "top_hold_s")
        if "approach_incomplete" in kinds:
            self._note_fail("approach_incomplete", self.tick, None,
                            float(self.data.qpos[0]), float(self.stage.approach_line_m or 0.),
                            "x_m")
        if "partial_progress" in kinds:
            self._note_fail("partial_progress", self.tick, "all",
                            float(self.ever_levels.min()), float(self.stage.levels),
                            "min_ever_level")
        hard = bool(terminated)
        zero_progress = (int(self.ever_levels.max()) == 0
                         and (self.max_x - self.start_x) < .12)
        all_wheels_top = bool(self.n_levels and int(self.ever_levels.min()) >= self.n_levels)
        longest_all4_top_s = (float(self.longest_all4[self.top_index]) * self.dt
                              if self.n_levels else 0.)
        complete = promotion_complete(
            self.promotion["signal"], top_stable=bool(self.course_completed),
            all_wheels_top=all_wheels_top, longest_all4_top_s=longest_all4_top_s,
            min_top_hold_s=float(self.promotion["min_top_hold_s"]),
            min_all4_top_s=float(self.promotion["min_all4_top_s"]),
            flat_stage=self.n_levels == 0)
        record = {
            "kind": "episode", "scene_seed": self.scene_seed, "stage": self.stage.name,
            "start_kind": self.start_kind,
            "started_from_bottom": bool(self.started_from_bottom),
            "manual": bool(self.manual),
            "promotion_signal": self.promotion["signal"],
            "eligible_for_promotion": bool(self.started_from_bottom and not self.manual
                                           and complete and not hard),
            "complete": bool(complete),
            "hard_protection": hard, "zero_progress": bool(zero_progress),
            "failure_kinds": kinds,
            # three accounting modes, always logged separately
            "ever_levels": [int(v) for v in self.ever_levels],
            "current_levels": [int(v) for v in self.current_levels],
            "final_levels": [int(v) for v in self.tracker.levels],
            "top_stable": bool(self.course_completed),
            "all_wheels_top": all_wheels_top,
            "longest_all4_top_s": round(longest_all4_top_s, 4),
            "top_hold_s": round(float(self.max_top_hold_s), 4),
            "max_forward_m": round(float(self.max_x - self.start_x), 5),
            "approach_min_gap_m": (None if self.approach_min_gap is None
                                   else round(self.approach_min_gap, 5)),
            "max_lateral_m": round(float(self.max_lateral), 5),
            "min_upright": round(float(self.min_upright), 5),
            "peak_impact_n": round(float(self.impact_peak), 3),
            "riser_peak_n": round(float(self.riser_peak), 3),
            "riser_frames": int(self.riser_frames),
            "steps": int(self.tick), "t_s": round(self.tick * self.dt, 4),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "reward": round(float(self.cumulative_reward), 6),
            "reward_terms": {key: round(float(value), 6)
                             for key, value in self.term_totals.items()},
            "milestones": list(self.milestones),
            "first_fails": list(self.first_fails.values()),
            "torque_clip": self.torque_clip,
            "torque_quantity": self.transmission["torque_quantity"],
            "torque_quantity_label": self.transmission["label"],
            "transmission": {
                "direct_motor_gear1": self.transmission["direct_motor_gear1"],
                "one_actuator_per_joint": self.transmission["one_actuator_per_joint"],
                "actuators_per_joint": self.transmission["actuators_per_joint"],
                "transmission_max_error": self.transmission_max_error,
                "actuators": self.transmission["actuators"],
            },
            "tau_request_peak_nm": round(float(self.tau_request_peak_nm), 6),
            "tau_joint_peak_nm": round(float(self.tau_joint_peak_nm), 6),
            "leg_saturated_substeps": int(self.leg_saturated_substeps),
            "wheel_saturated_substeps": int(self.wheel_saturated_substeps),
            "substep_clip_count": int(self.substep_clip_count),
            "milestone_without_load": int(self.milestone_without_load),
        }
        return record
