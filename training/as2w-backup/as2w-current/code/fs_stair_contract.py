"""FS stair line: control-contract definition, fingerprint and version guard.

Stdlib only on purpose: the fingerprint must be computable offline (dry-run,
selfcheck, CI, a laptop) without importing MuJoCo, torch or SB3.

Contract rule (also in fs_stair_README.md)
------------------------------------------
Any change to the filter position/alpha, the rate limit, the action delay, the
history field semantics, the action clip/scales, the torque dispatch or the
initial-state distribution changes ``contract_fingerprint``.  If the stored
fingerprint no longer matches, the runner prints

    CONTROL_CONTRACT_CHANGED -> bump control_contract_version and redo replay +
    baseline regression

and refuses ``--run`` unless ``--accept-contract-change`` is passed explicitly
(which is recorded in the run evidence).
"""

from __future__ import annotations

import hashlib
import json

CONTROL_KEYS = (
    "action_clip",
    "leg_action_scale",
    "wheel_action_scale",
    "wheel_target_filter_alpha",
    "wheel_target_rate_limit_per_s",
    "leg_target_filter_alpha",
    "action_delay_frames",
    "reset_rule",
    "history_reset_fill",
    "torque_clip_mode",
    "torque_budget_nm",
)
HISTORY_KEYS = (
    "mode",
    "frames",
    "frame_stride_steps",
    "span_ms",
    "order",
    "reset_fill",
    "current_frame_slot",
)
INITIAL_STATE_KEYS = (
    "base_x_uniform_m",
    "base_y_uniform_m",
    "yaw_uniform_rad",
    "joint_offset_uniform_rad",
    "base_height_m",
    "start_back_flat_m",
    "scene_seed_range",
)
DISPATCH_KEYS = (
    "wheel_target_unit",
    "dispatch_unit",
    "previous_action_semantics",
)

JOINTS = ("FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh",
          "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
          "FL_foot", "FR_foot", "RL_foot", "RR_foot")

# The 60-dim observation, field by field, in the exact order the policy sees it.
# It must equal the config's observation.components concatenation verbatim and it
# participates in contract_fingerprint, so any reorder is a contract change.
FIELD_ORDER = (
    ("command_error_vx", "command_error_vy", "command_error_wz")
    + ("base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z")
    + ("projected_gravity_x", "projected_gravity_y", "projected_gravity_z")
    + ("height_reference",)
    + tuple(f"joint_pos_{name}" for name in JOINTS)
    + tuple(f"joint_vel_{name}" for name in JOINTS)
    + tuple(f"previous_action_{index}" for index in range(16))
    + ("gait_clock_sin", "gait_clock_cos")
)

CHANGE_MESSAGE = ("CONTROL_CONTRACT_CHANGED -> bump control_contract_version and "
                  "redo replay + baseline regression")
ACCEPT_MESSAGE = ("CONTROL_CONTRACT_CHANGE_ACCEPTED -> the operator accepted a "
                  "contract change without bumping the version; replay and baseline "
                  "regression are still required")


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_payload(control: dict, history: dict, initial_state: dict,
                  dispatch: dict) -> dict:
    missing = [key for key in CONTROL_KEYS if key not in control]
    missing += [f"history.{key}" for key in HISTORY_KEYS if key not in history]
    missing += [f"initial_state.{key}" for key in INITIAL_STATE_KEYS
                if key not in initial_state]
    missing += [f"dispatch.{key}" for key in DISPATCH_KEYS if key not in dispatch]
    if missing:
        raise ValueError(f"control-contract fields missing: {missing}")
    if len(FIELD_ORDER) != 60:
        raise ValueError(f"FIELD_ORDER has {len(FIELD_ORDER)} entries, expected 60")
    return {
        "control": {key: control[key] for key in CONTROL_KEYS},
        "history": {key: history[key] for key in HISTORY_KEYS},
        "initial_state": {key: initial_state[key] for key in INITIAL_STATE_KEYS},
        "dispatch": {key: dispatch[key] for key in DISPATCH_KEYS},
        "observation": {"single_frame_dim": len(FIELD_ORDER),
                        "field_order": list(FIELD_ORDER)},
    }


def contract_fingerprint(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def verify(stored_fingerprint: str, stored_version: str, payload: dict,
           *, accept_change: bool = False, will_run: bool = False) -> dict:
    """Compare the stored fingerprint with the one derived from the config."""
    computed = contract_fingerprint(payload)
    changed = computed != stored_fingerprint
    return {
        "version": stored_version,
        "stored_fingerprint": stored_fingerprint,
        "computed_fingerprint": computed,
        "payload": payload,
        "changed": bool(changed),
        "message": CHANGE_MESSAGE if changed else None,
        "accepted": bool(changed and accept_change),
        "run_refused": bool(changed and will_run and not accept_change),
        "accept_message": ACCEPT_MESSAGE if changed and accept_change else None,
    }
