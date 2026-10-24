"""Allowlisted native reward scales; plain data, shared by CLI and web preflight."""
import json
import math

NAMES=frozenset('termination ang_vel_x ang_vel_y ang_vel_z lin_vel_z orientation_control feet_height_before_backflip height_control default_pose actions_symmetry gravity_y feet_distance action_rate action_jerk dof_vel_limits rotation_progress flip_completion flip_success recovery_upright recovery_height recovery_default_pose recovery_still recovery_feet_contact rear_leg_action_rate rear_leg_symmetry recovery_dof_velocity head_clearance undesired_body_contact dof_pos_limits joint_target_limits head_contact landing_impact'.split())


def parse_scales(text):
    data=json.loads(text)
    if not isinstance(data,dict) or set(data)-NAMES:raise ValueError('Unknown native flip reward scale')
    for key,value in data.items():
        if type(value) not in (int,float) or not math.isfinite(value) or not -1000<=value<=1000:
            raise ValueError('Native flip reward scales must be finite numbers in [-1000,1000]: '+key)
    return data


def apply_scales(cfg,overrides):
    values=parse_scales(json.dumps(overrides))
    if set(values)-set(cfg.rewards.scales):raise ValueError('Pinned native reward schema differs from configured overrides')
    cfg.rewards.scales.update(values)
    cfg.reward_scale_overrides=dict(values)
