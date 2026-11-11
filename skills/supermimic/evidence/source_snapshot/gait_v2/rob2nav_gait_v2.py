"""Bounded posture/foot-slip refinement of the blind velocity SuperMimic task.

The same analytic function evaluates real and imagined transitions. Contact in
this reward is a smooth height proxy, not a learned force or acceptance test.
Actor inputs remain unchanged; physical gait audits use raw collision contacts.
"""
from dataclasses import asdict, dataclass
import re

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from go2_supermimic.supermimic import state as st
import rob2nav_velocity as velocity
from rob2nav_reset_targets import install

TASK = "Go2-Velocity-Gait-V2"


@dataclass(frozen=True)
class GaitSettings:
    body_height_target: float = .29
    body_height_weight: float = .4
    stance_slip_weight: float = .15
    mapped_target_rate_weight: float = 1.
    mapped_target_rate_margin: float = .6
    knee_clearance_weight: float = .1
    support_weight: float = .1
    lateral_symmetry_weight: float = .03
    joint_speed_weight: float = .1


def velocity_limits(env):
    robot = env.scene["robot"]
    names = env.action_manager.get_term("joint_pos")._target_names
    limits = []
    for name in names:
        matches = [a for a in robot.cfg.articulation.actuators if any(re.fullmatch(p, name) for p in a.target_names_expr)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous actuator: {name}")
        limits.append(matches[0].velocity_limit)
    return torch.tensor(limits, device=env.device)


def reward_terms(s0, s1, pose0, pose1, command, action, previous_action, soft_limits, default_q, scale, limits, dt, settings):
    terms = velocity.reward_terms(s1, command, action, previous_action, soft_limits, default_q)
    points0 = st.limb_points_world(s0, pose0)
    points1 = st.limb_points_world(s1, pose1)
    foot_velocity = (points1["feet"] - points0["feet"])/dt
    soft_contact = torch.sigmoid((st.FOOT_RADIUS + .01 - points1["feet"][..., 2])/.006)
    moving = (command.abs().sum(-1) >= .05).float()
    minimum_support = 3.5 - 2.*moving
    mapped_rate = (action-previous_action)*scale/dt/limits
    feet_y = s1[:, st.FEET].reshape(-1, 4, 3)[..., 1]
    straight = ((command[:, 1].abs() < .05) & (command[:, 2].abs() < .1)).float()
    terms.update({
        "body_height": ((s1[:, st.Z].squeeze(-1)-settings.body_height_target)/.05).square(),
        "stance_slip": (soft_contact*foot_velocity[..., :2].square().sum(-1)/.25**2).mean(-1),
        "mapped_target_rate": (mapped_rate.abs()-settings.mapped_target_rate_margin).clamp(min=0).square().mean(-1),
        "knee_clearance": ((.08-points1["knees"][..., 2]).clamp(min=0)/.08).square().mean(-1),
        "support_shortfall": (minimum_support-soft_contact.sum(-1)).clamp(min=0).square(),
        "lateral_symmetry": ((feet_y[:, 0]+feet_y[:, 1]).square()+(feet_y[:, 2]+feet_y[:, 3]).square())/.03**2*straight,
        "joint_speed_margin": (s1[:, st.QD].abs()/limits-.8).clamp(min=0).square().mean(-1),
    })
    return terms


def reward(s0, s1, pose0, pose1, command, action, previous_action, soft_limits, default_q, scale, limits, dt, settings):
    terms = reward_terms(s0, s1, pose0, pose1, command, action, previous_action, soft_limits, default_q, scale, limits, dt, settings)
    weights = dict(velocity.WEIGHTS)
    weights.update(body_height=-settings.body_height_weight, stance_slip=-settings.stance_slip_weight,
                   mapped_target_rate=-settings.mapped_target_rate_weight, knee_clearance=-settings.knee_clearance_weight,
                   support_shortfall=-settings.support_weight, lateral_symmetry=-settings.lateral_symmetry_weight,
                   joint_speed_margin=-settings.joint_speed_weight)
    return sum(weights[name]*value for name, value in terms.items()), terms


def real_reward(env, settings):
    if not hasattr(env, "_gait_constants"):
        env._gait_constants = st.robot_consts(env)
        env._gait_velocity_limits = velocity_limits(env)
    c = env._gait_constants
    s1, pose1 = st.robot_state(env), st.planar_pose(env)
    s0 = getattr(env, "_gait_previous_state", s1)
    pose0 = getattr(env, "_gait_previous_pose", pose1)
    value, _ = reward(s0, s1, pose0, pose1, env.command_manager.get_term("target").gen.chase_vel,
                      env.action_manager.action, env.action_manager.prev_action, c.soft_limits,
                      c.default_q, c.action_scale, env._gait_velocity_limits, env.step_dt, settings)
    return value


def register(settings):
    def make_cfg(play):
        cfg = install(velocity.flat_cfg(play))
        cfg.rewards["velocity"] = RewardTermCfg(func=real_reward, weight=1., params={"settings": settings})
        return cfg
    register_mjlab_task(task_id=TASK, env_cfg=make_cfg(False), play_env_cfg=make_cfg(True), rl_cfg=RslRlOnPolicyRunnerCfg())


def imagination_class(settings):
    class GaitImagination(velocity.VelocityImagination):
        def _step(self, s, pose, a_prev, hist, gst, xi, enc_bias, k, dt, actor, wm):
            x = self.critic_input(s, pose, gst, xi)
            action = actor(hist)
            s1, dpose = wm(s, action, a_prev, xi, self.relief(pose, gst, self.cfg.sensor_grad))
            pose1 = st.apply_pose_delta(pose, dpose)
            # The velocity-limit order is captured from the actual training actuator mapping.
            value, _ = reward(s, s1, pose, pose1, gst["chase_vel"], action, a_prev,
                              k["soft_limits"], k["default_q"], k["action_scale"], self.gait_velocity_limits,
                              self.cfg.dt, settings)
            done = self.fall_prob(s1)
            value = (value + self.cfg.weights.terminated*done)*self.cfg.dt
            frame = st.obs_frame(s1, action, velocity.command_observation(gst["chase_vel"]), enc_bias, noise=True)
            history = st.push_history(hist, frame, self.h, self.terms)
            return s1, pose1, action, history, gst, value, done, x
    return GaitImagination
