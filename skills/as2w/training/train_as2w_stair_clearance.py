"""Train a wheel-assisted fall-recovery policy on the AS2-W URDF in Isaac Lab."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=413)
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--resume-checkpoint", type=Path, help="Resume PPO with optimizer state")
parser.add_argument("--initialize-checkpoint", type=Path,
                    help="Initialize model weights and start a fresh PPO run")
parser.add_argument("--bounded-recovery", action="store_true",
                    help="Limit motor targets and reward a settled four-wheel stance")
parser.add_argument("--settling-recovery", action="store_true",
                    help="Bound action history and train braking after righting")
parser.add_argument("--righting-resets", action="store_true",
                    help="Sample upright, side and back drops alongside broad rotations")
parser.add_argument("--wheel-scale", type=float, default=6.,
                    help="Wheel velocity target in rad/s per unit of policy action")
parser.add_argument("--full-leg-range", action="store_true",
                    help="Map clipped actions over the entire official leg joint range")
parser.add_argument("--stance-height-gate", action="store_true",
                    help="Scale post-righting stance rewards by chassis height")
parser.add_argument("--flat-domain-randomization", action="store_true",
                    help="Randomize friction, mass, gains and push disturbances on flat ground")
parser.add_argument("--mixed-terrain", action="store_true",
                    help="Train recovery on flat, sloped and rough ground")
parser.add_argument("--blind-stairs", action="store_true",
                    help="Train forward traversal with only body and joint observations")
parser.add_argument("--stair-clearance-reward", action="store_true",
                    help="Shape wheel lift and landing using terrain queries during training")
parser.add_argument("--shared-ground-friction", action="store_true",
                    help="Use a shared static and dynamic contact coefficient per environment")
parser.add_argument("--mirror-recovery", action="store_true",
                    help="Augment PPO with left-right reflected AS2-W recovery rollouts")
parser.add_argument("--mirror-loss-only", action="store_true",
                    help="Use the mirror loss while keeping PPO ratios on sampled transitions")
parser.add_argument("--conservative-ppo", action="store_true",
                    help="Use a fixed small learning rate and a tighter PPO clip")
parser.add_argument("--stance-streak-reward", action="store_true",
                    help="Reward uninterrupted stable four-wheel support during recovery")
parser.add_argument("--export-replay-contract", type=Path,
                    help="Export the instantiated observation and action order, then exit")
parser.add_argument("--preview-checkpoint", type=Path)
parser.add_argument("--audit-observations", action="store_true",
                    help="Print observation parity diagnostics for one recovery episode")
parser.add_argument("--audit-step", type=int, choices=range(100, 299), action="append",
                    metavar="100..298", help="Capture a control step in audit mode (repeatable, default 100)")
parser.add_argument("--audit-wheel-spin", action="store_true",
                    help="Apply fixed wheel commands to diagnose multi-turn angle reporting")
parser.add_argument("--isaac-screen", action="store_true",
                    help="Measure recovery by actual handoff orientation in the training simulator")
parser.add_argument("--isaac-fixed-screen", action="store_true",
                    help="Use the same initial poses and joint offsets as the 16 MuJoCo scenarios")
parser.add_argument("--preview-video", type=Path, default=Path("replay/fall-recovery-r01.mp4"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.stair_clearance_reward and not args.blind_stairs:
    parser.error("--stair-clearance-reward requires --blind-stairs")
if args.blind_stairs:
    if not args.mixed_terrain or args.righting_resets or args.stance_streak_reward or args.audit_observations:
        parser.error("--blind-stairs requires --mixed-terrain and a traversal reset/reward")
    args.settling_recovery = True
if args.mixed_terrain and (not args.flat_domain_randomization or args.isaac_screen):
    parser.error("--mixed-terrain requires dynamics randomization and uses independent MuJoCo screening")
if args.conservative_ppo and args.resume_checkpoint:
    parser.error("--conservative-ppo needs fresh optimizer state; use --initialize-checkpoint")
if args.mirror_loss_only and not args.mirror_recovery:
    parser.error("--mirror-loss-only requires --mirror-recovery")
if args.shared_ground_friction and not args.flat_domain_randomization:
    parser.error("--shared-ground-friction requires --flat-domain-randomization")
if args.resume_checkpoint and args.initialize_checkpoint:
    parser.error("Choose either --resume-checkpoint or --initialize-checkpoint")
if not 0. < args.wheel_scale < float("inf"):
    parser.error("--wheel-scale must be positive and finite")
if args.righting_resets:
    args.settling_recovery = True
if args.full_leg_range and not args.settling_recovery:
    parser.error("--full-leg-range requires --settling-recovery or --righting-resets")
if args.stance_height_gate and not args.settling_recovery:
    parser.error("--stance-height-gate requires --settling-recovery or --righting-resets")
if args.stance_streak_reward and not args.settling_recovery:
    parser.error("--stance-streak-reward requires --settling-recovery or --righting-resets")
if args.settling_recovery:
    args.bounded_recovery = True
if (args.audit_observations or args.isaac_screen) and not args.preview_checkpoint:
    parser.error("Diagnostics require --preview-checkpoint")
if args.audit_wheel_spin and not args.audit_observations:
    parser.error("--audit-wheel-spin requires --audit-observations")
if args.isaac_fixed_screen and (not args.isaac_screen or args.num_envs != 16):
    parser.error("--isaac-fixed-screen requires --isaac-screen and --num-envs 16")
if args.preview_checkpoint and not (args.audit_observations or args.isaac_screen):
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import trimesh
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg, MeshPlaneTerrainCfg, TerrainGeneratorCfg,
)
from isaaclab.utils.warp import raycast_mesh
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parent
URDF = ROOT / "unitree_ros/robots/as2w_description/as2w.urdf"
LEGS = ".*_(hip|thigh|calf)"
WHEELS = ".*_foot"
SETTLE_SECONDS = 0. if args.blind_stairs else 2.
RECOVERY_SECONDS = 9. if args.blind_stairs else 4.
FIXED_SCREEN_POSES = {"back": np.pi, "left": np.pi / 2, "right": -np.pi / 2, "belly": 0.}
JOINT_ORDER = tuple(f"{leg}_{joint}" for joint in ("hip", "thigh", "calf", "foot")
                    for leg in ("FL", "FR", "RL", "RR"))
MIRROR_JOINTS = [index ^ 1 for index in range(16)]


def mirror_joint_channels(channels):
    reflected = channels[..., MIRROR_JOINTS]
    reflected[..., :4].neg_()
    return reflected


def augment_recovery_mirror(env, obs=None, actions=None):
    augmented_obs = None
    augmented_actions = None
    if obs is not None:
        mirrored = obs.clone()
        for group in mirrored.keys():
            values = mirrored[group]
            if values.shape[-1] != 58:
                raise ValueError(f"Unsupported recovery observation group {group}: {values.shape}")
            # Polar vectors reflect across y; angular velocity is axial.
            values[..., [1, 3, 5, 7]] = -values[..., [1, 3, 5, 7]]
            for start in (10, 26, 42):
                values[..., start:start + 16] = mirror_joint_channels(values[..., start:start + 16])
        augmented_obs = torch.cat((obs, mirrored), dim=0)
    if actions is not None:
        augmented_actions = torch.cat((actions, mirror_joint_channels(actions)), dim=0)
    return augmented_obs, augmented_actions


def fixed_drop_cases():
    for pose, center in FIXED_SCREEN_POSES.items():
        for seed in range(12001, 12005):
            rng = np.random.default_rng(seed)
            rng.uniform(.55, .85)  # Match the MuJoCo generator's draw order.
            roll = center + rng.uniform(-.12, .12)
            yield pose, seed, roll, rng.uniform(-.55, .55, size=16)


def reset_fixed_drop_root(env, env_ids):
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    rolls = torch.tensor([roll for _, _, roll, _ in fixed_drop_cases()],
                         device=pose.device, dtype=pose.dtype)[env_ids]
    pose[:, 3] = (rolls / 2.).cos()
    pose[:, 4] = (rolls / 2.).sin()
    pose[:, 5:7] = 0.
    robot.write_root_pose_to_sim(pose, env_ids)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[env_ids, 7:]), env_ids)


def reset_fixed_drop_joints(env, env_ids):
    robot = env.scene["robot"]
    default = robot.data.default_joint_pos[env_ids]
    offsets = torch.as_tensor(np.stack([offset for _, _, _, offset in fixed_drop_cases()]),
                              device=default.device, dtype=default.dtype)[env_ids]
    positions = default + offsets
    bounds = robot.data.default_joint_pos_limits[env_ids, :12]
    positions[:, :12] = positions[:, :12].clamp(bounds[..., 0], bounds[..., 1])
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)


def reset_righting_poses(env, env_ids, pose_range, velocity_range):
    groups = torch.randint(5, (len(env_ids),), device=env_ids.device)
    for group, center in enumerate((0., torch.pi / 2, -torch.pi / 2, torch.pi)):
        selected = env_ids[groups == group]
        if not selected.numel():
            continue
        targeted_range = {
            **pose_range,
            "roll": (center - .3, center + .3),
            "pitch": (-.25, .25),
        }
        mdp.reset_root_state_uniform(env, selected, targeted_range, velocity_range)
    broad = env_ids[groups == 4]
    if broad.numel():
        mdp.reset_root_state_uniform(env, broad, pose_range, velocity_range)


def push_during_recovery(env, env_ids, max_delta_speed):
    active = env.episode_length_buf[env_ids] * env.step_dt >= SETTLE_SECONDS
    selected = env_ids[active]
    if not selected.numel():
        return
    robot = env.scene["robot"]
    velocity = robot.data.root_vel_w[selected].clone()
    perturbation = torch.empty_like(velocity[:, :2]).uniform_(-max_delta_speed, max_delta_speed)
    velocity[:, :2] += perturbation
    robot.write_root_velocity_to_sim(velocity, selected)


def randomize_shared_friction(env, env_ids, friction_range):
    view = env.scene["robot"].root_physx_view
    materials = view.get_material_properties()
    env_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    buckets = torch.linspace(*friction_range, 64, dtype=materials.dtype)
    samples = buckets[torch.randint(len(buckets), (len(env_ids),), device="cpu")]
    materials[env_ids, :, :2] = samples[:, None, None]
    materials[env_ids, :, 2] = 0.
    view.set_material_properties(materials, env_ids)
    print(f"AS2W_SHARED_FRICTION min={samples.min():.3f} max={samples.max():.3f}", flush=True)


def recovery_slope(difficulty, cfg):
    angle = np.interp(difficulty, (0., 1.), cfg.slope_range)
    width, depth = cfg.size
    rise = width / 2. * np.tan(angle)
    vertices = [[0., 0., -rise], [width, 0., rise], [width, depth, rise], [0., depth, -rise]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    return [mesh], np.array([width / 2., depth / 2., 0.])


def traversal_stairs(difficulty, cfg):
    width, depth = cfg.size
    rise = .045 + .04 * difficulty
    tread = .37 - .07 * difficulty
    edges = [0.] + [2.5 + index * tread for index in range(5)] + [width]
    meshes = []
    for level, (start, end) in enumerate(zip(edges, edges[1:])):
        top = level * rise
        meshes.append(trimesh.creation.box(
            extents=(end - start, depth, top + .15),
            transform=trimesh.transformations.translation_matrix(
                ((start + end) / 2., depth / 2., (top - .15) / 2.)),
        ))
    return meshes, np.array([1.8, depth / 2., 0.])


def configure_mixed_terrain(cfg):
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.max_init_terrain_level = 2
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=cfg.seed, size=(6., 6.), border_width=8., num_rows=3, num_cols=20,
        curriculum=True, horizontal_scale=.075, vertical_scale=.005,
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(proportion=.5),
            "up": HfPyramidSlopedTerrainCfg(
                function=recovery_slope, proportion=.15, slope_range=(.07, .14),
            ),
            "down": HfPyramidSlopedTerrainCfg(
                function=recovery_slope, proportion=.15, slope_range=(-.07, -.14),
            ),
            "rough": HfRandomUniformTerrainCfg(
                proportion=.2, noise_range=(-.02, .02), noise_step=.005, downsampled_scale=.15,
            ),
        },
    )
    if args.blind_stairs:
        generator = cfg.scene.terrain.terrain_generator
        generator.size = (8., 6.)
        generator.num_cols = 40
        generator.sub_terrains["up"].proportion = .075
        generator.sub_terrains["down"].proportion = .075
        generator.sub_terrains["rough"].proportion = .15
        generator.sub_terrains["stairs"] = MeshPlaneTerrainCfg(
            proportion=.2, function=traversal_stairs)
    # The downward ray is used in the reward. The policy observation contract is unchanged.
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 20.)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(0., 0.)),
        mesh_prim_paths=["/World/ground"],
    )


def configure_flat_randomization(cfg):
    cfg.events.physics_material.params["static_friction_range"] = (.95, 1.15)
    cfg.events.physics_material.params["dynamic_friction_range"] = (.45, .95)
    if args.shared_ground_friction:
        cfg.events.physics_material = EventTermCfg(
            func=randomize_shared_friction, mode="startup", params={"friction_range": (.45, .95)},
        )
    cfg.events.add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (.9, 1.1),
            "operation": "scale", "recompute_inertia": True,
        })
    cfg.events.actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains, mode="startup", params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (.85, 1.15),
            "damping_distribution_params": (.85, 1.15),
            "operation": "scale", "distribution": "uniform",
        })
    cfg.events.push_robot = EventTermCfg(
        func=push_during_recovery, mode="interval", interval_range_s=(2.8, 4.5),
        params={"max_delta_speed": .15},
    )


class RecoveryLegAction(mdp.JointPositionAction):
    def apply_actions(self):
        passive = (self._env.episode_length_buf * self._env.step_dt < SETTLE_SECONDS)[:, None]
        pos = self._asset.data.joint_pos[:, self._joint_ids]
        vel = self._asset.data.joint_vel[:, self._joint_ids]
        target = self.processed_actions
        if args.bounded_recovery:
            bounds = self._asset.data.default_joint_pos_limits[:, self._joint_ids]
            if args.full_leg_range:
                default = self._asset.data.default_joint_pos[:, self._joint_ids]
                action = self.raw_actions / 6.
                span = torch.where(action >= 0., bounds[..., 1] - default, default - bounds[..., 0])
                target = default + action * span
            target = target.clamp(bounds[..., 0], bounds[..., 1])
        self._asset.set_joint_position_target(torch.where(passive, pos, target),
                                              joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(torch.where(passive, vel, 0.), joint_ids=self._joint_ids)


class RecoveryWheelAction(mdp.JointVelocityAction):
    def apply_actions(self):
        passive = (self._env.episode_length_buf * self._env.step_dt < SETTLE_SECONDS)[:, None]
        vel = self._asset.data.joint_vel[:, self._joint_ids]
        target = self.processed_actions
        if args.bounded_recovery:
            limit = self._asset.data.joint_vel_limits[:, self._joint_ids]
            target = target.clamp(-limit, limit)
        self._asset.set_joint_velocity_target(torch.where(passive, vel, target),
                                              joint_ids=self._joint_ids)


def recovery_phase(env):
    t = env.episode_length_buf * env.step_dt
    return ((t - SETTLE_SECONDS) / RECOVERY_SECONDS).clamp(0., 1.)


def traversal_reward(env, wheel_cfg, impact_cfg):
    robot = env.scene["robot"].data
    velocity = robot.root_lin_vel_w
    yaw_rate = robot.root_ang_vel_b[:, 2]
    upright = (-robot.projected_gravity_b[:, 2]).clamp(0., 1.)
    ray_z = env.scene["height_scanner"].data.ray_hits_w[:, 0, 2]
    ground_z = torch.where(torch.isfinite(ray_z), ray_z, env.scene.env_origins[:, 2])
    height = robot.root_pos_w[:, 2] - ground_z
    support = ((height - .18) / .14).clamp(0., 1.) * upright.square()
    forces = env.scene[wheel_cfg.name].data.net_forces_w
    support_fraction = (forces[:, wheel_cfg.body_ids, 2] >= 8.).float().mean(dim=1)
    impact = forces[:, impact_cfg.body_ids].norm(dim=-1).amax(dim=1)
    tracking = torch.exp(-((velocity[:, 0] - .4).square() + velocity[:, 1].square()) / .12)
    return (support * (3. * tracking + velocity[:, 0].clamp(-.4, .65))
            + upright + .2 * support_fraction - .15 * yaw_rate.square() - .002 * impact.clamp(max=1200.)
            - .012 * env.action_manager.action.square().mean(dim=1))


def stair_clearance_reward(env, wheel_cfg):
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    stair_fraction = generator.sub_terrains["stairs"].proportion
    # Stairs occupy the final columns in the terrain generator's curriculum.
    stair_env = terrain.terrain_types / generator.num_cols + .001 >= 1. - stair_fraction
    robot = env.scene[wheel_cfg.name].data
    wheel_pos = robot.body_pos_w[:, wheel_cfg.body_ids]
    origins = torch.cat((wheel_pos, wheel_pos), dim=1)
    origins[:, 4:, 0] += .12
    origins[..., 2] += 1.
    directions = torch.zeros_like(origins)
    directions[..., 2] = -1.
    scanner = env.scene["height_scanner"]
    mesh = RayCaster.meshes[scanner.cfg.mesh_prim_paths[0]]
    hits = raycast_mesh(origins, directions, mesh, max_dist=2.)[0][..., 2]
    under, ahead = hits.split(4, dim=1)
    valid = torch.isfinite(under) & torch.isfinite(ahead)
    ground_z = terrain.env_origins[:, 2:3]
    under = torch.where(valid, under, ground_z)
    ahead = torch.where(valid, ahead, under)
    rise = ahead - under
    wheel_clearance = wheel_pos[..., 2] - .086 - under
    lift = (wheel_clearance / rise.clamp_min(.025)).clamp(0., 1.) * (rise > .025)
    landed = ((under - ground_z) / .4).clamp(0., 1.) * (wheel_clearance.abs() < .025)
    upright = (-robot.projected_gravity_b[:, 2]).clamp(0., 1.)
    return (1.5 * lift + 2. * landed).mean(dim=1) * upright.square() * stair_env


def recovery_reward(env, wheel_cfg, impact_cfg):
    robot = env.scene["robot"]
    sensor = env.scene.sensors[wheel_cfg.name]
    phase = recovery_phase(env)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)
    ground_z = env.scene.env_origins[:, 2]
    if args.mixed_terrain:
        ray_z = env.scene["height_scanner"].data.ray_hits_w[:, 0, 2]
        ground_z = torch.where(torch.isfinite(ray_z), ray_z, ground_z)
    height = robot.data.root_pos_w[:, 2] - ground_z
    pose_error = (robot.data.joint_pos[:, :12] - robot.data.default_joint_pos[:, :12]).square().mean(1)
    wheel_support = (sensor.data.net_forces_w[:, wheel_cfg.body_ids, 2] > 8.).all(dim=1)
    impact = sensor.data.net_forces_w[:, impact_cfg.body_ids].norm(dim=-1).amax(dim=1)
    standing = torch.exp(-pose_error / .25) * torch.exp(-((height - .36) / .12).square())
    settled = torch.exp(-robot.data.root_ang_vel_b.square().sum(1))
    active = env.episode_length_buf * env.step_dt >= SETTLE_SECONDS
    if args.settling_recovery:
        vertical_load = sensor.data.net_forces_w[:, wheel_cfg.body_ids, 2]
        support = (vertical_load / 8.).clamp(0., 1.)
        linear_speed = robot.data.root_lin_vel_b.norm(dim=1)
        angular_speed = robot.data.root_ang_vel_b.norm(dim=1)
        righted = ((upright - .65) / .35).clamp(0., 1.)
        braking = 2. / (1. + linear_speed / .2) + 2. / (1. + angular_speed / .4)
        action_cost = env.action_manager.action.square().mean(dim=1)
        stance = braking + 2. * support.mean(dim=1) + 2. * support.amin(dim=1)
        if args.stance_height_gate:
            # Keep a gradient through the 0.29 m screen threshold and toward nominal stance.
            stance = stance * ((height - .23) / .10).clamp(0., 1.)
        if args.stance_streak_reward:
            stable = (active & (upright >= .9396926208) & (height >= .29) & wheel_support
                      & (linear_speed <= .2) & (angular_speed <= .4) & (impact < 1.))
            if not hasattr(env, "recovery_stable_steps"):
                env.recovery_stable_steps = torch.zeros_like(env.episode_length_buf)
            streak = torch.where(stable, env.recovery_stable_steps + 1, 0)
            env.recovery_stable_steps = streak
            stance = stance + 4. * (streak * env.step_dt).clamp(max=1.)
        return active * (1.5 * upright + 2. * standing - .0005 * impact.clamp(max=2000.)
                         + righted * stance
                         - (.015 + .06 * righted) * action_cost)
    if args.bounded_recovery:
        slow_translation = torch.exp(-robot.data.root_lin_vel_b.square().sum(1) / .2**2)
        slow_rotation = torch.exp(-robot.data.root_ang_vel_b.square().sum(1) / .4**2)
        stance = upright.clamp_min(0.).pow(4) * wheel_support * slow_translation * slow_rotation
        return active * (1.5 * upright + (1. + 6. * phase.pow(3)) * standing
                         + 5. * phase * stance - .0005 * impact.clamp(max=2000.))
    return active * (1.5 * upright + (1. + 6. * phase.pow(3)) * standing
                     + 2. * phase * wheel_support * settled - .0005 * impact.clamp(max=2000.))


def actuator(kind, stiffness, damping):
    limits = {j.attrib["name"]: j.find("limit").attrib for j in ET.parse(URDF).getroot().findall("joint")
              if j.attrib["name"].endswith("_" + kind) and j.find("limit") is not None}
    if len(limits) != 4:
        raise ValueError(f"Expected four AS2-W {kind} joints, got {list(limits)}")
    return ImplicitActuatorCfg(
        joint_names_expr=list(limits), stiffness=stiffness, damping=damping,
        effort_limit_sim={name: float(limit["effort"]) for name, limit in limits.items()},
        velocity_limit_sim={name: float(limit["velocity"]) for name, limit in limits.items()},
    )


def screen_isaac_recovery(raw_env, env, policy, obs):
    robot = raw_env.scene["robot"].data
    sensor = raw_env.scene.sensors["contact_forces"]
    wheel_ids, _ = sensor.find_bodies(WHEELS)
    other_ids, _ = sensor.find_bodies("(?!.*_foot$).*")
    step_dt = raw_env.step_dt
    upright_threshold = .9396926208
    start_tick = round(SETTLE_SECONDS / step_dt)
    end_tick = round((SETTLE_SECONDS + RECOVERY_SECONDS) / step_dt)
    streak = torch.zeros_like(raw_env.episode_length_buf)
    longest = streak.clone()
    ever_upright = streak.bool()
    with torch.inference_mode():
        for tick in range(end_tick):
            if tick == start_tick:
                handoff = -robot.projected_gravity_b[:, 2].clone()
            obs = env.step(policy(obs))[0]
            if tick < start_tick:
                continue
            forces = sensor.data.net_forces_w
            upright = -robot.projected_gravity_b[:, 2] >= upright_threshold
            stable = (
                upright
                & (robot.root_pos_w[:, 2] - raw_env.scene.env_origins[:, 2] >= .29)
                & (robot.root_lin_vel_b.norm(dim=1) <= .2)
                & (robot.root_ang_vel_b.norm(dim=1) <= .4)
                & (forces[:, wheel_ids, 2].amin(dim=1) >= 8.)
                & (forces[:, other_ids, 2].amax(dim=1) < 1.)
            )
            ever_upright |= upright
            streak = torch.where(stable, streak + 1, 0)
            longest = torch.maximum(longest, streak)
    for label, selected in (
        ("inverted", handoff < -.5),
        ("tilted", (handoff >= -.5) & (handoff < upright_threshold)),
        ("already_upright", handoff >= upright_threshold),
    ):
        print("ISAAC_RECOVERY_SCREEN " + json.dumps({
            "handoff": label,
            "trials": int(selected.sum()),
            "reached_upright": int((selected & ever_upright).sum()),
            "stood_one_second": int((selected & (longest >= round(1. / step_dt))).sum()),
        }), flush=True)
    if args.isaac_fixed_screen:
        for index, (pose, seed, _, _) in enumerate(fixed_drop_cases()):
            print("ISAAC_FIXED_RECOVERY " + json.dumps({
                "pose": pose, "seed": seed, "handoff_upright_cosine": float(handoff[index]),
                "reached_upright": bool(ever_upright[index]),
                "longest_stable_s": int(longest[index]) * step_dt,
            }), flush=True)


def main():
    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 3.
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.episode_length_s = SETTLE_SECONDS + RECOVERY_SECONDS
    if args.isaac_screen:
        # Leave the final recovery sample available before auto-reset.
        cfg.episode_length_s += cfg.sim.dt * cfg.decimation
    cfg.seed = args.seed
    if args.mixed_terrain:
        configure_mixed_terrain(cfg)
    if args.audit_observations or args.isaac_screen:
        cfg.observations.policy.enable_corruption = False
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UrdfFileCfg(
            asset_path=str(URDF), fix_base=False, merge_fixed_joints=True,
            activate_contact_sensors=True,
            joint_drive=sim.UrdfFileCfg.JointDriveCfg(
                gains=sim.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., 1.1),
            joint_pos={".*_hip": 0., ".*_thigh": .8, ".*_calf": -1.6, WHEELS: 0.},
        ),
        actuators={kind: actuator(kind, kp, kd)
                   for kind, kp, kd in (("hip", 55., 2.), ("thigh", 55., 2.),
                                        ("calf", 55., 2.), ("foot", 0., 2.))},
    )
    if args.blind_stairs:
        cfg.scene.robot.init_state.pos = (0., 0., .37)
    cfg.actions.joint_pos.joint_names = [LEGS]
    cfg.actions.joint_pos.scale = .35
    cfg.actions.joint_pos.class_type = RecoveryLegAction
    cfg.actions.wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[WHEELS], scale=args.wheel_scale,
        use_default_offset=False, class_type=RecoveryWheelAction,
    )
    cfg.commands.base_velocity = None
    cfg.observations.policy.velocity_commands = ObservationTermCfg(
        func=lambda env: recovery_phase(env)[:, None])
    if args.blind_stairs:
        # Keep the warm-start width without feeding simulator root velocity to the actor.
        cfg.observations.policy.base_lin_vel = ObservationTermCfg(
            func=lambda env: torch.zeros((env.num_envs, 3), device=env.device))
        cfg.observations.policy.velocity_commands = ObservationTermCfg(
            func=lambda env: torch.full((env.num_envs, 1), .4, device=env.device))
    cfg.events.reset_base.params["pose_range"] = {
        "roll": (-3.1416, 3.1416), "pitch": (-1.57, 1.57), "yaw": (-3.1416, 3.1416),
    }
    cfg.events.reset_base.params["velocity_range"] = {}
    if args.righting_resets:
        cfg.events.reset_base.func = reset_righting_poses
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params["position_range"] = (-.55, .55)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0., 0.)
    if args.blind_stairs:
        cfg.events.reset_base.params["pose_range"] = {
            "x": (-.12, .12), "y": (-.15, .15), "yaw": (-.08, .08),
        }
        cfg.events.reset_robot_joints.params["position_range"] = (-.12, .12)
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-1., 1.)
    if args.isaac_fixed_screen:
        cfg.events.reset_base.func = reset_fixed_drop_root
        cfg.events.reset_base.params = {}
        cfg.events.reset_robot_joints.func = reset_fixed_drop_joints
        cfg.events.reset_robot_joints.params = {}
        cfg.events.add_base_mass = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.flat_domain_randomization:
        configure_flat_randomization(cfg)
    cfg.terminations.base_contact = None
    for term in vars(cfg.rewards):
        setattr(cfg.rewards, term, None)
    cfg.rewards.recover = RewardTermCfg(
        func=traversal_reward if args.blind_stairs else recovery_reward, weight=1., params={
            "wheel_cfg": SceneEntityCfg("contact_forces", body_names=WHEELS),
            "impact_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_foot$).*"),
        })
    if args.stair_clearance_reward:
        cfg.rewards.stair_clearance = RewardTermCfg(
            func=stair_clearance_reward, weight=1.,
            params={"wheel_cfg": SceneEntityCfg("robot", body_names=WHEELS)},
        )
    render_preview = args.preview_checkpoint and not (args.audit_observations or args.isaac_screen)
    if render_preview:
        cfg.viewer.resolution = (960, 540)
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.eye = (1.8, -1.8, 1.2)
    render_mode = "rgb_array" if render_preview else None
    raw_env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode)
    if args.export_replay_contract:
        try:
            robot = raw_env.scene["robot"]
            observations = raw_env.observation_manager
            actions = raw_env.action_manager
            contract = {
                "control_dt": raw_env.step_dt,
                "settle_seconds": SETTLE_SECONDS,
                "recovery_seconds": RECOVERY_SECONDS,
                "blind_stairs": args.blind_stairs,
                "stair_clearance_reward": args.stair_clearance_reward,
                "policy_observations": [
                    {"name": name, "shape": list(shape)}
                    for name, shape in zip(observations.active_terms["policy"],
                                           observations.group_obs_term_dim["policy"], strict=True)
                ],
                "action_terms": [
                    {"name": name, "joints": list(actions.get_term(name)._joint_names),
                     "scale": getattr(cfg.actions, name).scale}
                    for name in actions.active_terms
                ],
                "joint_names": list(robot.joint_names),
                "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            }
            args.export_replay_contract.parent.mkdir(parents=True, exist_ok=True)
            args.export_replay_contract.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        finally:
            raw_env.close()
        return
    action_clip = 6. if args.settling_recovery else None
    env = RslRlVecEnvWrapper(raw_env, clip_actions=action_clip)
    runner_cfg = UnitreeGo2FlatPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.num_steps_per_env = 24
    runner_cfg.save_interval = 1000
    if args.settling_recovery:
        runner_cfg.policy.init_noise_std = .65
        runner_cfg.algorithm.entropy_coef = .002
    if args.conservative_ppo:
        runner_cfg.algorithm.learning_rate = 5e-5
        runner_cfg.algorithm.schedule = "fixed"
        runner_cfg.algorithm.clip_param = .1
    if args.mirror_recovery:
        robot = raw_env.scene["robot"]
        if tuple(robot.joint_names) != JOINT_ORDER:
            raise ValueError("Recovery mirror requires the official AS2-W joint order")
        runner_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=not args.mirror_loss_only, use_mirror_loss=True,
            data_augmentation_func=augment_recovery_mirror, mirror_loss_coeff=.02,
        )
    run_dir = args.run_dir or ROOT / "runs/fall_recovery" / datetime.now().strftime("%Y%m%d-%H%M%S")
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(run_dir), device=args.device)
    try:
        if args.preview_checkpoint:
            import subprocess

            runner.load(str(args.preview_checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            args.preview_video.parent.mkdir(parents=True, exist_ok=True)
            obs = env.get_observations()
            if isinstance(obs, tuple):
                obs = obs[0]
            if args.isaac_screen:
                screen_isaac_recovery(raw_env, env, policy, obs)
                return
            if args.audit_observations:
                robot = raw_env.scene["robot"].data
                last = torch.zeros_like(policy(obs))
                audit_steps = args.audit_step or [100]
                previous_wheel_angle = robot.joint_pos[0, -4:].clone()
                with torch.inference_mode():
                    for tick in range(299):
                        current_wheel_angle = robot.joint_pos[0, -4:]
                        if args.audit_wheel_spin:
                            delta = current_wheel_angle - previous_wheel_angle
                            if (delta.abs() > torch.pi).any():
                                print("RECOVERY_WHEEL_WRAP " + json.dumps({
                                    "tick": tick, "before": previous_wheel_angle.tolist(),
                                    "after": current_wheel_angle.tolist(), "delta": delta.tolist(),
                                    "velocity": robot.joint_vel[0, -4:].tolist(),
                                }), flush=True)
                            previous_wheel_angle = current_wheel_angle.clone()
                        actual = obs["policy"] if not isinstance(obs, torch.Tensor) else obs
                        if tick in (0, 50, 99, 100, 125, 200, 298):
                            expected = torch.cat((
                                robot.root_lin_vel_b, robot.root_ang_vel_b, robot.projected_gravity_b,
                                recovery_phase(raw_env)[:, None],
                                robot.joint_pos - robot.default_joint_pos,
                                robot.joint_vel, last,
                            ), dim=1)
                            offsets = (0, 3, 6, 9, 10, 26, 42, 58)
                            errors = [(actual[:, lo:hi] - expected[:, lo:hi]).abs().max().item()
                                      for lo, hi in zip(offsets, offsets[1:])]
                            print("RECOVERY_AUDIT " + json.dumps({
                                "tick": tick, "max_error_by_term": errors,
                                "wheel_angles": robot.joint_pos[0, -4:].tolist(),
                                "policy_wheel_angles": actual[0, 22:26].tolist(),
                            }), flush=True)
                        last = policy(obs)
                        if action_clip is not None:
                            last = last.clamp(-action_clip, action_clip)
                        if args.audit_wheel_spin and tick >= round(SETTLE_SECONDS / raw_env.step_dt):
                            last[:, -4:] = 6.
                        if tick in audit_steps:
                            def snapshot():
                                return {name: getattr(robot, name)[0].tolist() for name in (
                                    "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_b",
                                    "joint_pos", "joint_vel",
                                )}

                            before = snapshot()
                            obs = env.step(last)[0]
                            print("RECOVERY_STEP " + json.dumps({
                                "tick": tick, "before": before, "action": last[0].tolist(), "after": snapshot(),
                            }), flush=True)
                        else:
                            obs = env.step(last)[0]
                return
            first_frame = raw_env.render()
            height, width = first_frame.shape[:2]
            with subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{width}x{height}", "-framerate", "50", "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.preview_video),
            ], stdin=subprocess.PIPE) as encoder, torch.inference_mode():
                for _ in range(round(cfg.episode_length_s / raw_env.step_dt) - 1):
                    encoder.stdin.write(raw_env.render().tobytes())
                    obs = env.step(policy(obs))[0]
                encoder.stdin.close()
                if encoder.wait():
                    raise RuntimeError("Failed to encode recovery preview")
        else:
            source = args.resume_checkpoint or args.initialize_checkpoint
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "seed": args.seed, "device": args.device, "num_envs": args.num_envs,
                "ppo_updates_requested": args.iterations, "save_interval": runner_cfg.save_interval,
                "control_dt": raw_env.step_dt, "settle_seconds": SETTLE_SECONDS,
                "recovery_seconds": RECOVERY_SECONDS, "action_clip": action_clip,
                "wheel_scale": args.wheel_scale, "bounded_recovery": args.bounded_recovery,
                "settling_recovery": args.settling_recovery, "righting_resets": args.righting_resets,
                "stance_height_gate": args.stance_height_gate, "full_leg_range": args.full_leg_range,
                "flat_domain_randomization": args.flat_domain_randomization,
                "mixed_terrain": args.mixed_terrain,
                "blind_stairs": args.blind_stairs,
                "stair_clearance_reward": args.stair_clearance_reward,
                "shared_ground_friction": args.shared_ground_friction,
                "mirror_recovery": args.mirror_recovery,
                "mirror_loss_only": args.mirror_loss_only,
                "conservative_ppo": args.conservative_ppo,
                "stance_streak_reward": args.stance_streak_reward,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            }
            if source is not None:
                manifest["source_checkpoint"] = str(source)
                manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
                manifest["resume_optimizer"] = bool(args.resume_checkpoint)
            (run_dir / "run-config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if args.resume_checkpoint:
                runner.load(str(args.resume_checkpoint), load_optimizer=True)
            if args.initialize_checkpoint:
                runner.load(str(args.initialize_checkpoint), load_optimizer=False)
                runner.current_learning_iteration = 0
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
