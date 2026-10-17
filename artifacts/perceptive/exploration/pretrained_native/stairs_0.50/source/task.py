"""Terrain-aware actor trained with PPO; no MoE, IK correction or scripted jump."""
import copy
import torch
from isaaclab.managers import RewardTermCfg, TerminationTermCfg, CurriculumTermCfg, SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from .terrain import LaneCfg


def reached(env):
    p = env.scene["robot"].data.root_pos_w-env.scene.env_origins
    return (p[:, 0] > 7.5) & (p[:, 1].abs() < .8)


def off_lane(env):
    p = env.scene["robot"].data.root_pos_w-env.scene.env_origins
    if getattr(env, "_capture_eval", False):
        env._eval_position = p.clone()
        env._eval_gravity = env.scene["robot"].data.projected_gravity_b.clone()
        env._eval_forces = env.scene["contact_forces"].data.net_forces_w.clone()
    return (p[:, 1].abs() > .85) | (p[:, 0] < -.7) | (p[:, 2] < -.08)


def forward_progress(env):
    return env.scene["robot"].data.root_lin_vel_w[:, 0].clamp(-1., 1.5)


def lane_center(env):
    return (env.scene["robot"].data.root_pos_w[:, 1]-env.scene.env_origins[:, 1]).square()


def terrain_curriculum(env, env_ids):
    terrain = env.scene.terrain
    p = env.scene["robot"].data.root_pos_w[env_ids]-env.scene.env_origins[env_ids]
    up = (p[:, 0] > 7.) & (p[:, 1].abs() < .8)
    down = (p[:, 0] < 3.) & ~up
    terrain.update_env_origins(env_ids, up, down)
    return terrain.terrain_levels.float().mean()


def make_cfg(num_envs=2048, seed=42, blind=False, kind="mixed", evaluation=False, difficulty=None, fused=False, teacher_velocity=False):
    cfg = UnitreeGo2RoughEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 3.
    cfg.episode_length_s = 16.
    kinds = ["flat", "stairs", "platform", "stones"] if kind == "mixed" else [kind]
    proportions = {"flat": .15, "stairs": .30, "platform": .25, "stones": .30}
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed, size=(10., 3.), border_width=3., num_rows=10, num_cols=20 if kind == "mixed" else 4,
        horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75,
        use_cache=False, curriculum=True, difficulty_range=(0., 1.),
        sub_terrains={k: LaneCfg(kind=k, proportion=proportions[k]) for k in kinds},
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.height_scanner.offset.pos = (.25, 0., 20.)
    cfg.scene.height_scanner.pattern_cfg.size = [1.6, 1.0]
    cfg.scene.height_scanner.pattern_cfg.resolution = .1
    cfg.scene.height_scanner.update_period = .02
    cfg.commands.base_velocity.ranges.lin_vel_x = (.5, .9)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-.5, .5)
    cfg.commands.base_velocity.ranges.heading = (0., 0.)
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.debug_vis = False
    cfg.events.reset_base.params["pose_range"] = {"x": (-.15, .15), "y": (-.08, .08), "yaw": (-.08, .08)}
    cfg.events.physics_material.params["static_friction_range"] = (.7, 1.1)
    cfg.events.physics_material.params["dynamic_friction_range"] = (.6, .9)
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-.5, 1.5)
    cfg.events.base_external_force_torque = None
    cfg.observations.critic = copy.deepcopy(cfg.observations.policy)
    cfg.observations.critic.enable_corruption = False
    # Actor gets 45 proprioceptive/command values + 187 height rays, no true base velocity.
    if not teacher_velocity:
        cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.height_scan.noise = AdditiveUniformNoiseCfg(n_min=-.015, n_max=.015)
    if blind:
        cfg.observations.policy.height_scan = None
    if fused:
        from .sensors import add_sensors, measured_height
        from isaaclab.managers import ObservationTermCfg
        add_sensors(cfg)
        cfg.observations.policy.height_scan = ObservationTermCfg(func=measured_height)
    cfg.rewards.forward_progress = RewardTermCfg(func=forward_progress, weight=1.)
    cfg.rewards.lane_center = RewardTermCfg(func=lane_center, weight=-1.)
    cfg.rewards.reached = RewardTermCfg(func=reached, weight=100.)
    cfg.rewards.undesired_contacts = RewardTermCfg(
        func=__import__("isaaclab.envs.mdp", fromlist=["undesired_contacts"]).undesired_contacts,
        weight=-.3, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]), "threshold": 5.})
    cfg.rewards.lin_vel_z_l2.weight = -.5
    cfg.rewards.dof_pos_limits.weight = -1.
    cfg.rewards.feet_air_time.weight = .1
    cfg.terminations.reached = TerminationTermCfg(func=reached, time_out=True)
    cfg.terminations.off_lane = TerminationTermCfg(func=off_lane)
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_curriculum)
    if evaluation:
        cfg.observations.policy.enable_corruption = False
        cfg.curriculum.terrain_levels = None
        cfg.events.add_base_mass = None
        cfg.commands.base_velocity.ranges.lin_vel_x = (.7, .7)
        if difficulty is not None:
            cfg.scene.terrain.terrain_generator.difficulty_range = (difficulty, difficulty)
            cfg.scene.terrain.terrain_generator.num_rows = 1
    return cfg
