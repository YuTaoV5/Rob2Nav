"""R284 transfer experiments on PhysX, with separate checkpoints and reports."""
import argparse
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def run(args, physics):
    import torch
    from as2w_isaac_flat_env_all import (
        RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
    )
    from as2w_isaac_robust_env import RobustEnvironment, robust_config

    cfg = robust_config(args, physics)
    cfg.as2w_rolling_stand_compensation = args.rolling_stand_compensation
    cfg.as2w_rolling_wheel_hold = args.rolling_wheel_hold
    cfg.as2w_park_precalibrated = args.park_precalibrated
    cfg.as2w_rolling_loaded_stance = args.rolling_loaded_stance
    cfg.as2w_park_odometry = args.park_odometry
    cfg.as2w_park_common_gain_scale = args.park_common_gain_scale
    cfg.as2w_park_common_speed_damping = args.park_common_speed_damping
    cfg.as2w_park_settle_joints = args.park_settle_joints
    cfg.as2w_park_continuous_bias = args.park_continuous_bias
    cfg.as2w_park_straight_bias = args.park_straight_bias
    cfg.as2w_pure_turn_common_scale = args.pure_turn_common_scale
    cfg.as2w_rolling_maneuvers = args.rolling_maneuvers
    cfg.as2w_ignore_lateral_command = args.speed_rolling
    cfg.as2w_rolling_low_speed_smooth = args.rolling_low_speed_smooth
    cfg.as2w_rolling_leg_filter_hz = args.rolling_leg_filter_hz
    if args.speed_cap and not args.evaluate:
        from as2w_isaac_speed_range import configure_speed
        configure_speed(cfg, args.speed_cap, args.speed_rolling, fixed_exit=args.rolling_fixed_exit)
        if args.rolling_maneuvers:
            cfg.as2w_motion_start_s = 4.
            cfg.as2w_park_until_s = 7.
            cfg.episode_length_s = 23.
    if args.rolling_low_speed_smooth and not args.evaluate:
        from isaaclab.managers import RewardTermCfg
        from as2w_isaac_speed_range import rolling_unslewed_leg_rate
        cfg.rewards.rolling_unslewed_leg_rate = RewardTermCfg(func=rolling_unslewed_leg_rate, weight=-.10)
        cfg.rewards.action_rate_l2.weight = -.02
    if args.rolling_leg_filter_hz and not args.evaluate:
        from isaaclab.managers import RewardTermCfg
        from as2w_rolling_filtered import rolling_phase_cost

        cfg.rewards.rolling_phase_cost = RewardTermCfg(func=rolling_phase_cost, weight=-.35)
    if args.teleop:
        from as2w_isaac_teleop import configure_teleop
        configure_teleop(cfg, args.teleop)
    if args.evaluate:
        cfg.episode_length_s = 26. if args.teleop_evaluate or args.speed_cap else 18.
    limits = None
    if args.safety:
        from as2w_training_safety import apply_safety
        limits = apply_safety(cfg, args.urdf, surface_guard=args.rolling_fixed_exit)
    if args.rolling_knee_margin:
        from isaaclab.managers import RewardTermCfg
        from as2w_training_safety import rolling_knee_margin_cost

        cfg.rewards.rolling_knee_margin = RewardTermCfg(
            func=rolling_knee_margin_cost, weight=-3.0, params={"margin_m": .10})
    env_class = RobustEnvironment
    if args.rolling_fixed_exit:
        from as2w_isaac_speed_range import RollingFixedExitEnvironment
        env_class = RollingFixedExitEnvironment
    elif args.rolling_residual:
        from as2w_isaac_speed_range import RollingResidualEnvironment
        env_class = RollingResidualEnvironment
    motion_audit = args.safety and args.evaluate and args.rolling_fixed_exit
    if motion_audit:
        from as2w_training_safety import configure_motion_substep_audit, install_motion_substep_audit
        configure_motion_substep_audit(cfg)
    env = env_class(cfg)
    try:
        substep_contract = install_motion_substep_audit(env) if motion_audit else None
        wrapped = RslRlVecEnvWrapper(env, clip_actions=6.)
        obs = wrapped.get_observations()["policy"]
        if obs.shape[-1] != 65 or wrapped.num_actions != 16:
            raise RuntimeError("R284 policy shape changed")
        runner_cfg = UnitreeGo2FlatPPORunnerCfg()
        runner_cfg.seed = args.seed
        runner_cfg.device = args.device
        runner_cfg.max_iterations = args.iterations
        runner_cfg.save_interval = min(25, args.iterations)
        runner_cfg.algorithm.learning_rate = args.learning_rate
        if args.rolling_maneuvers:
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
            from as2w_isaac_flat_env_all import mirror_flat

            runner_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
                data_augmentation_func=mirror_flat,use_data_augmentation=True,
                use_mirror_loss=True,mirror_loss_coeff=.15)
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(wrapped, runner_cfg, log_dir=str(args.out))
        runner.load(str(args.checkpoint), load_cfg={"actor": True, "critic": True},
                    map_location=args.device)
        (args.out / "contract.json").write_text(json.dumps({
            **checkpoint_record(args.checkpoint), "profile": args.profile,
            "motion_substep_audit": substep_contract,
            "teleop_arm": args.teleop, "teleop_evaluate": args.teleop_evaluate,
            "learning_rate": args.learning_rate, "speed_cap_m_s": args.speed_cap,
            "speed_rolling": args.speed_rolling, "simulation_safety": limits,
            "rolling_residual": args.rolling_residual,
            "rolling_fixed_exit": args.rolling_fixed_exit,
            "mirror_rolling": args.mirror_rolling,
            "rolling_stand_compensation": args.rolling_stand_compensation,
            "rolling_wheel_hold": args.rolling_wheel_hold,
            "park_precalibrated": args.park_precalibrated,
            "rolling_loaded_stance": args.rolling_loaded_stance,
            "park_odometry": args.park_odometry,
            "park_common_gain_scale": args.park_common_gain_scale,
            "park_common_speed_damping": args.park_common_speed_damping,
            "park_settle_joints": args.park_settle_joints,
            "park_continuous_bias": args.park_continuous_bias,
            "park_straight_bias": args.park_straight_bias,
            "rolling_maneuvers": args.rolling_maneuvers,
            "rolling_low_speed_smooth": args.rolling_low_speed_smooth,
            "wrapped_leg_slew_rad_s": .9 if args.rolling_low_speed_smooth else None,
            "rolling_knee_margin_m": .10 if args.rolling_knee_margin else None,
            "suppress_pure_turn_common": args.suppress_pure_turn_common,
            "pure_turn_common_scale": args.pure_turn_common_scale,
            "turn_yaw_slew_rad_s2": args.turn_yaw_slew,
            "seed": args.seed, "num_envs": args.num_envs, "iterations": args.iterations,
            "physics_step_s": env.physics_dt, "policy_step_s": env.step_dt,
            "pose_source": "simulation_proxy_no_raw_lowstate",
            "rolling_leg_filter_hz": args.rolling_leg_filter_hz,
            "policy_width": 65, "actions": 16, "evaluate": args.evaluate,
        }, indent=2) + "\n")
        if args.evaluate:
            if args.speed_cap:
                from eval_as2w_isaac_speed_range import evaluate
            elif args.teleop_evaluate:
                from eval_as2w_isaac_teleop import evaluate
            else:
                from eval_as2w_isaac_robust import evaluate
            actor = runner.get_inference_policy(args.device)
            if args.mirror_rolling:
                from as2w_rolling_fixed_exit import mirror_rolling_joints, mirror_rolling_observation

                def mirrored(obs):
                    other = obs.clone()
                    other["policy"] = mirror_rolling_observation(obs["policy"])
                    return .5*(actor(obs) + mirror_rolling_joints(actor(other)))

                evaluate(env, wrapped, mirrored, args)
            else:
                evaluate(env, wrapped, actor, args)
        else:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        print("AS2W_ROBUST_COMPLETE", args.profile, flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "nominal", "pose", "delay",
                                              "pose-delay", "friction", "combined", "expanded"), required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=286)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rolling-leg-filter-hz", type=float, default=0.)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--safety", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rolling-residual", action="store_true")
    mode.add_argument("--rolling-fixed-exit", action="store_true")
    parser.add_argument("--speed-rolling", action="store_true")
    parser.add_argument("--mirror-rolling", action="store_true")
    parser.add_argument("--rolling-stand-compensation", action="store_true")
    parser.add_argument("--rolling-wheel-hold", action="store_true")
    parser.add_argument("--park-precalibrated",action="store_true")
    parser.add_argument("--rolling-loaded-stance",action="store_true")
    parser.add_argument("--park-odometry",action="store_true")
    parser.add_argument("--park-common-gain-scale",type=float,default=1.)
    parser.add_argument("--park-common-speed-damping",type=float,default=0.)
    parser.add_argument("--park-settle-joints",action="store_true")
    parser.add_argument("--park-continuous-bias",action="store_true")
    parser.add_argument("--park-straight-bias",action="store_true")
    parser.add_argument("--suppress-pure-turn-common",action="store_true")
    parser.add_argument("--pure-turn-common-scale",type=float,default=1.)
    parser.add_argument("--rolling-knee-margin",action="store_true")
    parser.add_argument("--rolling-low-speed-smooth",action="store_true")
    parser.add_argument("--rolling-maneuvers",action="store_true")
    parser.add_argument("--turn-yaw-slew",type=float,default=0.)
    parser.add_argument("--speed-cap", type=float, default=0.)
    parser.add_argument("--teleop", choices=("a", "b"))
    parser.add_argument("--teleop-evaluate", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.suppress_pure_turn_common:
        if args.pure_turn_common_scale != 1.:
            parser.error("Choose one pure-turn common-mode option")
        args.pure_turn_common_scale = 0.
    if not 0. <= args.pure_turn_common_scale <= 1. or (
            args.pure_turn_common_scale != 1. and not args.rolling_fixed_exit):
        parser.error("Pure-turn common-mode scale must be 0–1 and requires rolling fixed exit")
    if (args.rolling_residual or args.rolling_fixed_exit) and not (args.speed_rolling and args.safety):
        parser.error("Rolling action mappings require --speed-rolling and --safety")
    if not 0. <= args.turn_yaw_slew <= 3. or (args.turn_yaw_slew and not (args.evaluate and args.speed_rolling)):
        parser.error("Turn yaw slew requires rolling evaluation and a 0–3 rad/s² cap")
    if args.rolling_knee_margin and not (args.speed_rolling and args.safety and args.rolling_fixed_exit and not args.evaluate):
        parser.error("--rolling-knee-margin requires guarded rolling fixed-exit training")
    if args.rolling_low_speed_smooth and not (args.rolling_fixed_exit and args.speed_rolling and args.speed_cap == .2):
        parser.error("Low-speed smooth curriculum requires rolling fixed exit at a 0.2 m/s cap")
    if not 0. <= args.rolling_leg_filter_hz <= 20. or (
        args.rolling_leg_filter_hz and not (args.rolling_fixed_exit and args.rolling_low_speed_smooth)
    ):
        parser.error("Leg filtering requires the guarded low-speed rolling profile and a cutoff in (0, 20] Hz")
    if args.rolling_maneuvers and not (args.speed_rolling and args.park_straight_bias and not args.evaluate):
        parser.error("Maneuver curriculum requires rolling training with straight-mode bias")
    if args.park_straight_bias and not args.park_continuous_bias:
        parser.error("--park-straight-bias requires continuous bias")
    if args.park_continuous_bias and (not args.park_precalibrated or args.rolling_loaded_stance):
        parser.error("Continuous bias requires precalibration without loaded-stance compensation")
    if args.park_settle_joints and not args.rolling_wheel_hold:
        parser.error("--park-settle-joints requires --rolling-wheel-hold")
    if not 0. <= args.park_common_speed_damping <= 1. or (args.park_common_speed_damping and not args.park_odometry):
        parser.error("Park speed damping must be in [0, 1] and requires odometry when enabled")
    if not 0. < args.park_common_gain_scale <= 1. or (args.park_common_gain_scale != 1. and not args.park_odometry):
        parser.error("Park common gain scale must be in (0, 1] and requires odometry when changed")
    if args.park_odometry and not args.park_precalibrated:
        parser.error("--park-odometry requires --park-precalibrated")
    if args.rolling_loaded_stance and not args.park_precalibrated:
        parser.error("--rolling-loaded-stance requires --park-precalibrated")
    if args.park_precalibrated and not args.rolling_wheel_hold:
        parser.error("--park-precalibrated requires --rolling-wheel-hold")
    if args.rolling_wheel_hold and not args.rolling_stand_compensation:
        parser.error("--rolling-wheel-hold requires --rolling-stand-compensation")
    if args.rolling_stand_compensation and not (args.rolling_fixed_exit and args.profile=="expanded"):
        parser.error("--rolling-stand-compensation requires expanded fixed-exit mode")
    if args.mirror_rolling and not (args.evaluate and args.rolling_fixed_exit):
        parser.error("--mirror-rolling requires fixed-exit evaluation")
    if args.speed_rolling and not args.speed_cap:
        parser.error("--speed-rolling requires --speed-cap")
    if args.speed_cap and not .10 <= args.speed_cap <= 3.:
        parser.error("--speed-cap must be 0 or between .10 and 3 m/s")
    if args.speed_cap and (args.teleop or args.teleop_evaluate):
        parser.error("Use a single command curriculum")
    if args.teleop_evaluate and not args.evaluate:
        parser.error("--teleop-evaluate requires --evaluate")
    if args.teleop and args.evaluate:
        parser.error("Evaluation uses scripted targets, omit --teleop")
    args.command_stage = "all"
    args.direction_focus = True
    args.zero_memory = True
    args.zero_focus = False
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "contract.json").exists():
        parser.error("Use a separate run directory")
    args.physics = "isaacsim_physx"
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        run(args, physics)


if __name__ == "__main__":
    main()