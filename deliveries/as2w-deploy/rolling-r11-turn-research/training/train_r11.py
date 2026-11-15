"""AS2-W flat rolling baseline with Go2-W inspired velocity rewards."""
import argparse
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

VX_CAP, YAW_CAP, MIN_CMD = .25, .15, .05


def train(args, physics):
    import torch
    from isaaclab.managers import RewardTermCfg
    from as2w_isaac_flat_env_all import (
        JOINTS, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
    )
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_isaac_speed_range import wheel_support_error
    from as2w_training_safety import apply_safety, rolling_knee_margin_cost

    class RollingCommands(TakeoverCommands):
        def _resample_command(self, env_ids):
            draw = torch.rand((self.targets[env_ids].shape[0], 5), device=self.device)
            mode = torch.bucketize(draw[:, 0].contiguous(), draw.new_tensor((.20, .35, .75)))
            vx = (MIN_CMD + (VX_CAP - MIN_CMD) * draw[:, 1]) * torch.where(draw[:, 2] < .5, -1., 1.)
            yaw = (MIN_CMD + (YAW_CAP - MIN_CMD) * draw[:, 3]) * torch.where(draw[:, 4] < .5, -1., 1.)
            self.targets[env_ids, 0] = vx * ((mode == 1) | (mode == 3))
            self.targets[env_ids, 1] = 0.
            self.targets[env_ids, 2] = yaw * (mode >= 2)

    cfg = robust_config(SimpleNamespace(
        **vars(args), profile="pose-delay", direction_focus=True,
        command_stage="all", zero_memory=True, zero_focus=True), physics)
    cmd = cfg.commands.base_velocity
    cmd.class_type = RollingCommands
    cmd.resampling_time_range = (10., 10.)
    cmd.ranges.lin_vel_x = (-VX_CAP, VX_CAP)
    cmd.ranges.lin_vel_y = (0., 0.)
    cmd.ranges.ang_vel_z = (-YAW_CAP, YAW_CAP)
    cfg.episode_length_s = 32.
    cfg.scene.robot.actuators["foot"].damping = 6.
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
    cfg.rewards.rolling_knee_margin = RewardTermCfg(
        func=rolling_knee_margin_cost, weight=-3., params={"margin_m": .10})

    # ---- guard component counter (which of the 3 stance conditions fires) ----
    import as2w_training_safety as _ts
    _guard = {"crossed": 0, "rear_overextended": 0, "low_knee": 0, "checks": 0}
    _orig_sv = _ts.stance_violation

    def _sv_counted(env, *, components=False):
        c, r, k = _orig_sv(env, components=True)
        _guard["crossed"] += int(c.sum())
        _guard["rear_overextended"] += int(r.sum())
        _guard["low_knee"] += int(k.sum())
        _guard["checks"] += 1
        return (c | r | k) if not components else (c, r, k)

    _ts.stance_violation = _sv_counted

    limits = apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=6.)
        if wrapped.get_observations()["policy"].shape[-1] != 65 or wrapped.num_actions != 16:
            raise RuntimeError("Unexpected 65x16 policy interface")
        ppo = UnitreeGo2FlatPPORunnerCfg()
        ppo.seed, ppo.device = args.seed, args.device
        ppo.max_iterations = args.iterations
        ppo.save_interval = min(int(os.environ.get("AS2W_R11_SAVE_INTERVAL", "250")),
                                args.iterations)
        ppo.actor.distribution_cfg.init_std = .25
        ppo.algorithm.learning_rate = 3e-4
        ppo = handle_deprecated_rsl_rl_cfg(ppo, check_rsl_rl_version())
        runner = create_rsl_rl_runner(wrapped, ppo, log_dir=str(args.out))
        if args.init_checkpoint is not None:
            args.init_checkpoint = args.init_checkpoint.resolve(strict=True)
            runner.load(str(args.init_checkpoint), map_location=args.device)
            print("R11_RESUMED_FROM", args.init_checkpoint,
                  "iter", runner.current_learning_iteration, flush=True)
        files = [Path(__file__), args.urdf, args.usd]
        record = {
            "scope": "flat rolling simulation training",
            "initial_checkpoint": (str(args.init_checkpoint)
                                   if args.init_checkpoint else None),
            "save_interval": ppo.save_interval, "num_envs": args.num_envs,
            "iterations": args.iterations, "seed": args.seed,
            "policy_step_s": env.step_dt, "physics_step_s": env.physics_dt,
            "joint_order": JOINTS, "action_clip": 6., "safety": limits,
            "command_mix": {"hold": .20, "straight": .15, "spin": .40, "arc": .25},
            "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            "environment": cfg.to_dict(), "ppo": ppo.to_dict(),
        }
        (args.out / "contract.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        runner.save(str(args.out / "r11-final.pt"))
        (args.out / "guard_counts.json").write_text(json.dumps(_guard, indent=1))
        print("R11_GUARD_COUNTS", _guard, flush=True)
        print("R11_FANZIQI_LAB_COMPLETE", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=311)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="resume actor/critic/optimizer from this checkpoint")
    add_launcher_args(parser)
    args = parser.parse_args()
    args.usd = args.usd.resolve(strict=True)
    args.urdf = args.urdf.resolve(strict=True)
    args.out.mkdir()
    args.physics = "isaacsim_physx"
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        train(args, physics)


if __name__ == "__main__":
    main()
