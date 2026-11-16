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
        """Command sampler.

        Default path is the unchanged armE mixture (hold 20 / straight 15 / pure turn 40 / arc 25).
        With AS2W_AB_CURRICULUM=1 the operator's B-group mixture takes over:

            35% pure turn   vx=0, vy=0, wz=+-0.05 / +-0.10
            25% arc         vx=+0.25 with the frozen yaw steps
            15% straight    low-speed continuous change plus +-0.25, vy=0
            15% brake       drive at +-0.25 then release to zero inside the same episode
            10% hold        zero command

        This lives HERE rather than in robust_config because the training entry sets
        cmd.class_type = RollingCommands AFTER robust_config returns, which overwrote an earlier
        attempt wired one layer down - see evidence/AB_ARM_B_ABORTED.md.  The startup line is printed
        from this class so it cannot be false-positive from a class that never runs.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._ab_on = os.environ.get("AS2W_AB_CURRICULUM", "0") == "1"
            if self._ab_on:
                f = [float(os.environ.get("AS2W_AB_TURN_FRACTION", 0.35)),
                     float(os.environ.get("AS2W_AB_ARC_FRACTION", 0.25)),
                     float(os.environ.get("AS2W_AB_STRAIGHT_FRACTION", 0.15)),
                     float(os.environ.get("AS2W_AB_BRAKE_FRACTION", 0.15)),
                     float(os.environ.get("AS2W_AB_HOLD_FRACTION", 0.10))]
                tot = sum(f)
                self._ab_frac = torch.tensor([x / tot for x in f], dtype=torch.float32, device=self.device)
                self._ab_brake = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._ab_release_at = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self._ab_count = torch.zeros(5, dtype=torch.long, device=self.device)
                print("AS2W_AB_CURRICULUM ON  class=RollingCommands  fractions=%s  brake_release=3-6 s"
                      % [round(float(v), 3) for v in self._ab_frac], flush=True)
            else:
                print("AS2W_AB_CURRICULUM OFF (armE sampling unchanged)", flush=True)

        def _resample_ab(self, env_ids):
            n = self.targets[env_ids].shape[0]
            cat = torch.multinomial(self._ab_frac, n, replacement=True)
            cmd = torch.zeros(n, 3, device=self.device)
            u = torch.rand(n, device=self.device)
            sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
            # 0 pure turn: |wz| in {0.05, 0.10}
            m = cat == 0
            mag = torch.where(torch.rand(n, device=self.device) < 0.5, 0.05, 0.10)
            cmd[m, 2] = (sign * mag)[m]
            # 1 arc: vx=+0.25 with the frozen yaw step
            m = cat == 1
            cmd[m, 0] = 0.25
            cmd[m, 2] = (sign * 0.05)[m]
            # 2 straight: continuous in [-0.25, +0.25], vy=0
            m = cat == 2
            cmd[m, 0] = (torch.rand(n, device=self.device) * 0.50 - 0.25)[m]
            # 3 brake: drive at +-0.25 now, released to zero later inside this episode
            m = cat == 3
            cmd[m, 0] = (torch.where(u < 0.5, 1.0, -1.0) * 0.25)[m]
            # 4 hold stays zero
            self.targets[env_ids] = cmd
            brk = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            rel = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            ep = self._env.episode_length_buf
            lo = int(3.0 / self._env.step_dt)
            hi = int(6.0 / self._env.step_dt)
            brk[env_ids] = m
            rel[env_ids] = ep[env_ids] + torch.randint(lo, hi + 1, (n,), device=self.device)
            self._ab_brake.copy_(brk)
            self._ab_release_at.copy_(rel)
            for c in range(5):
                self._ab_count[c] += int((cat == c).sum())

        def _resample_command(self, env_ids):
            if getattr(self, "_ab_on", False):
                return self._resample_ab(env_ids)
            draw = torch.rand((self.targets[env_ids].shape[0], 5), device=self.device)
            mode = torch.bucketize(draw[:, 0].contiguous(), draw.new_tensor((.20, .35, .75)))
            vx = (MIN_CMD + (VX_CAP - MIN_CMD) * draw[:, 1]) * torch.where(draw[:, 2] < .5, -1., 1.)
            yaw = (MIN_CMD + (YAW_CAP - MIN_CMD) * draw[:, 3]) * torch.where(draw[:, 4] < .5, -1., 1.)
            self.targets[env_ids, 0] = vx * ((mode == 1) | (mode == 3))
            self.targets[env_ids, 1] = 0.
            self.targets[env_ids, 2] = yaw * (mode >= 2)

        def _update_command(self):
            # brake cases: once the scheduled release step is reached the target is zeroed, so the
            # episode contains decelerate -> stop and the existing lerp produces the ramp
            if getattr(self, "_ab_on", False) and bool(self._ab_brake.any()):
                due = self._ab_brake & (self._env.episode_length_buf >= self._ab_release_at)
                if bool(due.any()):
                    self.targets[due] = 0.0
                    self._ab_brake &= ~due
            return super()._update_command()

    cfg = robust_config(SimpleNamespace(
        **vars(args), profile="pose-delay", direction_focus=True,
        command_stage="all", zero_memory=True, zero_focus=True), physics)
    cmd = cfg.commands.base_velocity
    cmd.class_type = RollingCommands
    cmd.resampling_time_range = (10., 10.)
    # Opt-in: the long-dwell curriculum. The DWELL_* variables were historically only printed, and
    # the real hold came from resampling_time_range=(10,10) above. This switch makes the hold
    # 8-12 s on a sub-fraction of turn/arc samples and emits START/END/SUMMARY events as proof.
    if os.environ.get("AS2W_AB_LONG_DWELL_CURRICULUM", "0") == "1":
        from as2w_long_dwell_curriculum import LongDwellCommands
        cmd.class_type = LongDwellCommands
        print("AS2W_LONG_DWELL_CLASS ON  LongDwellCommands", flush=True)
    cmd.ranges.lin_vel_x = (-VX_CAP, VX_CAP)
    cmd.ranges.lin_vel_y = (0., 0.)
    cmd.ranges.ang_vel_z = (-YAW_CAP, YAW_CAP)
    cfg.episode_length_s = 32.
    cfg.scene.robot.actuators["foot"].damping = 6.
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    # ---- goal-sanctioned yaw-training knobs (see GOAL_STATUS 42.5) ----
    # Rationale: the lambda sweep proved the policy earns left yaw from a
    # direction-biased LEG STANCE asymmetry.  The goal requires that leg changes
    # must not mask a wheel-end deficiency, so this arm both strengthens the real
    # yaw reward and (optionally) penalises the left-right stance asymmetry that
    # currently supplies the bias.
    _yw = float(os.environ.get("AS2W_YAW_WEIGHT", "1.5"))
    _ys = float(os.environ.get("AS2W_YAW_STD", "0.08"))
    _sw = float(os.environ.get("AS2W_SYM_W", "0.0"))
    cfg.rewards.track_ang_vel_z_exp.weight = _yw
    cfg.rewards.track_ang_vel_z_exp.params["std"] = _ys
    if _sw != 0.0:
        cfg.rewards.rolling_stance_sym = RewardTermCfg(
            func=stance_asymmetry_cost, weight=-abs(_sw))
    print("AS2W_YAW_ARM yaw_weight=%.2f yaw_std=%.3f sym_weight=%.2f" % (_yw, _ys, _sw),
          flush=True)
    # Wheel-support penalty is a RELATIVE weight against the yaw reward.  arm A doubled
    # the yaw reward (1.5 -> 3.0) but left this at -0.65, and its arc_left then tracked
    # in-band (1.09-1.10) while four-wheel support fell to 0.88-0.94, below the 0.95
    # gate (GOAL_STATUS 52).  This knob tests that imbalance as a single variable.
    _supw = float(os.environ.get("AS2W_SUPPORT_W", "-0.65"))
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=_supw)
    print("AS2W_SUPPORT_ARM wheel_support_weight=%.2f" % _supw, flush=True)
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


def stance_asymmetry_cost(env):
    """Left-right stance asymmetry in radians, summed over hip / thigh / calf.

    Layout of the 16-axis canonical order is hip x4, thigh x4, calf x4, foot x4 with
    legs FL, FR, RL, RR.  For each part the two PAIRS are (FL,FR) and (RL,RR), so the
    cost is |FL-FR| + |RL-RR| per part.  The four wheels (indices 12-15) are excluded.
    """
    import torch as _t
    q = env.scene["robot"].data.joint_pos[:, env.as2w_joint_ids][:, :12]
    d = _t.zeros(q.shape[0], device=q.device)
    for part in range(3):
        b = 4 * part
        d = d + (q[:, b + 0] - q[:, b + 1]).abs() + (q[:, b + 2] - q[:, b + 3]).abs()
    return d


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
