"""Non-inverted AS2-W get-up training, separate from locomotion weights."""
import argparse
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def train(args, physics):
    import torch
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_from_euler_xyz
    from as2w_isaac_flat_env_all import (
        FlatEnvironment, JOINTS, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
    )
    from as2w_blind_stair_gait import delayed_proprioception
    from as2w_recovery_control import augment_recovery, latch_handoff, recovery_roll_brace, staged_target, stand_integral
    from as2w_isaac_robust_env import robust_config, DelayedPosition, PredictiveDelayedPosition
    from as2w_training_safety import apply_safety, knee_surface_violation, stance_violation

    def pose_bank(path, family):
        rows = json.loads(path.read_text())[family]
        if not rows:
            raise ValueError(f"Pose bank has no {family} samples")
        values = [[r["height_m"], *r["root_wxyz"][1:], r["root_wxyz"][0], *r["leg_q"]]
                  for r in rows]
        bank = torch.tensor(values,device=args.device)
        if not torch.isfinite(bank).all():
            raise ValueError(f"Nonfinite get-up {family} pose bank")
        return bank

    supported = pose_bank(args.pose_bank,"prone") if args.pose_bank else None
    side_bank = (pose_bank(args.side_pose_bank,"left"),pose_bank(args.side_pose_bank,"right")) if args.side_pose_bank else ()

    def reset_posture(env, env_ids, pose_range, velocity_range):
        robot = env.scene["robot"]
        ids = slice(None) if env_ids is None else env_ids
        legs = env.as2w_joint_ids[:12]
        q = robot.data.default_joint_pos.torch[ids].clone()
        rnd = torch.rand((len(q), 7), device=env.device)
        kind = (rnd[:,0]*4).long()
        down = kind != 0
        side = kind >= 2
        side_sign = torch.where(kind == 2, -1., 1.)
        roll = .18*(rnd[:,1]-.5) + side*side_sign*(.38+.58*rnd[:,2])
        pitch = .20*(rnd[:,3]-.5) + (kind == 1)*(.22+.24*rnd[:,4])
        root = robot.data.default_root_state.torch[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        root[:,2] += down * (-.125 + .15*roll.sin().abs() + .10*pitch.sin().abs())
        root[:,3:7] = quat_from_euler_xyz(roll, pitch, .3*(rnd[:,5]-.5))
        q[:,legs[4:8]] += (.34+.12*rnd[:,6,None])*down[:,None]
        q[:,legs[8:12]] -= (.58+.14*rnd[:,6,None])*down[:,None]
        q[:,legs] += .09*(2.*torch.rand_like(q[:,legs])-1.)
        if supported is not None:
            index = torch.randint(len(supported), (len(q),), device=env.device)
            bank = supported[index]
            use = kind == 1
            root[use,2] = env.scene.env_origins[ids][use,2] + bank[use,0]
            root[use,3:7] = bank[use,1:5]
            q[:,legs] = torch.where(use[:,None], bank[:,5:], q[:,legs])
        for index, side_samples in enumerate(side_bank):
            selected = (kind == index+2) & (torch.rand(len(q),device=env.device)<args.side_pose_probability)
            sample = side_samples[torch.randint(len(side_samples),(len(q),),device=env.device)]
            root[selected,2] = env.scene.env_origins[ids][selected,2]+sample[selected,0]
            root[selected,3:7] = sample[selected,1:5]
            q[:,legs] = torch.where(selected[:,None],sample[:,5:],q[:,legs])
        angle = q.new_tensor(list(env.cfg.as2w_safety["angle_rad"].values()))
        q[:,legs] = q[:,legs].clamp(angle[:,0]+.12, angle[:,1]-.12)
        dq = torch.zeros_like(q)
        dq[:,legs] = .10*(2.*torch.rand_like(q[:,legs])-1.)
        robot.write_root_pose_to_sim_index(root_pose=root[:,:7], env_ids=ids)
        robot.write_root_velocity_to_sim_index(root_velocity=root[:,7:13]*0., env_ids=ids)
        robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=ids)
        if not hasattr(env, "recovery_kind"):
            env.recovery_kind = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.recovery_kind[ids] = kind

    def pose(env):
        robot = env.scene["robot"]
        ids = env.as2w_joint_ids[:12]
        pos = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:,ids]
        velocity = robot.data.joint_vel.torch[:,ids]
        sensor = torch.cat((robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
                            pos, velocity), dim=1)
        return torch.cat((delayed_proprioception(env, sensor), env.action_manager.action), dim=1)

    def standing(env):
        robot = env.scene["robot"]
        height = robot.data.root_pos_w.torch[:,2] - env.scene.env_origins[:,2]
        upright = -robot.data.projected_gravity_b.torch[:,2]
        return height, upright

    def wheel_support(env):
        sensor = env.scene["contact_forces"]
        if not hasattr(env, "_getup_wheels"):
            env._getup_wheels = [sensor.body_names.index(j) for j in JOINTS[12:]]
        return sensor.data.net_forces_w.torch[:,env._getup_wheels,2] > 2.

    def recovery_reward(env):
        height, up = standing(env)
        robot = env.scene["robot"]
        legs = env.as2w_joint_ids[:12]
        error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,legs].square().mean(dim=1)
        rise = ((height-.14)/.22).clamp(0.,1.)
        upright = ((up-.25)/.72).clamp(0.,1.)
        speed = robot.data.root_lin_vel_b.torch.norm(dim=1)
        spin = robot.data.root_ang_vel_b.torch.norm(dim=1)
        support = wheel_support(env).float().mean(dim=1)
        return upright + 2.*rise*upright + .70*support*(height > .22) + (height > .30) * (
            torch.exp(-3.*error) - .6*speed.clamp(max=1.) - .15*spin.clamp(max=2.))

    def bilateral_support_reward(env):
        height, up = standing(env)
        sensor = env.scene["contact_forces"]
        wheel_support(env)
        load = sensor.data.net_forces_w.torch[:, env._getup_wheels, 2].clamp_min(0.)
        left = load[:, (0, 2)].sum(dim=1)
        right = load[:, (1, 3)].sum(dim=1)
        bilateral = (torch.minimum(left, right) / 35.).clamp(0., 1.)
        age = env.episode_length_buf * env.step_dt
        early = ((1.4 - age) / .6).clamp(0., 1.)
        lifting = ((height - .16) / .10).clamp(0., 1.)
        return bilateral * early * lifting * ((up + .1) / .7).clamp(0., 1.)

    def settled_stand_reward(env):
        height, up = standing(env)
        robot = env.scene["robot"]
        legs = env.as2w_joint_ids[:12]
        q_error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,legs]
        max_error = q_error.abs().amax(dim=1)
        speed = robot.data.root_lin_vel_b.torch.norm(dim=1)
        spin = robot.data.root_ang_vel_b.torch.norm(dim=1)
        upright = ((up-.85)/.10).clamp(0.,1.)
        risen = ((height-.26)/.05).clamp(0.,1.)
        support = wheel_support(env).float().mean(dim=1).square()
        fit = torch.exp(-.5*((max_error/.18).square()+(speed/.06).square()+(spin/.12).square()))
        return upright*risen*support*fit

    def overturned(env):
        _, up = standing(env)
        return up < -.10

    def recovery_audit_phase(env):
        height, upright = standing(env)
        env._recovery_audit_established |= (height >= .31) & (upright >= .95)
        standing_guard = env._recovery_audit_established | ((height > .29) & (upright > .85))
        return standing_guard, upright < -.10

    def upright_stance_guard(env):
        h, up = standing(env)
        return (h > .29) & (up > .85) & stance_violation(env)

    def upright_surface_guard(env):
        h, up = standing(env)
        return (h > .29) & (up > .85) & knee_surface_violation(env)

    def upright_nonwheel_guard(env, threshold, sensor_cfg):
        h, up = standing(env)
        return (h > .29) & (up > .85) & mdp.illegal_contact(env, threshold, sensor_cfg)

    @configclass
    class Obs(ObservationGroupCfg):
        state = ObservationTermCfg(func=pose)
        enable_corruption = False

    class RecoveryEnvironment(FlatEnvironment):
        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, "_recovery_audit_established"):
                self._recovery_audit_established[ids] = False
            if hasattr(self, "getup_initial"):
                self._capture(ids)

        def _capture(self, ids):
            robot = self.scene["robot"]
            leg = self.as2w_joint_ids[:12]
            q = robot.data.joint_pos.torch[ids][:,leg]
            q0 = robot.data.default_joint_pos.torch[ids][:,leg]
            self.getup_initial[ids] = (q-q0)/.35
            self.steady[ids] = 0
            self.handoff_frames[ids] = 0
            self.stand_bias[ids] = 0.

        def step(self, action):
            if not hasattr(self, "getup_initial"):
                self.getup_initial = torch.zeros_like(action)
                self.steady = torch.zeros_like(self.episode_length_buf)
                self.handoff_frames = torch.zeros_like(self.episode_length_buf)
                self.stand_bias = torch.zeros_like(action)
                self._capture(slice(None))
            robot = self.scene["robot"]
            h, up = standing(self)
            quiet = robot.data.root_lin_vel_b.torch.norm(dim=1) < .08
            stable = (h>(.27 if args.low_support_handoff else .31)) & (up>.95) & quiet
            if args.staged_rise:
                legs = self.as2w_joint_ids[:12]
                q_error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,legs]
                stable &= wheel_support(self).all(dim=1)
                if args.support_handoff:
                    stable &= robot.data.root_ang_vel_b.torch.norm(dim=1)<.18
                else:
                    stable &= q_error.abs().amax(dim=1)<.20
            self.steady = torch.where(stable, self.steady+1, 0)
            if args.support_handoff:
                self.handoff_frames, exit_blend = latch_handoff(
                    self.steady, self.handoff_frames, h, up, self.step_dt,
                    min_height=.25 if args.low_support_handoff else .28)
            else:
                exit_blend = ((self.steady*self.step_dt-.45)/.65).clamp(0.,1.)
            entry = ((self.episode_length_buf*self.step_dt-.22)/.50).clamp(0.,1.)
            if args.staged_rise:
                age = self.episode_length_buf * self.step_dt
                target = staged_target(self.getup_initial, action, age)
                if args.roll_brace:
                    roll = self._step_delivered[:,4]
                    target = target + recovery_roll_brace(roll,age)
            else:
                target = self.getup_initial.lerp(action, entry[:,None])
            output = target * (1.-exit_blend[:,None])
            if args.fixed_stand_integral:
                measured_error = self._step_delivered[:,6:18]
                self.stand_bias = stand_integral(
                    self.stand_bias, measured_error, exit_blend, h, up, self.step_dt,
                    min_height=.25 if args.low_support_handoff else .29)
                output += exit_blend[:,None]*self.stand_bias/.35
            return super().step(output)

    args.profile = "expanded"
    args.command_stage = "all"
    args.zero_memory = args.zero_focus = args.direction_focus = False
    cfg = robust_config(args, physics)
    cfg.as2w_step_packet_loss = args.packet_loss
    cfg.episode_length_s = 12.
    cfg.as2w_guard_diagnostics = args.iterations < 100 and args.eval_checkpoint is None
    cfg.actions.joint_pos.class_type = PredictiveDelayedPosition if args.predictive_guard else DelayedPosition
    cfg.as2w_predictive_target_margin = .06
    cfg.actions.wheel_vel = None
    if args.rise_gain != 55.:
        for name in ("hip","thigh","calf"):
            cfg.scene.robot.actuators[name].stiffness = args.rise_gain
            cfg.scene.robot.actuators[name].damping = 3.2
    cfg.scene.robot.actuators["foot"].stiffness = 25.
    cfg.scene.robot.actuators["foot"].damping = 4.
    cfg.observations.policy = Obs()
    cfg.events.reset_base.func = reset_posture
    cfg.events.reset_robot_joints = None
    for name in vars(cfg.rewards):
        if name not in {"dof_torques_l2", "action_rate_l2", "dof_pos_limits"}:
            setattr(cfg.rewards, name, None)
    cfg.rewards.getup = RewardTermCfg(func=recovery_reward, weight=1.0)
    if args.bilateral_support_reward:
        cfg.rewards.bilateral_support = RewardTermCfg(func=bilateral_support_reward, weight=.55)
    if args.settled_stand_reward:
        cfg.rewards.settled_stand = RewardTermCfg(func=settled_stand_reward, weight=.85)
    cfg.terminations.base_contact = None
    limits = apply_safety(cfg, args.urdf, surface_guard=args.recovery_symmetry)
    cfg.actions.joint_pos.clip = {name: (low+.06, high-.06)
                                  for name,(low,high) in limits["angle_rad"].items()}
    cfg.terminations.as2w_stance_violation.func = upright_stance_guard
    cfg.terminations.as2w_nonwheel_contact.func = upright_nonwheel_guard
    if args.recovery_symmetry:
        cfg.terminations.as2w_knee_surface.func = upright_surface_guard
    cfg.terminations.as2w_overturned = cfg.terminations.as2w_angle_violation.replace(func=overturned)
    if args.eval_checkpoint is not None:
        from as2w_training_safety import configure_motion_substep_audit, install_motion_substep_audit
        configure_motion_substep_audit(cfg)
    env = RecoveryEnvironment(cfg)
    try:
        substep_contract = None
        if args.eval_checkpoint is not None:
            env._recovery_audit_established = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            substep_contract = install_motion_substep_audit(env, recovery_phase=recovery_audit_phase)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=2.)
        shape = wrapped.get_observations()["policy"].shape[-1]
        if shape != 42 or wrapped.num_actions != 12:
            raise RuntimeError(f"Unexpected recovery dimensions {shape}, {wrapped.num_actions}")
        contract = {
            "policy": "noninverted_getup", "policy_width": shape,
            "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            "actions": 12, "iterations": args.iterations, "num_envs": args.num_envs, "seed": args.seed,
            "step_dt_s": env.step_dt, "simulation_safety": limits,
            "initial_families": ["near_stand", "prone" if supported is not None else "prone_proxy",
                                 "left_tilt", "right_tilt"],
            "initial_contact": "supported nonwheel contact allowed during recovery only",
            "supported_pose_bank": str(args.pose_bank) if args.pose_bank else None,
            "supported_side_bank": str(args.side_pose_bank) if args.side_pose_bank else None,
            "supported_side_probability": args.side_pose_probability if args.side_pose_bank else 0.,
            "staged_rise": args.staged_rise, "leg_kp": args.rise_gain,
            "support_handoff": args.support_handoff,
            "fixed_stand_integral": args.fixed_stand_integral,
            "low_support_handoff": args.low_support_handoff,
            "recovery_symmetry": args.recovery_symmetry,
            "roll_brace": args.roll_brace,
            "observation_packet_loss": args.packet_loss,
            "predictive_guard": args.predictive_guard,
            "settled_stand_reward": args.settled_stand_reward,
            "bilateral_support_reward": args.bilateral_support_reward,
        }
        contract_path = args.out/"contract.json"
        if args.resume_checkpoint:
            previous = json.loads(contract_path.read_text())
            previous.setdefault("supported_side_bank",None)
            previous.setdefault("supported_side_probability",0.)
            previous.setdefault("roll_brace",False)
            previous.setdefault("predictive_guard",False)
            previous.setdefault("settled_stand_reward",False)
            previous.setdefault("bilateral_support_reward",False)
            previous.setdefault("observation_packet_loss",0.)
            ignored = {"num_envs","initial_checkpoint"}
            for key,value in json.loads(json.dumps(contract)).items():
                if key not in ignored and previous.get(key)!=value:
                    raise ValueError(f"Recovery resume contract differs at {key}")
        else:
            if args.eval_checkpoint:
                contract.update(checkpoint_record(args.eval_checkpoint))
                contract["motion_substep_audit"] = substep_contract
            contract_path.write_text(json.dumps(contract,indent=2)+"\n")
        runner_cfg = UnitreeGo2FlatPPORunnerCfg()
        runner_cfg.seed = args.seed
        runner_cfg.device = args.device
        runner_cfg.max_iterations = args.iterations
        runner_cfg.save_interval = min(25, args.iterations)
        runner_cfg.actor.distribution_cfg.init_std = .35
        runner_cfg.algorithm.learning_rate = 2e-4
        if args.recovery_symmetry:
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
            runner_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
                data_augmentation_func=augment_recovery, use_data_augmentation=True,
                use_mirror_loss=True, mirror_loss_coeff=.20)
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(wrapped, runner_cfg, log_dir=str(args.out))
        if args.probe_prior:
            from eval_as2w_isaac_getup import evaluate
            def zero_actor(obs):
                policy = obs["policy"]
                return policy.new_zeros((policy.shape[0], 12))
            evaluate(env, wrapped, zero_actor, args.out)
            return
        if args.eval_checkpoint:
            from eval_as2w_isaac_getup import evaluate
            runner.load(str(args.eval_checkpoint), load_cfg={"actor": True, "critic": True},
                        map_location=args.device)
            evaluate(env, wrapped, runner.get_inference_policy(args.device), args.out)
            return
        if args.init_checkpoint:
            runner.load(str(args.init_checkpoint), load_cfg={"actor": True, "critic": True},
                        map_location=args.device)
        remaining = args.iterations
        if args.resume_checkpoint:
            runner.load(str(args.resume_checkpoint),map_location=args.device)
            completed = runner.current_learning_iteration+1
            remaining -= completed
            if remaining<=0:
                raise ValueError("No recovery training iterations remain")
            runner.current_learning_iteration = completed
            resume_info = dict(checkpoint=str(args.resume_checkpoint.resolve()),
                               completed_iterations=completed,remaining_iterations=remaining,
                               prior_num_envs=previous["num_envs"],resume_num_envs=args.num_envs)
            (args.out/f"resume-{completed-1}.json").write_text(json.dumps(resume_info,indent=2)+"\n")
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
        print("AS2W_GETUP_COMPLETE", args.iterations, flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--usd", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--rise-gain", type=float, choices=(55.,85.), default=55.)
    p.add_argument("--support-handoff", action="store_true")
    p.add_argument("--fixed-stand-integral", action="store_true")
    p.add_argument("--low-support-handoff", action="store_true")
    p.add_argument("--staged-rise", action="store_true")
    p.add_argument("--probe-prior", action="store_true")
    p.add_argument("--pose-bank", type=Path)
    p.add_argument("--side-pose-bank",type=Path)
    p.add_argument("--side-pose-probability",type=float,default=.50)
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument("--eval-checkpoint", type=Path)
    checkpoints.add_argument("--init-checkpoint", type=Path)
    checkpoints.add_argument("--resume-checkpoint",type=Path)
    p.add_argument("--recovery-symmetry", action="store_true")
    p.add_argument("--bilateral-support-reward", action="store_true")
    p.add_argument("--roll-brace", action="store_true")
    p.add_argument("--predictive-guard", action="store_true")
    p.add_argument("--settled-stand-reward",action="store_true")
    p.add_argument("--packet-loss",type=float,default=0.)
    p.add_argument("--seed", type=int, default=292)
    add_launcher_args(p)
    args = p.parse_args()
    if not 0. <= args.packet_loss <= .10:
        p.error("Packet loss must be between 0 and .10")
    if not 0. <= args.side_pose_probability <= 1.:
        p.error("Side pose probability must be between zero and one")
    if args.roll_brace and not args.staged_rise:
        p.error("--roll-brace requires --staged-rise")
    if args.low_support_handoff and not args.fixed_stand_integral:
        p.error("--low-support-handoff requires --fixed-stand-integral")
    if args.fixed_stand_integral and not (args.support_handoff and args.staged_rise):
        p.error("--fixed-stand-integral requires --support-handoff and --staged-rise")
    if args.resume_checkpoint:
        if args.resume_checkpoint.resolve().parent!=args.out.resolve() or not (args.out/"contract.json").is_file():
            p.error("Recovery resume requires a checkpoint and contract from the same run directory")
        if args.probe_prior:
            p.error("Recovery resume cannot be combined with a prior probe")
    elif (args.out/"contract.json").exists():
        p.error("Use a fresh run directory")
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        train(args, physics)


if __name__ == "__main__":
    main()