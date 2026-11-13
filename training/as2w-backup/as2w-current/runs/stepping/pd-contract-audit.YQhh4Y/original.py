"""Train a fresh AS2-W flat stepping policy with four braked wheels."""
import argparse
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def train(args, physics):
    import torch
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass
    from as2w_isaac_flat_env_all import (
        GOALS, JOINTS, FlatEnvironment, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, configuration, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
    )

    period = 0.80
    duty = 0.62
    radius = 0.086
    legs = ("FL", "FR", "RL", "RR")
    wheel_names = JOINTS[12:]

    def phase(env):
        elapsed = env.episode_length_buf*env.step_dt
        if getattr(env.cfg,"as2w_sequence",False):
            elapsed = (elapsed-2.).clamp_min(0.)
        return (elapsed / getattr(env.cfg, "as2w_step_period", period)).remainder(1.0)

    def observe(env):
        robot = env.scene["robot"]
        ids = env.as2w_joint_ids[:12]
        q = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, ids]
        dq = robot.data.joint_vel.torch[:, ids]
        clock = 2.0 * torch.pi * phase(env)
        command = env.command_manager.get_command("base_velocity")
        sensor = torch.cat((robot.data.root_ang_vel_b.torch,
                            robot.data.projected_gravity_b.torch, q, dq), dim=1)
        if args.curriculum != "legacy":
            from as2w_blind_stair_gait import delayed_proprioception
            sensor = delayed_proprioception(env, sensor)
        return torch.cat((command, sensor,
                          env.action_manager.action, clock.sin()[:, None],
                          clock.cos()[:, None]), dim=1)

    def feet(env):
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        if not hasattr(env, "_stepping_body_ids"):
            env._stepping_body_ids = [robot.body_names.index(name) for name in wheel_names]
            env._stepping_contact_ids = [sensor.body_names.index(name) for name in wheel_names]
        z = robot.data.body_pos_w.torch[:, env._stepping_body_ids, 2]
        height = z - env.scene.env_origins[:, None, 2] - radius
        force = sensor.data.net_forces_w.torch[:, env._stepping_contact_ids]
        return height, force[..., 2] > 4.0

    def gait_contact(env):
        _, contact = feet(env)
        cycle = (phase(env)[:, None] + phase_offsets).remainder(1.0)
        moving = env.command_manager.get_command("base_velocity").abs().amax(dim=1) > .015
        swing = (cycle >= duty) & moving[:, None]
        return (contact == ~swing).float().mean(dim=1)

    def swing_clearance(env):
        height, _ = feet(env)
        cycle = (phase(env)[:, None] + phase_offsets).remainder(1.0)
        moving = env.command_manager.get_command("base_velocity").abs().amax(dim=1) > .015
        swing_progress = ((cycle - duty) / (1.0 - duty)).clamp(0.0, 1.0)
        target = .04 * torch.sin(torch.pi * swing_progress)
        match = torch.exp(-((height - target) / .025).square())
        return (match * (cycle >= duty)).sum(dim=1) * moving / 2.0

    def stance_slip(env):
        _, contact = feet(env)
        robot = env.scene["robot"]
        xy = robot.data.body_lin_vel_w.torch[:, env._stepping_body_ids, :2]
        return (xy.square().sum(dim=-1) * contact).mean(dim=1)

    def sequence_swing_clearance(env):
        from as2w_stepping_control import sequence_swing
        from as2w_training_safety import knee_clearance

        _, contact = feet(env)
        robot = env.scene["robot"]
        hubs = robot.data.body_pos_w.torch[:,env._stepping_body_ids]
        height = knee_clearance(env,hubs) - radius
        _, lift, _ = sequence_swing(phase(env))
        active = env.command_manager.get_command("base_velocity").abs().amax(dim=1) > .015
        desired = .050 * lift
        fit = torch.exp(-((height-desired)/.025).square())
        return (fit * (lift>.15) * ~contact).sum(dim=1) * active

    def support_aware_swing_reward(env):
        from as2w_stepping_control import sequence_swing
        from as2w_training_safety import knee_clearance

        _, contact = feet(env)
        hubs = env.scene["robot"].data.body_pos_w.torch[:, env._stepping_body_ids]
        clearance = knee_clearance(env, hubs) - radius
        _, lift, _ = sequence_swing(phase(env))
        moving = env.command_manager.get_command("base_velocity").abs().amax(dim=1) > .015

        # Reward clearance on the scheduled wheel only while the others carry load.
        stance_count = contact.sum(dim=1, keepdim=True) - contact.long()
        clearance_fraction = (clearance / (.035 * lift).clamp_min(.015)).clamp(0., 1.)
        swing_score = (clearance_fraction * lift * ~contact * (stance_count == 3)).sum(dim=1)
        return swing_score * moving

    if args.go2_support_reference:
        import numpy as np

        teacher_support = []
        for name in ("stairs_up8cm","stairs_down"):
            with np.load(args.go2_support_reference/f"{name}.npz",allow_pickle=False) as data:
                contact = torch.as_tensor(data["go2_contact"],device=args.device,dtype=torch.float32)
            if contact.ndim!=2 or contact.shape[1]!=4 or not len(contact):
                raise ValueError(f"Invalid Go2 support trace {name}")
            teacher_support.append(contact)

    def go2_support_reward(env):
        _, contact = feet(env)
        age = env.episode_length_buf*env.step_dt - 2.
        frame = (age.clamp_min(0.)*25.).long()
        columns = env.scene.terrain.terrain_types
        moving = env.command_manager.get_command("base_velocity")[:,0]>.04
        scores = torch.zeros_like(age)
        for direction, reference in enumerate(teacher_support):
            valid = ((columns>=4+2*direction) & (columns<6+2*direction)
                     & moving & (age>=0.) & (frame<len(reference)))
            sample = frame.clamp(max=len(reference)-1)
            expected = (reference[(sample-2).clamp(min=0)]+reference[sample]
                        +reference[(sample+2).clamp(max=len(reference)-1)])/3.
            fit = 1.-(contact.float()-expected).abs().mean(dim=1)
            scores += valid*fit
        return scores

    def terrain_for_value(env):
        from as2w_training_safety import knee_clearance

        robot = env.scene["robot"]
        base = robot.data.root_pos_w.torch
        if not hasattr(env,"_critic_terrain_points"):
            env._critic_terrain_points = base.new_tensor([
                (x,y) for x in (-.35,0.,.25,.45,.60) for y in (-.18,0.,.18)
            ])
        x,y,z,w = robot.data.root_quat_w.torch.unbind(dim=1)
        yaw = torch.atan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
        offsets = env._critic_terrain_points
        qx = yaw.cos()[:,None]*offsets[:,0]-yaw.sin()[:,None]*offsets[:,1]
        qy = yaw.sin()[:,None]*offsets[:,0]+yaw.cos()[:,None]*offsets[:,1]
        probes = base[:,None,:]+torch.stack((qx,qy,torch.zeros_like(qx)),dim=-1)
        return knee_clearance(env,probes).clamp(-.3,.8)

    @configclass
    class TerrainValueObservations(ObservationGroupCfg):
        grade = ObservationTermCfg(func=terrain_for_value)
        enable_corruption = False

    @configclass
    class SteppingObservations(ObservationGroupCfg):
        state = ObservationTermCfg(func=observe)
        enable_corruption = False

    args.command_stage = "all"
    args.direction_focus = True
    args.zero_memory = False
    args.zero_focus = False
    phase_offsets = torch.tensor((0., .5, .5, 0.), device=args.device)
    cfg = configuration(args, physics)
    if args.eval_leg_pd is not None:
        groups = ("hip", "thigh", "calf") if args.eval_leg_pd_group == "all" else (args.eval_leg_pd_group,)
        for name in groups:
            actuator_cfg = cfg.scene.robot.actuators[name]
            actuator_cfg.stiffness, actuator_cfg.damping = args.eval_leg_pd
    cfg.actions.wheel_vel = None
    cfg.actions.joint_pos.scale = .30
    wheel = cfg.scene.robot.actuators["foot"]
    wheel.stiffness = 45.
    wheel.damping = 4.
    cfg.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(JOINTS[:12]))
    cfg.observations.policy = SteppingObservations()
    cfg.rewards.rear_posture = None
    cfg.rewards.gait_contact = RewardTermCfg(func=gait_contact, weight=.8)
    cfg.rewards.swing_clearance = RewardTermCfg(func=swing_clearance, weight=1.2)
    cfg.rewards.stance_slip = RewardTermCfg(func=stance_slip, weight=-2.)
    limits = None
    env_class = FlatEnvironment
    if args.curriculum != "legacy":
        from as2w_blind_stair_gait import BlindStairEnvironment, gait_contacts, stair_config
        args.profile = "expanded"
        args.stairs = args.curriculum.startswith("stairs")
        cfg, limits = stair_config(args, physics, SteppingObservations())
        cfg.as2w_eval_body_raise_m = args.eval_body_raise_m
        cfg.rewards.gait_contact = RewardTermCfg(func=gait_contacts, weight=1.0)
        if args.directed_stair_curriculum:
            from as2w_blind_stair_gait import stair_course_progress
            cfg.rewards.stair_course_progress = RewardTermCfg(func=stair_course_progress,weight=1.6)
        if args.sequence_shaped_reward:
            cfg.rewards.swing_clearance = RewardTermCfg(func=sequence_swing_clearance, weight=1.4)
            cfg.rewards.stance_slip = RewardTermCfg(func=stance_slip, weight=-2.)
            cfg.rewards.signed_tracking.weight = 1.0
        if args.support_aware_swing:
            cfg.rewards.support_aware_swing = RewardTermCfg(func=support_aware_swing_reward, weight=1.4)
        if args.privileged_stair_critic:
            cfg.observations.terrain_value = TerrainValueObservations()
        if args.go2_support_reference:
            cfg.rewards.go2_support_reference = RewardTermCfg(func=go2_support_reward,weight=.35)
        env_class = BlindStairEnvironment
        if args.eval_checkpoint is not None and cfg.as2w_sequence:
            cfg.as2w_motion_end_s = 27.
            cfg.episode_length_s = 32.
    if args.calf_clearance_reward:
        import numpy as np
        from itertools import product
        from isaaclab.utils.math import quat_apply
        from as2w_knee_envelope import calf_hull_vertices
        from as2w_training_safety import knee_clearance

        directions = np.asarray([axis for axis in product((-1, 0, 1), repeat=3) if any(axis)])
        hulls = calf_hull_vertices(args.urdf.parent / "meshes")
        probes = torch.tensor(np.stack([hull[(hull @ directions.T).argmax(axis=0)] for hull in hulls]),
                              device=args.device)

        def calf_clearance_cost(env):
            robot = env.scene["robot"]
            if not hasattr(env, "_calf_clearance_ids"):
                env._calf_clearance_ids = [robot.body_names.index(j) for j in JOINTS[8:12]]
            ids = env._calf_clearance_ids
            pos = robot.data.body_pos_w.torch[:, ids, None, :]
            quat = robot.data.body_quat_w.torch[:, ids, None, :].expand(-1, -1, probes.shape[1], -1)
            world = pos + quat_apply(quat, probes.expand(env.num_envs, -1, -1, -1))
            gaps = knee_clearance(env, world.flatten(1, 2), padded=False).reshape(env.num_envs, 4, -1)
            return ((.025 - gaps.amin(dim=2)).clamp(0., .05) / .025).square().amax(dim=1)

        cfg.rewards.stepping_calf_clearance = RewardTermCfg(func=calf_clearance_cost, weight=-2.)
    if args.stance_margin_reward:
        from as2w_training_safety import rolling_knee_margin_cost, stepping_stance_margin_cost

        cfg.rewards.stepping_stance_margin = RewardTermCfg(
            func=stepping_stance_margin_cost, weight=-.5)
        cfg.rewards.stepping_knee_margin = RewardTermCfg(
            func=rolling_knee_margin_cost, weight=-2., params={"margin_m": .08})
    if args.probe_dynamics or args.probe_single_lift:
        cfg.events.bounded_push = None
        cfg.as2w_guard_diagnostics = True
    motion_audit = limits is not None and args.eval_checkpoint is not None
    if motion_audit:
        from as2w_training_safety import configure_motion_substep_audit, install_motion_substep_audit
        configure_motion_substep_audit(cfg)
    env = env_class(cfg)
    try:
        substep_contract = install_motion_substep_audit(env) if motion_audit else None
        wrapped = RslRlVecEnvWrapper(env, clip_actions=2.)
        obs = wrapped.get_observations()["policy"]
        if obs.shape[-1] != 47 or wrapped.num_actions != 12:
            raise RuntimeError(f"Unexpected stepping policy dimensions: {obs.shape}, {wrapped.num_actions}")
        feet(env)
        robot = env.scene["robot"]
        wheel_ids = env.as2w_joint_ids[12:]
        if robot.data.joint_pos_target.torch[:, wheel_ids].abs().max().item() > 1e-6:
            raise RuntimeError("Wheel brake target is not zero")
        contract = {
            "backend": "isaacsim_physx", "policy": "as2w_omnidirectional_stepping",
            "motion_substep_audit": substep_contract,
            "curriculum": args.curriculum, "simulation_safety": limits,
            "calibrated_sequence": getattr(cfg, "as2w_sequence", False),
            "sequence_stride_gain": args.sequence_stride,
            "sequence_shaped_reward": args.sequence_shaped_reward,
            "directed_stair_curriculum": args.directed_stair_curriculum,
            "support_aware_swing": args.support_aware_swing,
            "observation_packet_loss": args.packet_loss,
            "target_margin_rad": args.target_margin,
            "stance_margin_reward": args.stance_margin_reward,
            "calf_clearance_reward": args.calf_clearance_reward,
            "sequence_attitude": args.sequence_attitude,
            "predictive_guard": args.predictive_guard,
            "privileged_stair_critic": args.privileged_stair_critic,
            "actor_terrain_inputs": False,
            "imu_support_balance": getattr(cfg, "as2w_support_balance", False),
            "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            "seed": args.seed, "num_envs": args.num_envs,
            "iteration_budget": args.iterations, "step_dt": env.step_dt,
            "gait": "three_support_crawl" if getattr(cfg, "as2w_crawl", False) else "diagonal_trot",
            "legs": legs, "phase_offsets": [0.,.5,.25,.75] if getattr(cfg, "as2w_crawl", False)
                else phase_offsets.tolist(),
            "period_s": getattr(cfg, "as2w_step_period", period),
            "stance_fraction": getattr(cfg, "as2w_step_duty", duty),
            "clearance_m": .082 if getattr(cfg, "as2w_crawl", False) else
                (.064 if args.curriculum != "legacy" else .04),
            "commands": GOALS, "policy_width": 47, "joint_actions": JOINTS[:12],
            "observations": ["command", "body_gyro", "gravity", "leg_q",
                             "leg_dq", "previous_leg_action", "phase_sin_cos"],
            "wheel_position_target_rad": 0., "wheel_kp": 45., "wheel_kd": 4.,
            "action_clip": 2., "leg_action_scale": .30,
        }
        if args.go2_support_reference:
            contract["go2_support_reference"] = {
                name: checkpoint_record(args.go2_support_reference/f"{name}.npz")
                for name in ("stairs_up8cm","stairs_down")
            }
        runner_cfg = UnitreeGo2FlatPPORunnerCfg()
        runner_cfg.seed = args.seed
        runner_cfg.device = args.device
        runner_cfg.max_iterations = args.iterations
        runner_cfg.save_interval = min(25, args.iterations)
        runner_cfg.actor.distribution_cfg.init_std = .45
        runner_cfg.algorithm.learning_rate = 3e-4
        if args.privileged_stair_critic:
            runner_cfg.obs_groups = {"actor":["policy"],"critic":["policy","terrain_value"]}
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(wrapped, runner_cfg, log_dir=str(args.out))
        if args.probe_single_lift:
            from probe_as2w_stair_fk import single_lift
            single_lift(env, wrapped, args.out, sweep=args.probe_lift_sweep)
            return
        if args.probe_dynamics:
            from probe_as2w_stair_fk import dynamics
            dynamics(env, wrapped, args.out)
            return
        if args.probe_fk:
            from probe_as2w_stair_fk import evaluate
            evaluate(env, args.out)
            return
        if args.eval_checkpoint is not None:
            evaluation_contract = dict(contract,**checkpoint_record(args.eval_checkpoint))
            evaluation_contract["leg_pd"] = args.eval_leg_pd or (55., 2.)
            evaluation_contract["leg_pd_group"] = args.eval_leg_pd_group
            evaluation_contract["body_raise_m"] = args.eval_body_raise_m
            (args.out/"contract.json").write_text(json.dumps(evaluation_contract,indent=2)+"\n")
            runner.load(str(args.eval_checkpoint),load_cfg={"actor":True},map_location=args.device)
            actor = runner.get_inference_policy(args.device)
            if args.curriculum.startswith("stairs"):
                from eval_as2w_isaac_blind_stairs import evaluate
                evaluate(env, wrapped, actor, args.out)
            else:
                from eval_as2w_isaac_stepping import evaluate
                evaluate(env, wrapped, actor, GOALS, feet, args.out)
            return
        completed = 0
        contract_path = args.out / "contract.json"
        if args.extend_from is not None:
            previous = json.loads(contract_path.read_text())
            previous.setdefault("sequence_stride_gain",.10)
            previous.setdefault("sequence_shaped_reward",False)
            previous.setdefault("directed_stair_curriculum",False)
            previous.setdefault("support_aware_swing",False)
            previous.setdefault("observation_packet_loss",0.)
            previous.setdefault("target_margin_rad",0.)
            previous.setdefault("sequence_attitude",False)
            previous.setdefault("predictive_guard",False)
            previous.setdefault("calf_clearance_reward", False)
            previous.setdefault("privileged_stair_critic",False)
            previous.setdefault("actor_terrain_inputs",False)
            for key, value in json.loads(json.dumps(contract)).items():
                if key != "iteration_budget" and previous.get(key) != value:
                    raise ValueError(f"Incompatible stepping run field: {key}")
            runner.load(str(args.extend_from), map_location=args.device)
            completed = runner.current_learning_iteration + 1
            runner.current_learning_iteration = completed
            if completed >= args.iterations:
                raise ValueError("Extension budget must exceed completed iterations")
            contract["continued_after_iterations"] = completed
        if args.init_checkpoint:
            parent = json.loads((args.init_checkpoint.parent/"contract.json").read_text())
            target_contract = json.loads(json.dumps(contract))
            for key in ("policy_width","joint_actions","observations","period_s",
                        "sequence_stride_gain","sequence_shaped_reward"):
                if parent.get(key) != target_contract[key]:
                    raise ValueError(f"Warm-start stepping contract differs at {key}")
            same_critic = parent.get("privileged_stair_critic",False)==args.privileged_stair_critic
            contract["warm_start_critic"] = same_critic
            runner.load(str(args.init_checkpoint),load_cfg={"actor":True,"critic":same_critic},
                        map_location=args.device)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        runner.learn(num_learning_iterations=args.iterations - completed, init_at_random_ep_len=False)
        print("AS2W_STEPPING_COMPLETE", args.iterations, flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=600)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--eval-checkpoint", type=Path)
    parser.add_argument("--eval-leg-pd", nargs=2, type=float, metavar=("KP", "KD"))
    parser.add_argument("--eval-leg-pd-group", choices=("all", "hip", "thigh", "calf"), default="all")
    parser.add_argument("--eval-body-raise-m", type=float, default=0.)
    checkpoints.add_argument("--extend-from", type=Path)
    checkpoints.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--probe-lift-sweep", action="store_true")
    parser.add_argument("--probe-single-lift", action="store_true")
    parser.add_argument("--probe-dynamics", action="store_true")
    parser.add_argument("--probe-fk", action="store_true")
    parser.add_argument("--curriculum", choices=("legacy", "flat", "flat-crawl", "flat-support", "flat-slow", "flat-stiff", "flat-plant", "flat-preswing", "flat-sequence", "flat-sequence-guarded", "stairs", "stairs-crawl", "stairs-support", "stairs-slow", "stairs-stiff", "stairs-plant", "stairs-preswing", "stairs-sequence", "stairs-sequence-guarded"), default="legacy")
    parser.add_argument("--sequence-period", type=float, default=12.)
    parser.add_argument("--sequence-stride", type=float, default=.10)
    parser.add_argument("--sequence-shaped-reward", action="store_true")
    parser.add_argument("--directed-stair-curriculum",action="store_true")
    parser.add_argument("--go2-support-reference",type=Path)
    parser.add_argument("--support-aware-swing",action="store_true")
    parser.add_argument("--packet-loss",type=float,default=0.)
    parser.add_argument("--stance-margin-reward", action="store_true")
    parser.add_argument("--calf-clearance-reward", action="store_true")
    parser.add_argument("--target-margin",type=float,default=0.)
    parser.add_argument("--sequence-attitude",action="store_true")
    parser.add_argument("--predictive-guard",action="store_true")
    parser.add_argument("--privileged-stair-critic",action="store_true")
    parser.add_argument("--seed", type=int, default=285)
    add_launcher_args(parser)
    args = parser.parse_args()
    if not 0. <= args.eval_body_raise_m <= .04:
        parser.error("--eval-body-raise-m must be within 0–0.04 m")
    if args.eval_body_raise_m and (args.eval_checkpoint is None or not args.curriculum.endswith("-sequence-guarded")):
        parser.error("--eval-body-raise-m requires guarded sequence evaluation")
    if args.eval_leg_pd is not None and (args.eval_checkpoint is None or not all(0. < gain < float("inf") for gain in args.eval_leg_pd)):
        parser.error("--eval-leg-pd requires an evaluation checkpoint and finite positive gains")
    if args.support_aware_swing and (not args.curriculum.endswith("sequence-guarded")
                                   or args.go2_support_reference):
        parser.error("--support-aware-swing requires a guarded sequence and replaces Go2 wall-clock contact matching")
    if args.go2_support_reference and (args.curriculum!="stairs-sequence-guarded" or not args.directed_stair_curriculum):
        parser.error("Go2 support reference requires the directed guarded staircase curriculum")
    if args.stance_margin_reward and not args.curriculum.endswith("-sequence-guarded"):
        parser.error("Stance margin rewards require a guarded sequence")
    if not 0. <= args.target_margin <= .12 or (args.target_margin and not args.curriculum.endswith("-sequence-guarded")):
        parser.error("Target margin requires a guarded sequence and must be 0–.12 rad")
    if not 0. <= args.packet_loss <= .10 or (args.packet_loss and args.curriculum=="legacy"):
        parser.error("Packet loss must be 0–.10 and requires expanded observations")
    altered = args.sequence_period != 12. or args.sequence_stride != .10 or args.sequence_shaped_reward or args.sequence_attitude or args.predictive_guard
    if args.privileged_stair_critic and args.curriculum!="stairs-sequence-guarded":
        parser.error("Terrain value observations require the guarded stair curriculum")
    if args.directed_stair_curriculum and args.curriculum != "stairs-sequence-guarded":
        parser.error("Directed stair curriculum requires stairs-sequence-guarded")
    if altered and not args.curriculum.endswith("-sequence-guarded"):
        parser.error("Sequence tuning requires a guarded sequence curriculum")
    if not 3. <= args.sequence_period <= 12. or not .10 <= args.sequence_stride <= .65:
        parser.error("Sequence period must be 3–12 s and stride gain .10–.65 s")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.extend_from is not None and args.extend_from.resolve().parent != args.out.resolve():
        parser.error("Extend from a checkpoint in this same stepping run")
    if args.extend_from is None and (args.out / "contract.json").exists():
        parser.error("Choose a new stepping run directory")
    args.physics = "isaacsim_physx"
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        train(args, physics)


if __name__ == "__main__":
    main()
