"""AS2-W flat task and fresh PPO training, imported after Kit starts."""
import json
import xml.etree.ElementTree as ET

import torch
import isaaclab.sim as sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlSymmetryCfg, RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg
from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


JOINTS = tuple(f"{leg}_{part}" for part in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR"))
REAR = [2, 3, 6, 7, 10, 11]
MIRROR = [index ^ 1 for index in range(16)]
GOALS = ((.10, 0., 0.), (-.10, 0., 0.), (0., .10, 0.), (0., -.10, 0.),
         (.071, .071, 0.), (.071, -.071, 0.), (-.071, .071, 0.), (-.071, -.071, 0.),
         (0., 0., .30), (0., 0., -.30))


class FlatCommands(mdp.UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.targets = torch.zeros_like(self.vel_command_b)
        goals = {"sagittal": GOALS[:2], "heading": GOALS[:2] + GOALS[-2:], "all": GOALS}
        self.goals = torch.tensor(goals[cfg.as2w_stage], device=self.device)
        self.static = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = 0.
        self.static[ids] = torch.rand_like(self.static[ids], dtype=torch.float32) < .20
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        count = self.targets[env_ids].shape[0]
        sampled = self.goals[torch.randint(len(self.goals), (count,), device=self.device)]
        stop = self.static[env_ids] | (torch.rand(count, device=self.device) < .25)
        self.targets[env_ids] = torch.where(stop[:, None], 0., sampled)

    def _update_command(self):
        goal = self.targets
        if getattr(self._env.cfg,"as2w_ignore_lateral_command",False):
            goal = goal.clone()
            goal[:,1] = 0.
            self.vel_command_b[:,1] = 0.
        self.vel_command_b.lerp_(goal, .12)
        snap = (goal == 0.).all(dim=-1) & (self.vel_command_b.abs().amax(dim=-1) < .005)
        self.vel_command_b[snap] = 0.


def observation(env):
    robot = env.scene["robot"]
    q = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:, env.as2w_joint_ids]
    q = torch.cat((q[:, :12], (q[:, 12:] + 2. * torch.pi) % (4. * torch.pi) - 2. * torch.pi), dim=1)
    dq = robot.data.joint_vel.torch[:, env.as2w_joint_ids]
    command = env.command_manager.get_command("base_velocity")
    clock = 2. * torch.pi * env.episode_length_buf * env.step_dt / .62
    frame = torch.cat((command - command.new_tensor((.4, 0., 0.)),
                      robot.data.root_ang_vel_b.torch, robot.data.projected_gravity_b.torch,
                      torch.full_like(clock[:, None], .4), q, dq, env.action_manager.action,
                      clock.sin()[:, None], clock.cos()[:, None]), dim=-1)
    if not env.cfg.as2w_zero_memory:
        return frame
    wheel = q[:, 12:]
    zero = (command == 0.).all(dim=-1)
    if not hasattr(env, "_wheel_reference"):
        env._wheel_reference = wheel.clone()
        env._was_zero = torch.zeros_like(zero)
    entered = zero & (~env._was_zero | (env.episode_length_buf == 0))
    env._wheel_reference[entered] = wheel[entered]
    env._was_zero.copy_(zero)
    delta = (wheel - env._wheel_reference + 2. * torch.pi) % (4. * torch.pi) - 2. * torch.pi
    return torch.cat((frame, (delta / .20).clamp(-3., 3.) * zero[:, None], zero[:, None]), dim=-1)


def rear_error(env):
    robot = env.scene["robot"]
    q = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    return q[:, [env.as2w_joint_ids[index] for index in REAR]].square().mean(dim=-1)


def signed_tracking(env):
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    velocity = robot.data.root_lin_vel_b.torch[:, :2]
    yaw = robot.data.root_ang_vel_b.torch[:, 2]
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    along = (velocity * command[:, :2]).sum(dim=-1) / speed.clamp(min=.02).square()
    error = torch.linalg.vector_norm(velocity - command[:, :2], dim=-1) / speed.clamp(min=.02)
    translation = (along.clamp(-1., 1.) - .5 * error.clamp(max=3.)) * (speed > .02)
    requested_yaw = command[:, 2]
    yaw_scale = requested_yaw.abs().clamp(min=.05)
    rotation = ((yaw * requested_yaw.sign() / yaw_scale).clamp(-1., 1.)
                - .5 * ((yaw - requested_yaw).abs() / yaw_scale).clamp(max=3.))
    return translation + rotation * (requested_yaw.abs() > .05)


def mirror_joints(values):
    mirrored = values[..., MIRROR].clone()
    mirrored[..., :4] *= -1.
    return mirrored


def mirror_flat(env, obs=None, actions=None):
    mirrored_obs = mirrored_actions = None
    if obs is not None:
        reflected = obs.clone()
        frame = reflected["policy"]
        frame[..., [1, 2, 3, 5, 7]] *= -1.
        for start in (10, 26, 42):
            frame[..., start:start + 16] = mirror_joints(frame[..., start:start + 16])
        if frame.shape[-1] == 65:
            frame[..., 60:64] = frame[..., [61, 60, 63, 62]]
        mirrored_obs = torch.cat((obs, reflected), dim=0)
    if actions is not None:
        mirrored_actions = torch.cat((actions, mirror_joints(actions)), dim=0)
    return mirrored_obs, mirrored_actions


def zero_motion(env, focused=False):
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("base_velocity")
    speed = torch.linalg.vector_norm(robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
    yaw = robot.data.root_ang_vel_b.torch[:, 2].abs()
    xy = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    displacement = torch.linalg.vector_norm(xy, dim=-1)
    static = env.command_manager.get_term("base_velocity").static
    settled = env.episode_length_buf * env.step_dt >= 1.
    if focused:
        x, y, z, w = robot.data.root_quat_w.torch.unbind(dim=-1)
        heading = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square())).abs()
        wheel_effort = env.action_manager.action[:, 12:].square().mean(dim=-1)
        anchor = (displacement / .02).clamp(max=8.) + (heading / .05236).clamp(max=4.)
        penalty = speed / .025 + yaw / .08 + (static & settled) * (anchor + 4. * wheel_effort)
    else:
        penalty = speed / .05 + yaw / .10 + (static & settled) * (displacement / .02).clamp(max=2.)
    return (cmd == 0.).all(dim=-1) * penalty


@configclass
class PolicyObservations(ObservationGroupCfg):
    state = ObservationTermCfg(func=observation)
    enable_corruption = False


def actuator(kind, stiffness, limits):
    joints = [j for j in JOINTS if j.endswith("_" + kind)]
    return ImplicitActuatorCfg(
        joint_names_expr=joints, stiffness=stiffness, damping=2.,
        effort_limit_sim={j: float(limits[j]["effort"]) for j in joints},
        velocity_limit_sim={j: float(limits[j]["velocity"]) for j in joints},
    )


class FlatEnvironment(ManagerBasedRLEnv):
    @property
    def as2w_joint_ids(self):
        if not hasattr(self, "_as2w_joint_ids"):
            names = self.scene["robot"].joint_names
            if len(names) != len(JOINTS) or set(names) != set(JOINTS):
                raise ValueError(f"Unexpected AS2-W joints: {names}")
            self._as2w_joint_ids = [names.index(name) for name in JOINTS]
        return self._as2w_joint_ids


def configuration(args, physics):
    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 2.5
    cfg.sim.dt = .005
    cfg.sim.device = args.device
    cfg.sim.physics = physics
    cfg.sim.use_newton_actuators = False
    cfg.as2w_zero_memory = args.zero_memory
    cfg.decimation = 4
    cfg.episode_length_s = 10.
    cfg.seed = args.seed
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light = None
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
    limits = {j.attrib["name"]: j.find("limit").attrib for j in ET.parse(args.urdf).getroot().findall("joint")
              if j.find("limit") is not None}
    cfg.scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(usd_path=str(args.usd.resolve()), variants={"Physics": "physx"},
                             activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., .37), joint_pos={".*_hip": 0., ".*_thigh": .8, ".*_calf": -1.6, ".*_foot": 0.}),
        actuators={k: actuator(k, kp, limits) for k, kp in (("hip",55.),("thigh",55.),("calf",55.),("foot",0.))},
    )
    cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(JOINTS[:12]), preserve_order=True, scale=.35, use_default_offset=True)
    cfg.actions.wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=list(JOINTS[12:]), preserve_order=True, scale=16., use_default_offset=False)
    cfg.observations.policy = PolicyObservations()
    cmd = cfg.commands.base_velocity
    cmd.class_type = FlatCommands
    cmd.as2w_stage = args.command_stage
    cmd.heading_command = cmd.debug_vis = False
    cmd.rel_heading_envs = 0.
    cmd.ranges.heading = None
    cmd.resampling_time_range = (1.8, 3.)
    for name in vars(cfg.events):
        if name not in {"reset_base", "reset_robot_joints"}:
            setattr(cfg.events, name, None)
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params["position_range"] = (-.06, .06)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0., 0.)
    cfg.rewards.feet_air_time = None
    cfg.rewards.base_height_l2.weight = -3.
    cfg.rewards.base_height_l2.params["target_height"] = .37
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = .07
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .18
    cfg.rewards.rear_posture = RewardTermCfg(func=rear_error, weight=-.8)
    cfg.rewards.zero_motion = RewardTermCfg(func=zero_motion, weight=-.4, params={"focused": args.zero_focus})
    if args.direction_focus:
        cfg.rewards.signed_tracking = RewardTermCfg(func=signed_tracking, weight=.5)
    cfg.rewards.undesired_contacts = RewardTermCfg(func=mdp.undesired_contacts, weight=-1.,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"^(?!.*_foot$).*"), "threshold": 4.})
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
    return cfg


def train(args, physics):
    previous_contract = None
    if args.extend_from is not None:
        checkpoint = args.extend_from.resolve()
        if checkpoint.parent != args.out.resolve():
            raise ValueError("Only a checkpoint from this same run can extend training")
        previous_contract = json.loads((args.out / "contract.json").read_text())
        settings = dict(initial_checkpoint=None, seed=args.seed, num_envs=args.num_envs,
                        command_stage=args.command_stage, zero_focus=args.zero_focus,
                        zero_wheel_memory=args.zero_memory)
        if any(previous_contract.get(key) != value for key, value in settings.items()):
            raise ValueError("Extension settings must match the original randomly initialized run")
        if previous_contract.get("direction_focus", False) != args.direction_focus:
            raise ValueError("Extension requires the same directional reward and mirror settings")
    raw = FlatEnvironment(configuration(args, physics))
    try:
        env = RslRlVecEnvWrapper(raw, clip_actions=6.)
        obs = env.get_observations()
        expected_features = 65 if args.zero_memory else 60
        if obs["policy"].shape[-1] != expected_features or env.num_actions != 16:
            raise ValueError(f"Unexpected policy dimensions: {obs['policy'].shape}, {env.num_actions}")
        args.out.mkdir(parents=True, exist_ok=True)
        contract = {
            "backend": "isaacsim_physx", "initial_checkpoint": None, "seed": args.seed,
            "num_envs": args.num_envs, "steps_per_iteration": args.num_envs * 24,
            "command_stage": args.command_stage,
            "zero_focus": args.zero_focus,
            "direction_focus": args.direction_focus,
            "zero_wheel_memory": args.zero_memory, "policy_width": expected_features,
            "iteration_budget": args.iterations,
            "policy_features": ["joystick_command", "body_gyro", "gravity", "constant",
                                "joint_q", "joint_dq", "previous_action", "phase"],
            "joint_order": JOINTS, "step_dt": raw.step_dt,
        }
        runner_cfg = UnitreeGo2FlatPPORunnerCfg()
        runner_cfg.seed = args.seed
        runner_cfg.device = args.device
        runner_cfg.max_iterations = args.iterations
        runner_cfg.save_interval = min(25, args.iterations)
        runner_cfg.actor.distribution_cfg.init_std = .25
        runner_cfg.algorithm.learning_rate = 3e-4
        if args.direction_focus:
            runner_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
                use_mirror_loss=True, mirror_loss_coeff=.10, data_augmentation_func=mirror_flat)
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, check_rsl_rl_version())
        runner = create_rsl_rl_runner(env, runner_cfg, log_dir=str(args.out))
        completed = 0
        if previous_contract is not None:
            runner.load(str(checkpoint), map_location=args.device)
            completed = runner.current_learning_iteration + 1
            runner.current_learning_iteration = completed
            contract["continued_after_iterations"] = completed
        if completed >= args.iterations:
            raise ValueError("The iteration budget must exceed the completed updates")
        (args.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        runner.learn(num_learning_iterations=args.iterations - completed, init_at_random_ep_len=False)
        print("AS2W_ISAAC_PPO_COMPLETE", args.num_envs * runner_cfg.num_steps_per_env * args.iterations, flush=True)
    finally:
        raw.close()
