"""Blind AS2-W gait reference and up/down stair curriculum."""
import torch
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.terrains import (
    MeshInvertedPyramidStairsTerrainCfg, MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg, TerrainGeneratorCfg,
)
from as2w_isaac_flat_env_all import FlatEnvironment, JOINTS
from as2w_isaac_robust_env import robust_config, DelayedPosition, PredictiveDelayedPosition, TakeoverCommands
from as2w_training_safety import apply_safety
from as2w_stepping_control import calibrated_sequence, guarded_sequence_action, sequence_swing

PERIOD = .80
DUTY = .62


def gait_reference(command, clock, *, period=PERIOD, duty=DUTY, crawl=False, up=None, q=None, preswing=False):
    side = command.new_tensor((1., -1., 1., -1.))
    hip_x = command.new_tensor((.2119,.2119,-.2119,-.2119))
    hip_y = side * .06
    offset = (0., .5, .25, .75) if crawl else (0., .5, .5, 0.)
    phase = (clock[:, None] + command.new_tensor(offset)).remainder(1.)
    swing = ((phase-duty)/(1.-duty)).clamp(0.,1.)
    stride = torch.where(phase < duty, 1.-2.*phase/duty, -torch.cos(torch.pi*swing))
    vx = command[:, :1] - command[:, 2:3]*hip_y
    vy = command[:, 1:2] + command[:, 2:3]*hip_x
    active = command.abs().amax(dim=1, keepdim=True) > .015
    stride_period = min(period, .95)
    x = .5*stride_period*duty*vx*stride
    y = side*.1054 + .5*stride_period*duty*vy*stride
    lift = torch.sin(torch.pi*swing)*active
    z = -.424*torch.cos(command.new_tensor(.8)) + (.082 if crawl else .064)*lift
    if crawl:
        x += .028*(lift*hip_x.sign()).sum(dim=1,keepdim=True)
        y += .024*(lift*side).sum(dim=1,keepdim=True)
    if preswing:
        # Move the trunk toward the support triangle before raising the swing foot.
        before = ((phase-(duty-.16))/.16).clamp(0.,1.)
        after = ((.06-phase)/.06).clamp(0.,1.)
        shift = torch.maximum(before,after)*active
        x += .045*(shift*hip_x.sign()).sum(dim=1,keepdim=True)
        y += .040*(shift*side).sum(dim=1,keepdim=True)
    if up is not None:
        # Correct contact-height targets in the body frame using only the IMU up vector.
        tilt = (up[:, :1]*(hip_x+x) + up[:, 1:2]*(hip_y+y)) / up[:, 2:3].clamp_min(.70)
        z = z - tilt.clamp(-.055,.055) + .025*lift - .015*(phase < duty)*active
    if q is not None:
        hip_now, thigh_now, calf_now = q.split(4,dim=1)
        px = hip_x-.212*(thigh_now.sin()+(thigh_now+calf_now).sin())
        zp = -.212*(thigh_now.cos()+(thigh_now+calf_now).cos())
        py = hip_y+side*.1054*hip_now.cos()-zp*hip_now.sin()
        pz = side*.1054*hip_now.sin()+zp*hip_now.cos()
        estimated_h = .086-(up[:,:1]*px+up[:,1:2]*py+up[:,2:3]*pz).amin(dim=1,keepdim=True)
        extend = (.355-estimated_h).clamp(-.02,.045)
        z = z - (.022+.8*extend)*(phase < duty)*active + .023*lift
    plane = -(y.square()+z.square()-.1054**2).clamp_min(.01).sqrt()
    hip = torch.atan2(z,y)-torch.atan2(plane,side*.1054)
    calf = -torch.acos(((x.square()+plane.square())/(2.*.212**2)-1.).clamp(-.99,.99))
    thigh = torch.atan2(-x,-plane)-torch.atan2(calf.sin(),1.+calf.cos())
    return torch.cat((hip, thigh-.8, calf+1.6), dim=1)


class StairCommands(TakeoverCommands):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.goals[:, :2] *= 1.6

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self._env.cfg.as2w_directed_stairs:
            columns = self._env.scene.terrain.terrain_types[env_ids]
            moving = ~self.static[env_ids]
            select = (columns>=4) & moving
            target = self.targets[env_ids]
            direct = torch.zeros_like(target)
            direct[:,0] = .12+.08*torch.rand(len(target),device=self.device)
            self.targets[env_ids] = torch.where(select[:,None],direct,target)

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self._env.episode_length_buf * self._env.step_dt >= self._env.cfg.as2w_motion_end_s] = 0.


def delayed_proprioception(env, frame):
    if not hasattr(env, "_step_measurement"):
        env._step_measurement = frame.clone()
        env._step_previous_measurement = frame.clone()
        env._step_delivered = frame.clone()
        env._step_packet_missed = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        env._step_measurement_tick = -1
        env._step_lag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tick = env._sim_step_counter
    new_tick = tick != env._step_measurement_tick
    if new_tick:
        env._step_previous_measurement.copy_(env._step_measurement)
        noise = frame.new_tensor([.015]*3+[.006]*3+[.004]*12+[.04]*12)
        env._step_measurement.copy_(frame + torch.randn_like(frame)*noise)
        env._step_measurement_tick = tick
    reset = env.episode_length_buf == 0
    env._step_previous_measurement[reset] = env._step_measurement[reset]
    env._step_lag[reset] = torch.rand_like(env._step_lag[reset], dtype=torch.float32) < .5
    selected = torch.where(env._step_lag[:,None],env._step_previous_measurement,env._step_measurement)
    packet_loss = getattr(env.cfg,"as2w_step_packet_loss",0.)
    if not packet_loss:
        env._step_delivered.copy_(selected)
    elif new_tick:
        miss = (torch.rand(env.num_envs,device=env.device)<packet_loss) & ~env._step_packet_missed & ~reset
        env._step_delivered.copy_(torch.where(miss[:,None],env._step_delivered,selected))
        env._step_packet_missed.copy_(miss)
    env._step_delivered[reset] = selected[reset]
    env._step_packet_missed[reset] = False
    return env._step_delivered


class BlindStairEnvironment(FlatEnvironment):
    def step(self, action):
        cmd = self.command_manager.get_command("base_velocity")
        phase = self.episode_length_buf * self.step_dt / self.cfg.as2w_step_period
        robot = self.scene["robot"]
        up = -robot.data.projected_gravity_b.torch if self.cfg.as2w_support_balance else None
        joints = robot.data.joint_pos.torch[:,self.as2w_joint_ids[:12]] if self.cfg.as2w_plant else None
        prior = gait_reference(cmd, phase, period=self.cfg.as2w_step_period,
                               duty=self.cfg.as2w_step_duty, crawl=self.cfg.as2w_crawl, up=up, q=joints,
                               preswing=self.cfg.as2w_preswing)
        age = self.episode_length_buf * self.step_dt
        if self.cfg.as2w_guarded_sequence:
            sensed = self._step_delivered
            return super().step(guarded_sequence_action(
                cmd,age,action,sensed[:,12:14],self.cfg.as2w_motion_end_s,
                period=self.cfg.as2w_step_period,stride_gain=self.cfg.as2w_stride_gain,
                gravity=sensed[:,3:6] if self.cfg.as2w_sequence_attitude else None,
                gyro=sensed[:,:3] if self.cfg.as2w_sequence_attitude else None,
                body_raise_m=getattr(self.cfg, "as2w_eval_body_raise_m", 0.)))
        if self.cfg.as2w_sequence:
            prior = calibrated_sequence(cmd,(age-2.).clamp_min(0.)/12.)
        blend = ((age-1.).clamp(0.,1.) * (self.cfg.as2w_motion_end_s+2.-age).clamp(0.,1.))[:,None]
        return super().step(blend * (prior + .12 * action) / .30)


def terrain(stairs):
    flat = MeshPlaneTerrainCfg(proportion=.5 if stairs else 1.)
    if not stairs:
        return {"flat": flat}
    options = dict(proportion=.25, step_height_range=(.018,.065),
                   step_width=.34, platform_width=1.15, border_width=.3)
    return {"flat": flat, "up": MeshInvertedPyramidStairsTerrainCfg(**options),
            "down": MeshPyramidStairsTerrainCfg(**options)}


def stair_config(args, physics, observations):
    cfg = robust_config(args, physics)
    cfg.episode_length_s = 22.
    cfg.as2w_motion_end_s = 20.
    cfg.as2w_directed_stairs = args.directed_stair_curriculum
    cfg.as2w_step_packet_loss = args.packet_loss
    cfg.as2w_sequence_attitude = args.sequence_attitude
    cfg.as2w_guarded_sequence = args.curriculum.endswith("-sequence-guarded")
    cfg.as2w_sequence = cfg.as2w_guarded_sequence or args.curriculum.endswith("-sequence")
    cfg.as2w_preswing = args.curriculum.endswith("-preswing")
    cfg.as2w_plant = args.curriculum.endswith(("-plant","-preswing"))
    cfg.as2w_support_balance = args.curriculum.endswith(("-support","-plant","-preswing"))
    cfg.as2w_crawl = args.curriculum.endswith(("-crawl","-support","-slow","-stiff","-plant","-preswing"))
    cfg.as2w_step_period = 2.40 if args.curriculum.endswith(("-slow","-stiff","-plant","-preswing")) else (1.20 if cfg.as2w_crawl else PERIOD)
    if cfg.as2w_sequence:
        cfg.as2w_step_period = args.sequence_period
        cfg.as2w_stride_gain = args.sequence_stride
    cfg.as2w_step_duty = .78 if cfg.as2w_crawl else DUTY
    cfg.actions.joint_pos.scale = .30
    cfg.actions.joint_pos.class_type = PredictiveDelayedPosition if args.predictive_guard else DelayedPosition
    cfg.actions.wheel_vel = None
    if cfg.as2w_sequence or args.curriculum.endswith(("-stiff","-plant","-preswing")):
        for name in ("hip","thigh","calf"):
            cfg.scene.robot.actuators[name].stiffness = 85.
            cfg.scene.robot.actuators[name].damping = 3.2
    wheel = cfg.scene.robot.actuators["foot"]
    wheel.stiffness = 45.
    wheel.damping = 4.
    cfg.observations.policy = observations
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(JOINTS[:12]))
    cfg.events.reset_robot_joints.params["position_range"] = (-.10,.10)
    cfg.events.reset_robot_joints.params["velocity_range"] = (-.12,.12)
    cfg.events.reset_wheels = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINTS[12:])),
        "position_range": (0.,0.), "velocity_range": (0.,0.),
    })
    cfg.rewards.rear_posture = None
    cfg.rewards.base_height_l2 = None
    cfg.commands.base_velocity.class_type = StairCommands
    cfg.commands.base_velocity.resampling_time_range = (6., 10.)
    if args.stairs:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(6.,6.), border_width=4., num_rows=4, num_cols=8, curriculum=True,
            sub_terrains=terrain(True), seed=args.seed)
        cfg.scene.terrain.max_init_terrain_level = 1
        if cfg.as2w_directed_stairs:
            cfg.events.reset_base.params["pose_range"].update(x=(.14,.24),yaw=(-.05,.05))
            cfg.commands.base_velocity.resampling_time_range = (16.,20.)
    limits = apply_safety(cfg, args.urdf, floor_guard=False, surface_guard=cfg.as2w_guarded_sequence)
    if args.target_margin:
        cfg.actions.joint_pos.clip = {
            name:(lower+args.target_margin,upper-args.target_margin)
            for name,(lower,upper) in limits["angle_rad"].items()
        }
        limits["target_margin_rad"] = args.target_margin
    return cfg, limits


def gait_contacts(env):
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "_gait_contact_ids"):
        env._gait_contact_ids = [sensor.body_names.index(name) for name in JOINTS[12:]]
    contact = sensor.data.net_forces_w.torch[:, env._gait_contact_ids, 2] > 4.
    clock = env.episode_length_buf * env.step_dt / env.cfg.as2w_step_period
    offset = (0., .5, .25, .75) if env.cfg.as2w_crawl else (0., .5, .5, 0.)
    phase = (clock[:, None] + clock.new_tensor(offset)).remainder(1.)
    active = env.command_manager.get_command("base_velocity").abs().amax(dim=1) > .015
    want_contact = (phase < env.cfg.as2w_step_duty) | ~active[:,None]
    if env.cfg.as2w_sequence:
        clock = (env.episode_length_buf*env.step_dt-2.).clamp_min(0.)/env.cfg.as2w_step_period
        _, lift, _ = sequence_swing(clock)
        want_contact = (lift <= .05) | ~active[:,None]
    return (contact == want_contact).float().mean(dim=1)

def stair_course_progress(env):
    from as2w_training_safety import knee_clearance

    columns = env.scene.terrain.terrain_types
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    if not hasattr(env,"_course_hub_ids"):
        env._course_hub_ids = [robot.body_names.index(name) for name in JOINTS[12:]]
        env._course_sensor_ids = [sensor.body_names.index(name) for name in JOINTS[12:]]
    hubs = robot.data.body_pos_w.torch[:,env._course_hub_ids]
    grade = hubs[...,2]-knee_clearance(env,hubs)
    level = env.scene.env_origins[:,2,None]
    sign = torch.where(columns<6,1.,-1.)
    signed_gain = sign[:,None]*(grade-level)
    support = sensor.data.net_forces_w.torch[:,env._course_sensor_ids,2]>4.
    moving = env.command_manager.get_command("base_velocity")[:,0]>.04
    upright = -robot.data.projected_gravity_b.torch[:,2]>.90
    return (signed_gain.clamp(0.,.25)*support).mean(dim=1)/.10 * ((columns>=4)&moving&upright)
