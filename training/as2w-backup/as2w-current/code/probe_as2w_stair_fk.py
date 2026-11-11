"""Compare reference foot displacement with native articulated forward kinematics."""
import json
import torch
from isaaclab.utils.math import quat_apply_inverse
from as2w_blind_stair_gait import gait_reference
from as2w_isaac_flat_env_all import JOINTS


def evaluate(env, out):
    robot = env.scene["robot"]
    feet = [robot.body_names.index(name) for name in JOINTS[12:]]
    legs = env.as2w_joint_ids[:12]
    default = robot.data.default_joint_pos.torch.clone()
    phase = torch.arange(env.num_envs, device=env.device)/env.num_envs

    def actual(offset):
        q = default.clone()
        q[:, legs] += offset
        robot.write_joint_position_to_sim_index(position=q, env_ids=slice(None))
        env.sim.forward()
        env.scene.update(env.physics_dt)
        root = robot.data.root_pos_w.torch
        quat = robot.data.root_quat_w.torch
        relative = robot.data.body_pos_w.torch[:, feet] - root[:,None]
        return quat_apply_inverse(quat[:,None].expand(-1, len(feet), -1),
                                  relative).clone()

    zero = torch.zeros((env.num_envs,3), device=env.device)
    neutral = actual(torch.zeros_like(default[:,legs]))
    cases = {}
    for name, cmd in (("forward", (.16,0.,0.)), ("left", (0.,.16,0.)), ("yaw", (0.,0.,.30))):
        targets = zero + zero.new_tensor(cmd)
        offsets = gait_reference(targets, phase)
        measured = actual(offsets)-neutral
        sides = offsets.new_tensor((1.,-1.,1.,-1.))
        qh,qt,qc = (offsets+offsets.new_tensor([0.]*4+[.8]*4+[-1.6]*4)).split(4,dim=1)
        x = -.212*(qt.sin()+(qt+qc).sin())
        planar_z = -.212*(qt.cos()+(qt+qc).cos())
        y = sides*.1054*qh.cos()-planar_z*qh.sin()
        z = sides*.1054*qh.sin()+planar_z*qh.cos()
        expected = torch.stack((x,y-sides*.1054,z+.424*torch.cos(z.new_tensor(.8))),dim=2)
        gain = (expected*measured).sum(dim=0)/expected.square().sum(dim=0).clamp_min(1e-8)
        cases[name] = dict(gain=gain.tolist(),expected_range=(expected.amax(dim=0)-expected.amin(dim=0)).tolist(),
                           measured_range=(measured.amax(dim=0)-measured.amin(dim=0)).tolist())
    result = dict(nominal=neutral.mean(dim=0).tolist(),cases=cases)
    (out/"fk-diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print("AS2W_FK", json.dumps(result),flush=True)


@torch.inference_mode()
def dynamics(env, wrapped, out):
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    feet = [robot.body_names.index(name) for name in JOINTS[12:]]
    wheels = [sensor.body_names.index(name) for name in JOINTS[12:]]
    legs = env.as2w_joint_ids[:12]
    term = env.command_manager.get_term("base_velocity")
    wrapped.reset()
    term.time_left.fill_(100.)
    action = torch.zeros((env.num_envs,12), device=env.device)
    offset = action.new_tensor((0.,.5,.25,.75) if env.cfg.as2w_crawl else (0.,.5,.5,0.))
    swing = torch.zeros((env.num_envs,4), device=env.device)
    swing_contact = torch.zeros_like(swing)
    stance = torch.zeros_like(swing)
    stance_contact = torch.zeros_like(swing)
    peak = torch.zeros_like(swing)
    tracking = torch.zeros((env.num_envs,12),device=env.device)
    tracked = 0
    vx = []
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    first_failure = {}
    termination = env.termination_manager
    if env.cfg.as2w_sequence:
        from as2w_blind_stair_gait import sequence_swing
    for tick in range(950 if env.cfg.as2w_sequence else 350):
        term.targets[:,0] = .16 if tick >= 100 else 0.
        age = env.episode_length_buf * env.step_dt
        planned = (age[:,None]/env.cfg.as2w_step_period+offset).remainder(1.) >= env.cfg.as2w_step_duty
        if env.cfg.as2w_sequence:
            _, lift, _ = sequence_swing((age-2.).clamp_min(0.)/env.cfg.as2w_step_period)
            planned = lift>.05
        _,_,done,_ = wrapped.step(action)
        failed = alive & done.bool()
        for index in failed.nonzero(as_tuple=True)[0].tolist():
            first_failure[index] = {
                "time_s": round(tick * env.step_dt, 3),
                "episode_age_s": round(float(age[index]), 3),
                "swing": planned[index].nonzero(as_tuple=True)[0].tolist(),
                "terms": [name for name in termination.active_terms
                          if termination.get_term(name)[index].item()],
            }
            if hasattr(env, "_guard_stance_detail"):
                detail = env._guard_stance_detail[index].tolist()
                first_failure[index]["stance_flags"] = dict(zip(("crossed", "rear_thigh", "knee"), detail))
                first_failure[index]["stance_metrics"] = env._guard_stance_metrics[index].tolist()
        alive &= ~done.bool()
        if tick < 150:
            continue
        q = robot.data.joint_pos.torch[:,legs]
        target = robot.data.joint_pos_target.torch[:,legs]
        tracking += (q-target).abs()
        tracked += 1
        contact = sensor.data.net_forces_w.torch[:,wheels,2] > 4.
        swing += planned
        swing_contact += planned & contact
        stance += ~planned
        stance_contact += ~planned & contact
        center = robot.data.body_pos_w.torch[:,feet,2] - env.scene.env_origins[:,None,2]
        peak = torch.maximum(peak, center)
        vx.append(robot.data.root_lin_vel_b.torch[:,0].clone())
    result = dict(alive=int(alive.sum()), total=env.num_envs, first_failure=first_failure,
                  swing_contact_fraction=(swing_contact.sum(0)/swing.sum(0)).tolist(),
                  stance_contact_fraction=(stance_contact.sum(0)/stance.sum(0)).tolist(),
                  max_wheel_center_m=peak.mean(0).tolist(),
                  vx_m_s=torch.stack(vx).mean(dim=0).tolist(),
                  mean_joint_target_error_rad=(tracking.mean(0)/tracked).tolist())
    (out/"gait-dynamics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("AS2W_GAIT_DYNAMICS",json.dumps(result),flush=True)


@torch.inference_mode()
def single_lift(env, wrapped, out, *, sweep=False):
    from as2w_isaac_flat_env_all import FlatEnvironment
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    wheels = [sensor.body_names.index(j) for j in JOINTS[12:]]
    feet = [robot.body_names.index(j) for j in JOINTS[12:]]
    legs = env.as2w_joint_ids[:12]
    wrapped.reset()
    ids = torch.arange(env.num_envs, device=env.device)
    leg = ids % 4
    choices = 27 if sweep else 4
    variant = ids // 4 % choices
    group = ids // (4*choices)
    side = torch.tensor([1.,-1.,1.,-1.],device=env.device)
    fore = torch.tensor([1.,1.,-1.,-1.],device=env.device)
    zero = torch.zeros((env.num_envs,4),device=env.device)
    lift_mask = (torch.arange(4,device=env.device)[None,:] == leg[:,None])
    alive = torch.ones(env.num_envs,device=env.device,dtype=torch.bool)
    free = torch.zeros_like(ids)
    run = torch.zeros_like(ids)
    max_run = torch.zeros_like(ids)
    peak = torch.full_like(ids,-10,dtype=torch.float32)
    tracking = torch.zeros_like(peak)
    tracked = torch.zeros_like(peak)
    if sweep:
        shift_x = zero.new_tensor((.04,.08,.12))[variant//9]
        shift_y = zero.new_tensor((.03,.06,.09))[(variant//3)%3]
        extension = zero.new_tensor((.018,.030,.042))[variant%3]
    else:
        shift_x = shift_y = zero.new_tensor((0.,.04,.07,.07))[variant]
        extension = .025*(variant==3)
    for tick in range(270):
        time = tick * env.step_dt
        shift = min(1.,max(0.,(time-1.3)/1.2))
        swing = min(1.,max(0.,(time-2.5)/1.2),max(0.,(5.2-time)/.8))
        x = zero + shift*shift_x[:,None]*fore[leg,None]
        y = side*.1054 + zero + shift*shift_y[:,None]*side[leg,None]
        z = zero - .424*torch.cos(zero.new_tensor(.8))
        z += .105*swing*lift_mask - shift*extension[:,None]*(~lift_mask)
        planar = -(y.square()+z.square()-.1054**2).clamp_min(.01).sqrt()
        h = torch.atan2(z,y)-torch.atan2(planar,side*.1054)
        c = -torch.acos(((x.square()+planar.square())/(2.*.212**2)-1.).clamp(-.99,.99))
        t = torch.atan2(-x,-planar)-torch.atan2(c.sin(),1.+c.cos())
        target = torch.cat((h,t-.8,c+1.6),dim=1)
        _,_,terminated,truncated,_ = FlatEnvironment.step(env,target/.30)
        alive &= ~(terminated|truncated)
        if not 3.65 <= time <= 4.45:
            continue
        height = robot.data.body_pos_w.torch[:,feet,2] - env.scene.env_origins[:,None,2]
        load = sensor.data.net_forces_w.torch[:,wheels,2]
        airborne = alive & (load[ids,leg] < 2.) & (height[ids,leg]>.106)
        free += airborne
        run = torch.where(airborne,run+1,0)
        max_run = torch.maximum(max_run,run)
        peak = torch.maximum(peak,torch.where(alive,height[ids,leg],-10.))
        err = (robot.data.joint_pos.torch[:,legs]-robot.data.joint_pos_target.torch[:,legs]).abs()
        tracking += torch.where(alive,err[ids,8+leg],0.)
        tracked += alive
    rows = [dict(leg=JOINTS[12+int(leg[i])],variant=int(variant[i]),rep=int(group[i]),
                 shift_x_m=float(shift_x[i]),shift_y_m=float(shift_y[i]),
                 extension_m=float(extension[i]),alive=bool(alive[i]),peak_wheel_center_m=float(peak[i]),
                 free_s=float(max_run[i])*env.step_dt,
                 calf_tracking_rad=float(tracking[i]/tracked[i].clamp_min(1.)))
            for i in range(env.num_envs)]
    (out/"single-lift.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("AS2W_SINGLE_LIFT",int((max_run>=10).sum()),"/",len(rows),flush=True)
