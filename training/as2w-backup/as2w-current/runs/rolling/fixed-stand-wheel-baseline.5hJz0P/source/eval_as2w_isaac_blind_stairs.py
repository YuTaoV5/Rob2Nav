"""Native blind staircase tests with terrain clearance measured outside policy."""
import json
import torch
from as2w_knee_envelope import MIN_KNEE_SURFACE_M
from as2w_training_safety import MOTION_SUBSTEP_TERMS, knee_clearance, knee_surface_clearance
from as2w_isaac_flat_env_all import JOINTS


@torch.inference_mode()
def evaluate(env, wrapped, actor, output):
    terrain = env.scene.terrain
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    wheel = [sensor.body_names.index(j) for j in JOINTS[12:]]
    knee = [robot.body_names.index(j) for j in JOINTS[8:12]]
    nonwheel = [i for i, n in enumerate(sensor.body_names) if not n.endswith("_foot")]
    term = env.command_manager.get_term("base_velocity")
    obs, _ = wrapped.reset()
    term.time_left.fill_(100.)
    columns = terrain.terrain_types.clone()
    alive = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    touched = torch.zeros_like(alive)
    knee_min = torch.full_like(alive, float("inf"), dtype=torch.float32)
    surface_min = knee_min.clone()
    air = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.long)
    steps = torch.zeros_like(air)
    peak_wheel = torch.zeros_like(air,dtype=torch.float32)
    wheel_bodies = [robot.body_names.index(j) for j in JOINTS[12:]]
    failures = {}
    termination = env.termination_manager
    last_position = robot.data.root_pos_w.torch.clone()
    top = None
    bottom = None
    origin = None
    initial_surface = None
    end_surface = None
    sequence = getattr(env.cfg,"as2w_sequence",False)
    end_tick = round((env.cfg.as2w_motion_end_s if sequence else 19.)/env.step_dt)
    horizon = end_tick + round((4. if sequence else 2.5)/env.step_dt)
    release_frames = torch.zeros_like(air[:,0])
    entry_frames = release_frames.clone()
    start_ok = torch.zeros_like(alive)
    leg_ids = env.as2w_joint_ids[:12]
    for tick in range(horizon):
        term.targets[:, :] = 0.
        if 100 <= tick < end_tick:
            term.targets[:, 0] = .16
        obs, _, done, _ = wrapped.step(actor(obs))
        failed = alive & done.bool()
        for index in failed.nonzero(as_tuple=True)[0].tolist():
            terms = [name for name in termination.active_terms
                     if termination.get_term(name)[index].item()]
            substep = []
            if hasattr(env, "_motion_substep_last"):
                substep = [name for name, bad in zip(MOTION_SUBSTEP_TERMS,
                           env._motion_substep_last[index].tolist()) if bad]
            failures[index] = dict(time_s=(tick+1)*env.step_dt,terms=terms,physics_checks=substep)
            if "nonwheel_contact" in substep and hasattr(env, "_motion_first_contact_body"):
                body_index = int(env._motion_first_contact_body[index])
                if body_index >= 0:
                    failures[index]["first_contact_body"] = sensor.body_names[body_index]
            touched[index] |= "as2w_nonwheel_contact" in terms or "nonwheel_contact" in substep
        alive &= ~done.bool()
        force = sensor.data.net_forces_w.torch
        touched |= alive & (force[:, nonwheel].norm(dim=-1) > 4.).any(dim=1)
        pos = robot.data.root_pos_w.torch
        last_position = torch.where(alive[:,None],pos,last_position)
        contact = force[:,wheel,2] > 4.
        joint_error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,leg_ids]
        quiet = ((robot.data.root_lin_vel_b.torch.norm(dim=1)<.06)
                 & (robot.data.root_ang_vel_b.torch.norm(dim=1)<.12))
        fixed = quiet & contact.all(dim=1) & (joint_error.abs().amax(dim=1)<.18)
        fixed &= alive & (-robot.data.projected_gravity_b.torch[:,2] > .95)
        if tick < 100:
            entry_frames = torch.where(fixed,entry_frames+1,0)
        elif tick >= end_tick:
            release_frames = torch.where(fixed,release_frames+1,0)
        if tick == 100:
            origin = last_position.clone()
            start_ok = alive & (entry_frames*env.step_dt >= .40)
        if origin is not None and top is not None and not bool(alive.any()):
            break
        if tick >= 100:
            clearance = knee_clearance(env, robot.data.body_pos_w.torch[:, knee])
            knee_min = torch.minimum(knee_min,torch.where(alive,clearance.amin(dim=1),float("inf")))
            surface_min = torch.minimum(surface_min,torch.where(
                alive,knee_surface_clearance(env).amin(dim=1),float("inf")))
            surface = pos[:,2] - knee_clearance(env, pos[:,None,:]).squeeze(1)
            if top is None:
                top, bottom = surface.clone(), surface.clone()
                initial_surface = surface.clone()
                end_surface = surface.clone()
            if tick < end_tick:
                end_surface = torch.where(alive,surface,end_surface)
                top = torch.where(alive,torch.maximum(top,surface),top)
                bottom = torch.where(alive,torch.minimum(bottom,surface),bottom)
            if tick < end_tick:
                hub_clearance = knee_clearance(env,robot.data.body_pos_w.torch[:,wheel_bodies])-.086
                steps += contact & (air>=3) & (peak_wheel>=.02) & alive[:,None]
                air = torch.where(contact | ~alive[:,None],0,air+1)
                peak_wheel = torch.where(contact | ~alive[:,None],0.,torch.maximum(peak_wheel,hub_clearance))
    xy = last_position[:, :2] - origin[:,:2]
    final_v = robot.data.root_lin_vel_b.torch.norm(dim=1)
    final_up = -robot.data.projected_gravity_b.torch[:,2]
    rise = top-bottom
    stable = alive & ~touched & (knee_min >= env.cfg.as2w_min_knee_clearance_m)
    stable &= surface_min >= MIN_KNEE_SURFACE_M
    finish = alive & (final_v < .08) & (final_up > .95)
    if sequence:
        finish &= release_frames*env.step_dt >= 2.
        stable &= start_ok
    passes = stable & finish & (xy[:,0] >= 1.0) & (rise >= .045) & (steps >= 2).all(dim=1)
    report = {}
    for name, selection in (("up", (columns>=4)&(columns<6)), ("down", columns>=6)):
        ids = torch.where(selection)[0]
        rows = []
        for i in ids.tolist():
            elevation = float(end_surface[i] - initial_surface[i])
            direction_ok = elevation >= .045 if name == "up" else elevation <= -.045
            rows.append(dict(passed=bool(passes[i]) and direction_ok, alive=bool(alive[i]),
                             signed_elevation_m=elevation,
                             nonwheel=bool(touched[i]), fixed_stand_exit=bool(finish[i]),
                             fixed_stand_entry=bool(start_ok[i]),
                             travel_m=float(xy[i].norm()), forward_progress_m=float(xy[i,0]), terrain_span_m=float(rise[i]),
                             min_knee_clearance_m=float(knee_min[i]) if torch.isfinite(knee_min[i]) else None,
                             min_knee_surface_m=float(surface_min[i]) if torch.isfinite(surface_min[i]) else None,
                             steps=steps[i].tolist(), first_failure=failures.get(i)))
        report[name] = dict(total=len(rows), passed=sum(r["passed"] for r in rows), rows=rows)
    (output/"stairs-evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    print("AS2W_BLIND_STAIRS", {k:(v["passed"],v["total"]) for k,v in report.items()}, flush=True)