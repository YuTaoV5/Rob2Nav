"""Gate each speed stage on translation in eight directions without turning."""
import json
import torch

from as2w_rolling_brake_limits import rolling_brake_limits
from as2w_training_safety import MOTION_SUBSTEP_TERMS

NAMES = ("forward", "forward-left", "left", "back-left",
         "back", "back-right", "right", "forward-right")


def angle_error(x, reference):
    return ((x - reference + torch.pi) % (2.*torch.pi) - torch.pi).abs()


@torch.inference_mode()
def evaluate(env, wrapped, actor, args):
    speed = args.speed_cap
    names = ("forward", "back") if args.speed_rolling else NAMES
    if env.num_envs % len(names):
        raise ValueError("Use a multiple of eight environments")
    trials = env.num_envs // len(names)
    angle = torch.arange(len(names), device=env.device) * (2.*torch.pi / len(names))
    direction = torch.stack((angle.cos(), angle.sin()), dim=1).repeat_interleave(trials, dim=0)
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    other = [i for i, n in enumerate(sensor.body_names) if not n.endswith("_foot")]
    wheel_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    wheels = [sensor.body_names.index(name) for name in wheel_names]
    command = env.command_manager.get_term("base_velocity")
    obs, _ = wrapped.reset()
    command.time_left.fill_(100.)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    touches = torch.zeros_like(alive, dtype=torch.long)
    steady = torch.zeros_like(touches)
    unloaded = torch.zeros((env.num_envs, 4), dtype=torch.long, device=env.device)
    max_unloaded = unloaded.clone()
    first_unload_tick = torch.full_like(unloaded, -1)
    stop = torch.full_like(touches, -1)
    termination_tick = torch.full_like(touches, -1)
    min_knee = torch.full((env.num_envs,), float("inf"), device=env.device)
    min_surface = min_knee.clone()
    effort_ratio = torch.zeros_like(min_knee)
    entry_frames = torch.zeros_like(touches)
    exit_frames = torch.zeros_like(touches)
    leg_ids = env.as2w_joint_ids[:12]
    safety = getattr(env.cfg, "as2w_safety", None)
    if safety:
        from as2w_training_safety import knee_clearance, knee_surface_clearance
        from as2w_knee_envelope import MIN_KNEE_SURFACE_M
        knee_ids = [robot.body_names.index(name) for name in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
        effort = min_knee.new_tensor([safety["effort_nm"][name] for name in robot.joint_names])
    samples = []
    park = []
    initial_heading = None
    origin = None
    brake_distance = torch.zeros(env.num_envs, device=env.device)
    start = 200 if args.rolling_wheel_hold else 100
    release = start+300
    first_lock = torch.full_like(touches, -1)
    lock_xy = torch.zeros((env.num_envs, 2), device=env.device)
    release_xy = lock_xy.clone()
    latch_time = .35 if args.park_settle_joints else .12
    exit_checks = ("wheel_support", "upright", "joint_pose", "linear_speed", "angular_speed")
    exit_window_frames = round(2. / env.step_dt)
    exit_failures = torch.zeros((env.num_envs, len(exit_checks)), dtype=torch.long, device=env.device)
    first_failures = {}
    termination = env.termination_manager
    for tick in range(release+750):
        time = tick * env.step_dt
        amount = max(0., min(speed, 1.2*(time-start*env.step_dt),
                            speed-1.2*max(0.,time-release*env.step_dt)))
        command.targets[:, :2] = amount * direction
        command.targets[:, 2] = 0.
        if args.rolling_wheel_hold and tick >= release:
            pre_step_xy = robot.data.root_pos_w.torch[:, :2].clone()
            if tick == release:
                release_xy.copy_(pre_step_xy)
        obs, _, done, _ = wrapped.step(actor(obs))
        termination_tick = torch.where(alive & done.bool(), tick, termination_tick)
        for index in (alive & done.bool()).nonzero(as_tuple=True)[0].tolist():
            terms = [name for name in termination.active_terms if termination.get_term(name)[index].item()]
            checks = []
            if hasattr(env, "_motion_substep_last"):
                checks = [name for name, failed in zip(MOTION_SUBSTEP_TERMS,
                          env._motion_substep_last[index].tolist()) if failed]
            first_failures[index] = {"time_s": (tick + 1) * env.step_dt,
                                     "terms": terms, "physics_checks": checks}
            if "nonwheel_contact" in checks and hasattr(env, "_motion_first_contact_body"):
                body_index = int(env._motion_first_contact_body[index])
                if body_index >= 0:
                    first_failures[index]["first_contact_body"] = sensor.body_names[body_index]
        alive &= ~done.bool()
        if args.rolling_wheel_hold and tick >= release:
            captured = alive & (first_lock < 0) & (env.rolling_wheel_hold_frames * env.step_dt >= latch_time)
            first_lock = torch.where(captured, tick - release, first_lock)
            lock_xy = torch.where(captured[:, None], pre_step_xy, lock_xy)
        touches += (sensor.data.net_forces_w.torch[:, other].norm(dim=-1) > 4.).any(dim=1)
        load = sensor.data.net_forces_w.torch[:, wheels, 2]
        if tick >= 100:
            unloaded = torch.where(load < 2., unloaded+1, 0)
            max_unloaded = torch.maximum(max_unloaded, unloaded)
            first_unload_tick = torch.where(
                (first_unload_tick < 0) & (unloaded * env.step_dt > .06),
                tick, first_unload_tick)
        q = robot.data.root_quat_w.torch
        x, y, z, w = q.unbind(dim=1)
        heading = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
        v = robot.data.root_lin_vel_b.torch[:, :2]
        yaw = robot.data.root_ang_vel_b.torch[:, 2]
        pos = robot.data.root_pos_w.torch[:, :2]
        q_error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,leg_ids]
        fixed_checks = torch.stack((
            (load > 2.).all(dim=1),
            -robot.data.projected_gravity_b.torch[:, 2] > .95,
            q_error.abs().amax(dim=1) < .18,
            robot.data.root_lin_vel_b.torch.norm(dim=1) < .06,
            robot.data.root_ang_vel_b.torch.norm(dim=1) < .12,
        ), dim=1)
        fixed = fixed_checks.all(dim=1) & alive
        if tick >= release + 750 - exit_window_frames:
            exit_failures += ~fixed_checks & alive[:, None]
        if tick < start:
            entry_frames = torch.where(fixed,entry_frames+1,0)
        elif tick >= release:
            exit_frames = torch.where(fixed,exit_frames+1,0)
        if safety:
            ratio = robot.data.applied_torque.torch.abs() / effort
            effort_ratio = torch.maximum(effort_ratio, ratio.amax(dim=1))
            if tick >= 100:
                clearance = knee_clearance(env, robot.data.body_pos_w.torch[:,knee_ids])
                min_knee = torch.minimum(min_knee, clearance.amin(dim=1))
                min_surface = torch.minimum(min_surface,knee_surface_clearance(env).amin(dim=1))
        if tick == start:
            initial_heading = heading.clone()
        if release-100 <= tick < release:
            signed = (v * direction).sum(dim=1)
            cross = (v[:, 0] * direction[:, 1] - v[:, 1] * direction[:, 0]).abs()
            heading_drift = torch.rad2deg(angle_error(heading, initial_heading))
            samples.append(torch.stack((signed, cross, heading_drift, yaw.abs()), dim=1))
        if tick >= release:
            if origin is None:
                origin = pos.clone()
            distance = (pos-origin).norm(dim=1)
            brake_distance = torch.maximum(brake_distance, distance * ((stop < 0) & alive))
            norm = v.norm(dim=1)
            still = (norm <= .025) & (yaw.abs() <= .06)
            steady = torch.where(still, steady + 1, 0)
            stop = torch.where((stop < 0) & (steady >= 25) & alive, tick - release, stop)
            park.append(torch.cat((pos, heading[:, None], norm[:, None]), dim=1))
    samples = torch.stack(samples)
    signed, cross, _, yaw_rate = samples.mean(dim=0).unbind(dim=1)
    heading = samples[:, :, 2].amax(dim=0)
    parked = torch.stack(park)
    groups = {}
    brake_time_limit = speed/1.2 + 1.0
    brake_distance_limit = speed**2/(2.*1.2) + .25*speed + .05
    min_speed = 2.95 if speed >= 3. else .80 * speed
    if args.speed_rolling:
        brake_time_limit, brake_distance_limit = rolling_brake_limits(speed)
    for group, name in enumerate(names):
        rows = []
        for i in range(group*trials, (group+1)*trials):
            t = int(stop[i])
            displacement = drift = residual = None
            end_tick = int(termination_tick[i])
            if 0 <= t < len(parked) - 500 and (end_tick < 0 or release + t + 500 < end_tick):
                v = parked[t:t+501, i]
                displacement = float((v[:, :2] - v[0, :2]).norm(dim=1).max())
                drift = float(torch.rad2deg(angle_error(v[:, 2], v[0, 2]).max()))
                residual = float(v[:, 3].quantile(.95))
            unload_time = float(max_unloaded[i].max()) * env.step_dt
            unload_events = {}
            for wheel_name, tick in zip(wheel_names, first_unload_tick[i].tolist()):
                if tick < 0:
                    continue
                phase = "entry"
                if tick >= release:
                    phase = "braking_or_hold"
                elif tick >= start:
                    phase = "accelerating" if (tick-start)*env.step_dt < speed/1.2 else "cruise"
                unload_events[wheel_name] = dict(time_s=tick*env.step_dt, phase=phase)

            stable = bool(alive[i]) and not int(touches[i])
            if args.speed_rolling:
                stable = stable and unload_time <= .06
            if safety:
                stable = stable and bool(min_knee[i] >= env.cfg.as2w_min_knee_clearance_m
                                         and min_surface[i] >= MIN_KNEE_SURFACE_M
                                         and effort_ratio[i] <= 1.00001)
            track = bool(stable and min_speed <= signed[i] <= 1.20*speed
                         and cross[i] <= .2*speed+.01 and heading[i] <= 10.
                         and yaw_rate[i] <= .15)
            brake_s = (t+1)*env.step_dt if t >= 0 else None
            brake = bool(stable and brake_s is not None and brake_s <= brake_time_limit
                         and brake_distance[i] <= brake_distance_limit)
            hd, hy, hr = (.05, 5., .035) if speed >= 1. else (.02, 3., .02)
            hold = bool(brake and displacement is not None and displacement <= hd
                        and drift <= hy and residual <= hr)
            fixed_entry_s = int(entry_frames[i])*env.step_dt
            fixed_exit_s = int(exit_frames[i])*env.step_dt
            if args.rolling_fixed_exit:
                hold &= fixed_entry_s>=.40 and fixed_exit_s>=2.
            lock_tick = int(first_lock[i])
            park_lock = None
            if lock_tick >= 0 and bool(alive[i]):
                park_lock = dict(
                    first_lock_s=lock_tick * env.step_dt,
                    release_to_lock_m=float((lock_xy[i] - release_xy[i]).norm()),
                    post_lock_peak_m=float((parked[lock_tick:, i, :2] - lock_xy[i]).norm(dim=1).max()),
                )
            rows.append(dict(track=track, stable=stable, brake=brake, hold=hold, park_lock=park_lock,
                             termination_s=(end_tick + 1) * env.step_dt if end_tick >= 0 else None,
                             first_failure=first_failures.get(i),
                             fixed_entry_s=fixed_entry_s,fixed_exit_s=fixed_exit_s,
                             fixed_exit_window_frames=exit_window_frames,
                             fixed_exit_failed_frames=dict(zip(exit_checks, exit_failures[i].tolist())),
                             along_m_s=float(signed[i]), cross_m_s=float(cross[i]),
                             heading_deg=float(heading[i]), brake_s=brake_s,
                             wheel_unload_s=unload_time,
                             first_wheel_unload_over_limit=unload_events,
                             knee_clearance_m=float(min_knee[i]) if safety else None,
                             knee_surface_clearance_m=float(min_surface[i]) if safety else None,
                             simulated_effort_ratio=float(effort_ratio[i]) if safety else None,
                             brake_distance_m=float(brake_distance[i]),
                             brake_time_limit_s=brake_time_limit,
                             brake_distance_limit_m=brake_distance_limit, hold_displacement_m=displacement,
                             hold_heading_deg=drift, hold_residual_p95_m_s=residual))
        groups[name] = rows
    summary = {name: {
        "track": sum(r["track"] for r in rows), "stable": sum(r["stable"] for r in rows),
        "track_and_brake": sum(r["track"] and r["brake"] for r in rows),
        "hold": sum(r["hold"] for r in rows),
        "track_brake_hold": sum(r["track"] and r["hold"] for r in rows), "total": trials,
        "mean_along_m_s": sum(r["along_m_s"] for r in rows)/trials,
    } for name, rows in groups.items()}
    gate = all(s["track_brake_hold"] >= .8 * trials for s in summary.values())
    safety_gate = all(s["stable"] == trials for s in summary.values())
    turns = {}
    if args.speed_rolling:
        from eval_as2w_isaac_rolling_turn import evaluate_turns
        turns = evaluate_turns(env, wrapped, actor, args.seed, fixed_stand=args.rolling_fixed_exit,yaw_slew=args.turn_yaw_slew)
        gate = gate and all(row["passed"] >= .8 * row["total"] and row["wrong_direction"] == 0
                            for row in turns.values())
        safety_gate &= all(row["stable"] == row["total"] for row in turns.values())
    if args.rolling_fixed_exit:
        gate &= safety_gate
    (args.out / "speed-evaluation.json").write_text(json.dumps(
        {"cap_m_s": speed, "gate": gate, "safety_gate": safety_gate, "summary": summary, "turns": turns, "rows": groups}, indent=2) + "\n")
    print("AS2W_SPEED_EVAL", speed, gate, json.dumps(summary), flush=True)
    if turns:
        print("AS2W_TURN_EVAL", json.dumps(turns), flush=True)