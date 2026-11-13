"""Score joystick transitions and mixed commands on the same test traces."""
import json
import torch

CASES = (
    (.07, 0., .20), (.07, 0., -.20), (-.07, 0., .20), (-.07, 0., -.20),
    (0., .07, .20), (0., .07, -.20), (0., -.07, .20), (0., -.07, -.20),
)


@torch.inference_mode()
def evaluate(env, wrapped, actor, args):
    if env.num_envs % len(CASES):
        raise ValueError("Use a multiple of eight environments")
    targets = torch.tensor(CASES, device=env.device).repeat_interleave(
        env.num_envs // len(CASES), dim=0)
    robot = env.scene["robot"]
    legs = env.as2w_joint_ids[:12]
    wheels = env.as2w_joint_ids[12:]
    sensor = env.scene["contact_forces"]
    other = [i for i, n in enumerate(sensor.body_names) if not n.endswith("_foot")]
    command = env.command_manager.get_term("base_velocity")
    obs, _ = wrapped.reset()
    command.time_left.fill_(100.)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    contact = torch.zeros_like(alive, dtype=torch.long)
    steady = torch.zeros_like(contact)
    stopped = torch.full_like(contact, -1)
    velocity = []
    output = []
    parking = []
    phases = ((200, (.02, 0., 0.)), (300, (.10, 0., 0.)), (400, (-.10, 0., 0.)))
    phase_goals = [(end, targets.new_tensor(goal)) for end, goal in phases]
    zero = targets.new_zeros(3)
    for tick in range(1250):
        goal = next((goal for end, goal in phase_goals if tick < end),
                    targets if tick < 550 else zero)
        command.targets[:] = goal
        obs, _, done, _ = wrapped.step(actor(obs))
        alive &= ~done.bool()
        contact += (sensor.data.net_forces_w.torch[:, other].norm(dim=-1) > 4.).any(dim=1)
        xy = robot.data.root_lin_vel_b.torch[:, :2]
        yaw = robot.data.root_ang_vel_b.torch[:, 2]
        velocity.append(torch.cat((xy, yaw[:, None]), dim=1))
        if 300 <= tick < 325:
            q = robot.data.joint_pos_target.torch[:, legs]
            dq = robot.data.joint_vel_target.torch[:, wheels]
            output.append(torch.cat((q, dq), dim=1))
        if tick >= 550:
            x, y, z, w = robot.data.root_quat_w.torch.unbind(dim=1)
            heading = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
            speed = xy.norm(dim=1)
            below = (speed <= .025) & (yaw.abs() <= .06)
            steady = torch.where(below, steady + 1, 0)
            stopped = torch.where((stopped < 0) & (steady >= 25), tick - 550, stopped)
            pos = robot.data.root_pos_w.torch[:, :2]
            parking.append(torch.cat((pos, heading[:, None], speed[:, None]), dim=1))
    velocity = torch.stack(velocity)
    rate = torch.stack(output).diff(dim=0)
    axes = rate.new_tensor([.02]*12 + [.75]*4)
    reversals = ((rate[1:] * rate[:-1] < 0.) & (rate[1:].abs() > axes)
                 & (rate[:-1].abs() > axes)).sum(dim=(0,2))
    straight = velocity[250:300]
    fine = velocity[150:200, :, 0].mean(dim=0)
    reverse = velocity[300:400, :, 0] < -.01
    mixed = velocity[475:550].mean(dim=0)
    cross = straight[:, :, 1].abs().mean(dim=0)
    yaw_projection = mixed[:, 2] * targets[:, 2].sign()
    combined_xy_error = (mixed[:, :2] - targets[:, :2]).norm(dim=1)
    combined_yaw_error = (mixed[:, 2] - targets[:, 2]).abs()
    parking = torch.stack(parking)
    rows = []
    for i in range(env.num_envs):
        stop = int(stopped[i])
        drift = angle = residual = None
        if 0 <= stop < len(parking) - 500:
            parked = parking[stop:stop+501, i]
            drift = float((parked[:, :2] - parked[0, :2]).norm(dim=1).max())
            dh = ((parked[:, 2] - parked[0, 2] + torch.pi) % (2.*torch.pi) - torch.pi).abs()
            angle = float(torch.rad2deg(dh.max()))
            residual = float(parked[:, 3].quantile(.95))
        stable = bool(alive[i]) and int(contact[i]) == 0
        held = bool(stable and drift is not None and drift <= .02 and angle <= 3. and residual <= .02)
        rows.append({
            "goal": targets[i].tolist(), "stable": stable, "alive": bool(alive[i]),
            "nonwheel_frames": int(contact[i]), "fine_forward_m_s": float(fine[i]),
            "straight_lateral_abs_m_s": float(cross[i]),
            "reverse_s": int(reverse[:, i].float().argmax()) * env.step_dt if reverse[:, i].any() else None,
            "reverse_output_oscillations": int(reversals[i]),
            "combined_actual": mixed[i].tolist(), "yaw_wrong_direction": bool(yaw_projection[i] < 0.),
            "yaw_useful": bool(yaw_projection[i] >= .05),
            "stop_s": (stop+1)*env.step_dt if stop >= 0 else None, "held": held,
            "hold_drift_m": drift, "hold_heading_deg": angle, "hold_residual_p95_m_s": residual,
        })
    summary = {
        "profile": args.profile, "checkpoint": str(args.checkpoint), "seed": args.seed,
        "stable": sum(r["stable"] for r in rows), "total": len(rows),
        "yaw_wrong_direction": sum(r["yaw_wrong_direction"] for r in rows),
        "yaw_useful": sum(r["yaw_useful"] for r in rows),
        "held": sum(r["held"] for r in rows),
        "combined_xy_error_m_s": float(combined_xy_error.mean()),
        "combined_yaw_error_rad_s": float(combined_yaw_error.mean()),
        "straight_lateral_abs_mean_m_s": float(cross.mean()),
        "reverse_oscillations_p95": float(reversals.float().quantile(.95)),
    }
    (args.out / "teleop-evaluation.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2) + "\n")
    print("AS2W_TELEOP_EVAL", json.dumps(summary), flush=True)