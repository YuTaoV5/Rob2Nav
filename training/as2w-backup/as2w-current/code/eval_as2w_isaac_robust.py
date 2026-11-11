"""Compare frozen R284 policies through full takeover, travel and release."""
import json
import torch
from as2w_isaac_flat_env_all import GOALS


@torch.inference_mode()
def evaluate(env, wrapped, actor, args):
    if env.num_envs % len(GOALS):
        raise ValueError("Use a multiple of ten environments")
    robot = env.scene["robot"]
    legs = env.as2w_joint_ids[:12]
    wheel = env.as2w_joint_ids[12:]
    contact = env.scene["contact_forces"]
    other = [i for i, name in enumerate(contact.body_names) if not name.endswith("_foot")]
    command = env.command_manager.get_term("base_velocity")
    goals = torch.tensor(GOALS, device=env.device)
    targets = goals.repeat_interleave(env.num_envs // len(GOALS), dim=0)
    obs, _ = wrapped.reset()
    command.time_left.fill_(100.)
    shape = (env.num_envs,)
    alive = torch.ones(shape, dtype=torch.bool, device=env.device)
    touches = torch.zeros(shape, dtype=torch.long, device=env.device)
    takeover = []
    targets_during_takeover = []
    first_touch = torch.full(shape, -1, dtype=torch.long, device=env.device)
    tracking = []
    parking = []
    parking_xy = None
    parking_yaw = None
    for tick in range(800):
        command.targets[:] = targets if 150 <= tick < 300 else 0.
        obs, _, done, _ = wrapped.step(actor(obs))
        alive &= ~done.bool()
        forces = contact.data.net_forces_w.torch[:, other].norm(dim=-1)
        touched = (forces > 4.).any(dim=1)
        touches += touched
        first_touch[(first_touch < 0) & touched] = tick
        q = robot.data.joint_pos.torch
        dq = robot.data.joint_vel.torch
        q_target = robot.data.joint_pos_target.torch[:, legs]
        xy = robot.data.root_lin_vel_b.torch[:, :2]
        angular = robot.data.root_ang_vel_b.torch[:, 2]
        if 100 <= tick < 125:
            targets_during_takeover.append(q_target.clone())
            takeover.append(torch.cat((q_target - q[:, legs], dq[:, legs], dq[:, wheel]), dim=1))
        if 200 <= tick < 300:
            track = torch.linalg.vector_norm(xy - targets[:, :2], dim=1)
            tracking.append(torch.stack((track, (angular - targets[:, 2]).abs()), dim=1))
        if tick >= 350:
            position = robot.data.root_pos_w.torch[:, :2]
            x, y, z, w = robot.data.root_quat_w.torch.unbind(dim=1)
            yaw = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
            if parking_xy is None:
                parking_xy = position.clone()
                parking_yaw = yaw.clone()
            distance = (position - parking_xy).norm(dim=1)
            heading = ((yaw - parking_yaw + torch.pi) % (2. * torch.pi) - torch.pi).abs()
            parking.append(torch.stack((distance, torch.rad2deg(heading), xy.norm(dim=1)), dim=1))
    transient = torch.stack(takeover).abs().amax(dim=0)
    q_changes = torch.stack(targets_during_takeover).diff(dim=0)
    reversals = ((q_changes[1:] * q_changes[:-1] < 0)
                 & (q_changes[1:].abs() > .01) & (q_changes[:-1].abs() > .01)).sum(dim=(0, 2))
    move = torch.stack(tracking).mean(dim=0)
    parked = torch.stack(parking)
    displacement, heading, _ = parked.amax(dim=0).unbind(dim=1)
    residual = parked[:, :, 2].quantile(.95, dim=0)
    stable = alive & (touches == 0)
    held = stable & (displacement <= .02) & (heading <= 3.) & (residual <= .02)
    rows = []
    for i in range(env.num_envs):
        rows.append({
            "command": targets[i].tolist(), "alive": bool(alive[i]),
            "nonwheel_frames": int(touches[i]), "held": bool(held[i]),
            "first_nonwheel_s": None if first_touch[i] < 0 else (int(first_touch[i]) + 1) * env.step_dt,
            "takeover_target_reversals": int(reversals[i]),
            "peak_error_rad": float(transient[i, :12].max()),
            "peak_leg_speed_rad_s": float(transient[i, 12:24].max()),
            "peak_wheel_speed_rad_s": float(transient[i, 24:].max()),
            "xy_error_m_s": float(move[i, 0]), "yaw_error_rad_s": float(move[i, 1]),
            "hold_displacement_m": float(displacement[i]),
            "hold_yaw_deg": float(heading[i]), "hold_residual_p95_m_s": float(residual[i]),
        })
    summary = {
        "profile": args.profile, "checkpoint": str(args.checkpoint),
        "stable": int(stable.sum()), "held": int(held.sum()), "total": len(rows),
        "peak_leg_speed_p95_rad_s": float(transient[:, 12:24].amax(dim=1).quantile(.95)),
        "peak_error_p95_rad": float(transient[:, :12].amax(dim=1).quantile(.95)),
        "xy_tracking_mean_m_s": float(move[:, 0].mean()),
        "yaw_tracking_mean_rad_s": float(move[:, 1].mean()),
    }
    (args.out / "evaluation.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("R284_ROBUST_EVAL", json.dumps(summary), flush=True)