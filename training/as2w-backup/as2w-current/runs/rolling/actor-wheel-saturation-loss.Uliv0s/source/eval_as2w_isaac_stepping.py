"""Score foot lift and commanded motion in the native AS2-W stepping task."""
import json

import torch

from as2w_training_safety import MOTION_SUBSTEP_TERMS


@torch.inference_mode()
def evaluate(env, wrapped, actor, goals, feet, out):
    if env.num_envs % len(goals):
        raise ValueError("Evaluation environments must be a multiple of ten")
    trials = env.num_envs // len(goals)
    command = torch.tensor(goals, device=env.device).repeat_interleave(trials, dim=0)
    turn = command[:, 2] != 0
    speed = command[:, :2].norm(dim=1).clamp_min(.01)
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    nonwheel = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_foot")]
    wheel_ids = env.as2w_joint_ids[12:]
    term = env.command_manager.get_term("base_velocity")
    sequence = getattr(env.cfg, "as2w_sequence", False)
    warmup = round((2. if sequence else 1.) / env.step_dt)
    motion = round((env.cfg.as2w_motion_end_s if sequence else 5.)/env.step_dt)
    horizon = motion + (round(4./env.step_dt) if sequence else 0)
    legs = env.as2w_joint_ids[:12]
    report = {}
    termination = env.termination_manager
    for seed in (28501, 38501):
        wrapped.seed(seed)
        obs, _ = wrapped.reset()
        term.time_left.fill_(100.)
        alive = torch.ones_like(turn)
        touches = torch.zeros_like(turn, dtype=torch.long)
        lifts = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.long)
        airtime = torch.zeros_like(lifts)
        peak = torch.zeros_like(lifts, dtype=torch.float32)
        stats = torch.zeros((env.num_envs,5), device=env.device)
        samples = torch.zeros_like(touches)
        failures = {}
        fixed_frames = torch.zeros_like(touches)
        entry_frames = torch.zeros_like(touches)
        for tick in range(horizon):
            term.targets.copy_(command if tick < motion else torch.zeros_like(command))
            obs, _, done, _ = wrapped.step(actor(obs))
            failed = alive & done.bool()
            for index in failed.nonzero(as_tuple=True)[0].tolist():
                terms = [name for name in termination.active_terms if termination.get_term(name)[index].item()]
                substep = []
                if hasattr(env, "_motion_substep_last"):
                    substep = [name for name, bad in zip(MOTION_SUBSTEP_TERMS,
                               env._motion_substep_last[index].tolist()) if bad]
                failures[index] = dict(time_s=(tick+1)*env.step_dt,terms=terms,physics_checks=substep)
                if "nonwheel_contact" in substep and hasattr(env, "_motion_first_contact_body"):
                    body_index = int(env._motion_first_contact_body[index])
                    if body_index >= 0:
                        failures[index]["first_contact_body"] = sensor.body_names[body_index]
            alive &= ~done.bool()
            if not bool(alive.any()):
                break
            force = sensor.data.net_forces_w.torch[:, nonwheel].norm(dim=-1)
            touches += alive & (force > 4.).any(dim=-1)
            height, contact = feet(env)
            if tick < warmup or tick >= motion:
                error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:, legs]
                steady = (contact.all(dim=1) & (error.abs().amax(dim=1) < .18)
                          & (robot.data.root_lin_vel_b.torch.norm(dim=1) < .06)
                          & (robot.data.root_ang_vel_b.torch.norm(dim=1) < .12))
                steady &= -robot.data.projected_gravity_b.torch[:,2] > .95
                if tick < warmup:
                    entry_frames = torch.where(steady & alive,entry_frames+1,0)
                else:
                    fixed_frames = torch.where(steady & alive,fixed_frames+1,0)
                continue
            lifts += alive[:,None] & contact & (airtime >= 3) & (peak >= .02)
            airtime = torch.where(contact, 0, airtime + 1)
            peak = torch.where(contact, 0., torch.maximum(peak, height))
            xy = robot.data.root_lin_vel_b.torch[:, :2]
            yaw = robot.data.root_ang_vel_b.torch[:, 2]
            progress = torch.where(turn, yaw * command[:, 2].sign(),
                                   (xy * command[:, :2]).sum(dim=1) / speed)
            slip = robot.data.body_lin_vel_w.torch[:, env._stepping_body_ids, :2].norm(dim=-1)
            wheel_rate = robot.data.joint_vel.torch[:, wheel_ids].abs().mean(dim=1)
            values = torch.stack((progress, (xy - command[:, :2]).norm(dim=1),
                                  (yaw - command[:, 2]).abs(),
                                  (slip * contact).sum(dim=1) / contact.sum(dim=1).clamp_min(1),
                                  wheel_rate), dim=1)
            stats += torch.where(alive[:,None], values, 0.)
            samples += alive
        required = .5 * torch.where(turn, command[:, 2].abs(), speed)
        stats /= samples.clamp_min(1)[:,None]
        passed = alive & (touches == 0) & (lifts >= 2).all(dim=1) & (stats[:, 0] >= required)
        if sequence:
            passed &= (fixed_frames*env.step_dt >= 2.) & (entry_frames*env.step_dt >= .40)
        rows = []
        for i, values in enumerate(stats.tolist()):
            rows.append(dict(command=command[i].tolist(), alive=bool(alive[i]),
                             nonwheel_frames=int(touches[i]), lifts=lifts[i].tolist(),
                             signed_progress=values[0], xy_error_m_s=values[1],
                             yaw_error_rad_s=values[2], stance_slip_m_s=values[3],
                             wheel_speed_rad_s=values[4], fixed_exit_s=int(fixed_frames[i])*env.step_dt,
                             fixed_entry_s=int(entry_frames[i])*env.step_dt,
                             motion_observation_s=int(samples[i])*env.step_dt,
                             first_failure=failures.get(i), passed=bool(passed[i])))
        report[str(seed)] = rows
        print("stepping_eval", seed, passed.view(len(goals), trials).sum(dim=1).tolist(), flush=True)
    (out / "stepping-eval.json").write_text(json.dumps(report, indent=2) + "\n")
