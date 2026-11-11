"""Evaluate left/right turns for the rolling controller, with no strafe target."""
import torch
from as2w_knee_envelope import MIN_KNEE_SURFACE_M
from as2w_training_safety import knee_surface_clearance

CASES = {
    "turn_left": (0., 0., .30),
    "turn_right": (0., 0., -.30),
    "arc_left": (.07, 0., .20),
    "arc_right": (.07, 0., -.20),
}


@torch.inference_mode()
def evaluate_turns(env, wrapped, actor, seed, *, fixed_stand=False, yaw_slew=0.):
    if env.num_envs % len(CASES):
        raise ValueError("Turn evaluation needs a multiple of four environments")
    trials = env.num_envs // len(CASES)
    targets = torch.tensor(list(CASES.values()), device=env.device).repeat_interleave(trials, dim=0)
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    wheel_ids = [sensor.body_names.index(name) for name in
                 ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
    other = [i for i, n in enumerate(sensor.body_names) if not n.endswith("_foot")]
    term = env.command_manager.get_term("base_velocity")
    wrapped.seed(seed + 1000)
    obs, _ = wrapped.reset()
    term.time_left.fill_(100.)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    contacts = torch.zeros_like(alive, dtype=torch.long)
    unloaded = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.long)
    max_unloaded = unloaded.clone()
    min_surface = torch.full((env.num_envs,),float("inf"),device=env.device)
    measured = []
    termination = env.termination_manager
    first_failure = {}
    legs = env.as2w_joint_ids[:12]
    entry = torch.zeros_like(contacts)
    exit_ = torch.zeros_like(contacts)
    start = 200 if getattr(env.cfg,"as2w_rolling_wheel_hold",False) else 100
    release = start+200
    for tick in range(release+(650 if fixed_stand else 30)):
        onset = start+25
        term.targets[:] = targets if onset<=tick<release else 0.
        if yaw_slew:
            ramp_up = max(0.,(tick-onset)*env.step_dt*yaw_slew)
            ramp_down = ((tick-release)*env.step_dt*yaw_slew)
            magnitude = targets[:,2].abs()
            allowed = (magnitude-ramp_down).clamp_min(0.)
            term.targets[:,2] = targets[:,2].sign()*torch.minimum(
                torch.minimum(magnitude,allowed),magnitude.new_full(magnitude.shape,ramp_up))
        obs, _, done, _ = wrapped.step(actor(obs))
        failed = alive & done.bool()
        for index in failed.nonzero(as_tuple=True)[0].tolist():
            first_failure[index] = dict(time_s=tick*env.step_dt,
                terms=[name for name in termination.active_terms
                       if termination.get_term(name)[index].item()])
        alive &= ~done.bool()
        forces = sensor.data.net_forces_w.torch
        contacts += (forces[:, other].norm(dim=-1) > 4.).any(dim=1)
        if fixed_stand and (tick < start or tick >= release):
            q_error = (robot.data.joint_pos.torch-robot.data.default_joint_pos.torch)[:,legs]
            fixed = ((forces[:,wheel_ids,2]>2.).all(dim=1) & (q_error.abs().amax(dim=1)<.18)
                     & (-robot.data.projected_gravity_b.torch[:,2]>.95)
                     & (robot.data.root_lin_vel_b.torch.norm(dim=1)<.06)
                     & (robot.data.root_ang_vel_b.torch.norm(dim=1)<.12))
            if tick < start:
                entry = torch.where(fixed,entry+1,0)
            else:
                exit_ = torch.where(fixed,exit_+1,0)
        if tick >= 100:
            min_surface = torch.minimum(min_surface,knee_surface_clearance(env).amin(dim=1))
            unloaded = torch.where(forces[:, wheel_ids, 2] < 2., unloaded+1, 0)
            max_unloaded = torch.maximum(max_unloaded, unloaded)
        if release-100 <= tick < release:
            xy = robot.data.root_lin_vel_b.torch[:, :2]
            yaw = robot.data.root_ang_vel_b.torch[:, 2]
            measured.append(torch.cat((xy, yaw[:, None]), dim=1))
    actual = torch.stack(measured).mean(dim=0)
    signed_yaw = actual[:, 2] * targets[:, 2].sign()
    stable = alive & (contacts == 0) & (max_unloaded.amax(dim=1) * env.step_dt <= .06)
    arc = targets[:, 0] > 0.
    direction_ok = signed_yaw >= .40 * targets[:, 2].abs()
    translation_ok = torch.where(arc, (actual[:, 0] >= .035) & (actual[:, 1].abs() <= .04),
                                 actual[:, :2].norm(dim=1) <= .05)
    stable &= min_surface >= MIN_KNEE_SURFACE_M
    passed = stable & direction_ok & translation_ok
    if fixed_stand:
        passed &= (entry*env.step_dt>=.40) & (exit_*env.step_dt>=2.)
    result = {}
    for i, name in enumerate(CASES):
        idx = slice(i*trials, (i+1)*trials)
        result[name] = {
            "passed": int(passed[idx].sum()), "total": trials,
            "stable": int(stable[idx].sum()), "wrong_direction": int((signed_yaw[idx] < 0.).sum()),
            "mean_yaw_rad_s": float(actual[idx, 2].mean()),
            "min_knee_surface_m": float(min_surface[idx].min()),
            "mean_vx_m_s": float(actual[idx, 0].mean()),
            "rows": [
                dict(passed=bool(passed[j]), alive=bool(alive[j]), direction_ok=bool(direction_ok[j]),
                     translation_ok=bool(translation_ok[j]), mean_velocity=actual[j].tolist(),
                     unloaded_s=float(max_unloaded[j].max())*env.step_dt,
                     min_knee_surface_m=float(min_surface[j]), nonwheel_frames=int(contacts[j]),
                     fixed_entry_s=int(entry[j])*env.step_dt, fixed_exit_s=int(exit_[j])*env.step_dt,
                     first_failure=first_failure.get(j))
                for j in range(idx.start,idx.stop)
            ],
        }
    return result