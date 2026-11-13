"""Read-only recovery acceptance with explicit initial and final support checks."""
import json
import torch
from as2w_knee_envelope import MIN_KNEE_SURFACE_M
from as2w_training_safety import MOTION_SUBSTEP_TERMS, knee_clearance, knee_surface_clearance
from as2w_isaac_flat_env_all import JOINTS


@torch.inference_mode()
def evaluate(env, wrapped, actor, out):
    obs, _ = wrapped.reset()
    initial = env.recovery_kind.clone()
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    wheel = [sensor.body_names.index(j) for j in JOINTS[12:]]
    knee = [robot.body_names.index(j) for j in JOINTS[8:12]]
    legs = env.as2w_joint_ids[:12]
    nonwheel = [i for i,n in enumerate(sensor.body_names) if not n.endswith("_foot")]
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    initial_contact = torch.zeros_like(alive)
    initial_height = (robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]).clone()
    initial_upright = (-robot.data.projected_gravity_b.torch[:, 2]).clone()
    peak_height = torch.zeros_like(initial_height)
    max_fixed_window = torch.zeros_like(initial)
    first_fixed = torch.full_like(initial, -1)
    steady = torch.zeros_like(initial)
    regained = torch.zeros_like(alive)
    contact_after = torch.zeros_like(alive)
    min_knee = torch.full_like(initial_height, float("inf"))
    min_surface = min_knee.clone()
    maximum_handoff_frames = torch.zeros_like(initial)
    failed_fixed_gate_frames = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.int32)
    termination = env.termination_manager
    failures = {}
    for tick in range(550):
        obs, _, done, _ = wrapped.step(actor(obs))
        failed = alive & done.bool()
        for index in failed.nonzero(as_tuple=True)[0].tolist():
            terms = [name for name in termination.active_terms if termination.get_term(name)[index].item()]
            substep = []
            if hasattr(env, "_motion_substep_last"):
                substep = [name for name, bad in zip(MOTION_SUBSTEP_TERMS,
                           env._motion_substep_last[index].tolist()) if bad]
            failures[index] = dict(time_s=(tick+1)*env.step_dt,terms=terms,physics_checks=substep)
            if "nonwheel_contact" in substep:
                body_index = int(env._motion_first_contact_body[index])
                if body_index >= 0:
                    failures[index]["first_nonwheel_body"] = sensor.body_names[body_index]
        alive &= ~done.bool()
        h = robot.data.root_pos_w.torch[:,2] - env.scene.env_origins[:,2]
        upright = -robot.data.projected_gravity_b.torch[:,2]
        peak_height = torch.maximum(peak_height, torch.where(alive,h,0.))
        forces = sensor.data.net_forces_w.torch
        body = (forces[:,nonwheel].norm(dim=-1) > 4.).any(dim=1)
        if tick < 12:
            initial_contact |= body & alive
        wheels = (forces[:,wheel,2] > 2.).all(dim=1)
        q = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch)[:,legs]
        speed = robot.data.root_lin_vel_b.torch.norm(dim=1)
        spin = robot.data.root_ang_vel_b.torch.norm(dim=1)
        clearance = knee_clearance(env, robot.data.body_pos_w.torch[:,knee]).amin(dim=1)
        established = (h >= .31) & (upright >= .95)
        regained |= established & alive
        contact_after |= body & regained & alive
        envelope = knee_surface_clearance(env).amin(dim=1)
        min_surface = torch.minimum(min_surface,torch.where(regained & alive,envelope,float("inf")))
        min_knee = torch.minimum(min_knee, torch.where(regained & alive, clearance, float("inf")))
        gates = torch.stack((h >= .31, upright >= .95, wheels, ~body,
                             q.abs().amax(dim=1) < .18, speed < .06, spin < .12), dim=1)
        if tick >= 400:
            failed_fixed_gate_frames += (~gates & alive[:, None]).int()
        maximum_handoff_frames = torch.maximum(maximum_handoff_frames, env.handoff_frames)
        fixed = gates.all(dim=1) & alive
        steady = torch.where(fixed, steady+1, 0)
        max_fixed_window = torch.maximum(max_fixed_window, torch.where(alive,steady,0))
        first_fixed = torch.where((first_fixed < 0) & (steady>=100), tick, first_fixed)
        if not bool(alive.any()):
            break
    passed = (alive & (first_fixed >= 0) & (steady>=100) & ~contact_after &
              (min_knee >= env.cfg.as2w_min_knee_clearance_m) & (min_surface >= MIN_KNEE_SURFACE_M))
    result = {}
    families = ("near_stand", "prone", "left_tilt", "right_tilt")
    fixed_gate_names = ("height", "upright", "wheel_support", "body_clear",
                        "joint_error", "linear_speed", "angular_speed")
    for k,name in enumerate(families):
        ids = (initial==k).nonzero().flatten().tolist()
        rows = []
        for i in ids:
            rows.append(dict(passed=bool(passed[i]), alive=bool(alive[i]), first_failure=failures.get(i),
                             initial_body_contact=bool(initial_contact[i]),
                             initial_upright=float(initial_upright[i]),
                             initial_height_m=float(initial_height[i]),
                             fixed_stand_s=float(first_fixed[i])*env.step_dt if first_fixed[i]>=0 else None,
                             min_knee_after_upright_m=float(min_knee[i]) if torch.isfinite(min_knee[i]) else None,
                             min_knee_surface_m=float(min_surface[i]) if torch.isfinite(min_surface[i]) else None,
                             nonwheel_after_upright=bool(contact_after[i]),
                             peak_alive_height_m=float(peak_height[i]), final_height_m=float(h[i]),
                             final_upright=float(upright[i]), final_wheel_support=bool(wheels[i]),
                             final_max_stand_joint_error_rad=float(q[i].abs().max()),
                             max_fixed_stand_window_s=int(max_fixed_window[i])*env.step_dt,
                             max_handoff_progress_s=int(maximum_handoff_frames[i])*env.step_dt,
                             final_150_frames_failed=dict(zip(fixed_gate_names,
                                 failed_fixed_gate_frames[i].tolist()))))
        result[name] = dict(total=len(rows), passed=sum(r["passed"] for r in rows),
                            actual_start_support=sum(r["initial_body_contact"] for r in rows), rows=rows)
    (out/"getup-evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
    print("AS2W_GETUP_EVAL", {k:(v["passed"],v["total"],v["actual_start_support"])
                               for k,v in result.items()}, flush=True)