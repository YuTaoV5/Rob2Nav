"""Isaac Lab switch diagnostic with the same ONNX actors and timeline as MuJoCo."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def evaluate(args, physics):
    import numpy as np
    import torch

    from isaaclab.envs import mdp
    from as2w_isaac_flat_env_all import FlatEnvironment, JOINTS, configuration
    from as2w_knee_envelope import MIN_KNEE_SURFACE_M
    from as2w_mode_switch import Feedback, ModeSwitch, Phase, fixed_stand_checks
    from as2w_predictive_joint_guard import guarded_target
    from as2w_shadow_replay import Policy
    from as2w_switch_targets import SwitchTargets, diagnostic_duration, request_at
    from as2w_training_safety import (
        MOTION_SUBSTEP_TERMS, apply_safety, configure_motion_substep_audit,
        install_motion_substep_audit, knee_surface_clearance,
    )

    torch.set_num_threads(1)
    fixed_kp = args.fixed_leg_kp
    fixed_kd = 2. * (fixed_kp / 55.) ** .5
    args.zero_memory = False
    args.command_stage = "all"
    args.zero_focus = args.direction_focus = False
    cfg = configuration(args,physics)
    if args.expanded:
        from as2w_isaac_robust_env import robust_config

        args.profile = "expanded"
        cfg.events = robust_config(args,physics).events
        cfg.events.gains = None
    push_end_s = diagnostic_duration(args.reverse_order)
    observation_end_s = push_end_s + args.observe_after_push_s
    cfg.episode_length_s = observation_end_s + 1.
    push_times = [[] for _ in range(args.num_envs)]
    t = 0.
    limits = apply_safety(cfg,args.urdf,perturb=args.expanded,surface_guard=True)
    if args.expanded:
        def bounded_push(env, env_ids, velocity_range, asset_cfg):
            if t <= push_end_s:
                mdp.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg=asset_cfg)
                for index in env_ids.tolist():
                    push_times[index].append(round(t, 6))

        cfg.events.bounded_push.func = bounded_push
    if args.native_position_pd:
        cfg.actions.switch_wheel_position = mdp.JointPositionActionCfg(
            asset_name="robot",joint_names=list(JOINTS[12:]),preserve_order=True,
            scale=1.,use_default_offset=False)
    configure_motion_substep_audit(cfg)
    env = FlatEnvironment(cfg)
    try:
        rolling_audit = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        physics_audit = install_motion_substep_audit(env,rolling_active=rolling_audit)
        env.reset()
        dt = env.step_dt
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        ids = env.as2w_joint_ids
        wheel_ids = [sensor.body_names.index(name) for name in JOINTS[12:]]
        other_ids = [i for i,name in enumerate(sensor.body_names) if not name.endswith("_foot")]
        axis_bounds = np.array([limits["angle_rad"][name] for name in JOINTS[:12]])
        lo,hi = axis_bounds.T
        wheel_limit = np.array([limits["wheel_rad_s"][name] for name in JOINTS[12:]])
        effort = torch.tensor([limits["effort_nm"][n] for n in JOINTS],device=env.device)
        nominal = robot.data.default_joint_pos.torch[:,ids[:12]].cpu().numpy()
        q = robot.data.joint_pos.torch[:,ids].cpu().numpy()
        policies = {"rolling":Policy(args.rolling,(65,16)),"stepping":Policy(args.stepping,(47,12))}
        supervisors = [ModeSwitch() for _ in range(env.num_envs)]
        controllers = [SwitchTargets(policies,nominal[i],lo,hi,wheel_limit,dt,q[i,12:])
                       for i in range(env.num_envs)]
        active = np.ones(env.num_envs,dtype=bool)
        unloaded = np.zeros((env.num_envs,4),dtype=int)
        max_effort = np.zeros(env.num_envs)
        clearance_min = np.full(env.num_envs,np.inf)
        phases = [[] for _ in supervisors]
        failures = [None for _ in supervisors]
        stand_checks = [{} for _ in supervisors]
        stand_leg_errors = [{} for _ in supervisors]
        distance = [dict(rolling=0.,stepping=0.) for _ in supervisors]
        terms = env.termination_manager
        if args.native_position_pd:
            expected = ["joint_pos","wheel_vel","switch_wheel_position"]
            if list(env.action_manager.active_terms)!=expected:
                raise RuntimeError(f"Unexpected switch action order: {env.action_manager.active_terms}")
        previous_gains = None
        rng = np.random.default_rng(args.seed)
        sensor_lag = rng.integers(2,size=env.num_envs) if args.expanded else np.zeros(env.num_envs,dtype=int)
        target_lag = rng.integers(2,size=env.num_envs) if args.expanded else np.zeros(env.num_envs,dtype=int)
        prior_sensors = [None]*env.num_envs
        delivered_sensors = [None]*env.num_envs
        missed = np.zeros(env.num_envs,dtype=bool)
        prior_targets = None
        signed_body_distance = [dict(rolling=0.,stepping=0.) for _ in supervisors]
        for tick in range(round(observation_end_s/dt)+1):
            t = tick*dt
            requested = request_at(t,args.reverse_order)
            q = robot.data.joint_pos.torch[:,ids].cpu().numpy()
            dq = robot.data.joint_vel.torch[:,ids].cpu().numpy()
            gyro = robot.data.root_ang_vel_b.torch.cpu().numpy()
            gravity = robot.data.projected_gravity_b.torch.cpu().numpy()
            speed = robot.data.root_lin_vel_b.torch.norm(dim=1).cpu().numpy()
            forces = sensor.data.net_forces_w.torch
            load = forces[:,wheel_ids,2].cpu().numpy()
            body = (forces[:,other_ids].norm(dim=-1)>4.).any(dim=1).cpu().numpy()
            clearance = knee_surface_clearance(env).amin(dim=1).cpu().numpy()
            clearance_min[active] = np.minimum(clearance_min[active],clearance[active])
            old_x = robot.data.root_pos_w.torch[:,0].cpu().numpy().copy()
            forward_speed = robot.data.root_lin_vel_b.torch[:,0].cpu().numpy()
            action = np.zeros((env.num_envs,20 if args.native_position_pd else 16),dtype=np.float32)
            stiffness = np.broadcast_to(np.r_[[55.]*12,[0.]*4],(env.num_envs,16)).astype(np.float32).copy()
            damping = np.full((env.num_envs,16),2.,dtype=np.float32)
            decisions = [None for _ in supervisors]
            rolling_motion = np.zeros(env.num_envs,dtype=bool)
            for i in np.flatnonzero(active):
                supervisor = supervisors[i]
                controller = controllers[i]
                moving = supervisor.phase==Phase.ACTIVE and requested==supervisor.policy
                rolling = supervisor.policy=="rolling" and np.linalg.norm(controller.filtered)>.02
                unloaded[i] = np.where(rolling & (load[i]<2.),unloaded[i]+1,0)
                safe = not body[i] and clearance[i]>=MIN_KNEE_SURFACE_M
                guard = (np.all((lo<=q[i,:12])&(q[i,:12]<=hi))
                         and unloaded[i].max()*dt<=.06 and max_effort[i]<=1.00001)
                feedback = Feedback(
                    t,0.,float(speed[i]),float(gyro[i,2]),float(-gravity[i,2]),
                    float(np.max(np.abs(q[i,:12]-nominal[i]))),int(np.sum(load[i]>2.)),
                    not moving,bool(safe),guard_ok=bool(guard))
                if supervisor.phase==Phase.EXIT and t-supervisor.phase_start>=.75:
                    key = f"{supervisor.policy}_exit"
                elif supervisor.phase==Phase.STAND and requested is None and t>=28.:
                    key = "final_stand"
                else:
                    key = None
                if key is not None:
                    counts = stand_checks[i].setdefault(key,{"frames": 0})
                    counts["frames"] += 1
                    error = q[i,:12] - nominal[i]
                    total, peak = stand_leg_errors[i].setdefault(
                        key, (np.zeros(12), np.zeros(12)))
                    total += error
                    np.maximum(peak, np.abs(error), out=peak)
                    for term, passed_check in fixed_stand_checks(feedback).items():
                        counts[term] = counts.get(term,0)+int(not passed_check)
                decision = supervisor.step(feedback,requested)
                tag = (decision.phase.value,decision.policy)
                if not phases[i] or tag!=tuple(phases[i][-1]["state"]):
                    phases[i].append(dict(time_s=t,state=tag,fault=decision.fault))
                if decision.fault:
                    failures[i] = dict(time_s=t,terms=[decision.fault])
                    active[i] = False
                    continue
                decisions[i] = decision
                frame = (q[i].copy(),dq[i].copy(),gyro[i].copy(),gravity[i].copy())
                sensor_frame = prior_sensors[i] if sensor_lag[i] and prior_sensors[i] is not None else frame
                prior_sensors[i] = frame
                if args.expanded:
                    missed[i] = rng.random()<.02 and not missed[i] and delivered_sensors[i] is not None
                    if not missed[i]:
                        sampled_q,sampled_dq,sampled_gyro,sampled_gravity = sensor_frame
                        delivered_sensors[i] = (sampled_q+rng.normal(0.,.004,16),
                            sampled_dq+rng.normal(0.,.04,16),sampled_gyro+rng.normal(0.,.015,3),
                            sampled_gravity+rng.normal(0.,.006,3))
                    sensor_frame = delivered_sensors[i]
                fixed,fixed_wheel,legs,wheel = controller.update(
                    *sensor_frame,decision,moving,t,fade_stand_bias=True,
                    recalibrate_fixed_exit=args.recalibrate_fixed_exit)
                weight = decision.weight
                rolling_motion[i] = (decision.policy=="rolling" and weight>.99
                                     and np.linalg.norm(controller.filtered)>.02)
                if decision.policy=="stepping":
                    leg = guarded_target(legs,q[i,:12],dq[i,:12],lo,hi)
                    if args.native_position_pd:
                        stiffness[i,:12] = fixed_kp*(1.-weight)+85.*weight
                        damping[i,:12] = fixed_kd*(1.-weight)+3.2*weight
                        stiffness[i,12:] = 45.*weight
                        damping[i,12:] = 2.*(1.-weight)+4.*weight
                        action[i,16:] = controller.wheel_anchor
                        target_q = (fixed_kp*(1.-weight)*fixed+85.*weight*leg)/stiffness[i,:12]
                        target_wheel = 2.*(1.-weight)*fixed_wheel/damping[i,12:]
                    else:
                        equivalent_leg = q[i,:12]+(85.*(leg-q[i,:12])-1.2*dq[i,:12])/55.
                        equivalent_wheel = 22.5*(controller.wheel_anchor-q[i,12:])-dq[i,12:]
                        target_q = (1.-weight)*fixed+weight*equivalent_leg
                        target_wheel = (1.-weight)*fixed_wheel+weight*equivalent_wheel
                else:
                    target_q = (1.-weight)*fixed+weight*legs
                    if args.native_position_pd:
                        stiffness[i,:12] = fixed_kp*(1.-weight)+55.*weight
                        damping[i,:12] = fixed_kd*(1.-weight)+2.*weight
                        target_q = (fixed_kp*(1.-weight)*fixed+55.*weight*legs)/stiffness[i,:12]
                    target_wheel = (1.-weight)*fixed_wheel+weight*wheel
                action[i,:12] = (np.clip(target_q,lo,hi)-nominal[i])/.35
                action[i,12:16] = np.clip(target_wheel,-wheel_limit,wheel_limit)/16.
            if not active.any():
                break
            if args.expanded:
                current_targets = action.copy(),stiffness.copy(),damping.copy()
                if prior_targets is not None:
                    for latest,prior in zip((action,stiffness,damping),prior_targets):
                        latest[target_lag>0] = prior[target_lag>0]
                prior_targets = current_targets
            if args.native_position_pd and (previous_gains is None
                    or not np.array_equal(stiffness,previous_gains[0])
                    or not np.array_equal(damping,previous_gains[1])):
                kp = torch.as_tensor(stiffness,device=env.device)
                kd = torch.as_tensor(damping,device=env.device)
                robot.write_joint_stiffness_to_sim_index(stiffness=kp,joint_ids=ids)
                robot.write_joint_damping_to_sim_index(damping=kd,joint_ids=ids)
                for j,part in enumerate(("hip","thigh","calf","foot")):
                    actuator = robot.actuators[part]
                    actuator.stiffness[:] = kp[:,4*j:4*j+4]
                    actuator.damping[:] = kd[:,4*j:4*j+4]
                previous_gains = stiffness,damping
            rolling_audit.copy_(torch.as_tensor(rolling_motion,device=env.device))
            _,_,terminated,truncated,_ = env.step(torch.as_tensor(action,device=env.device))
            ratio = (robot.data.applied_torque.torch[:,ids].abs()/effort).amax(dim=1).cpu().numpy()
            max_effort[active] = np.maximum(max_effort[active],ratio[active])
            done = (terminated|truncated).cpu().numpy()
            new_x = robot.data.root_pos_w.torch[:,0].cpu().numpy()
            for i in np.flatnonzero(active):
                if done[i]:
                    checks = [name for name,bad in zip(MOTION_SUBSTEP_TERMS,env._motion_substep_last[i].tolist()) if bad]
                    failures[i] = dict(time_s=t+dt,terms=[k for k in terms.active_terms if terms.get_term(k)[i].item()],
                                       physics_checks=checks)
                    active[i] = False
                elif decisions[i].phase==Phase.ACTIVE:
                    mode = decisions[i].policy
                    distance[i][mode] += float(new_x[i]-old_x[i])
                    signed_body_distance[i][mode] += float(forward_speed[i])*dt
        rows = []
        for i,supervisor in enumerate(supervisors):
            fixed_s = 0. if supervisor.fixed_since is None else t-supervisor.fixed_since
            passed = (active[i] and supervisor.phase==Phase.STAND and fixed_s>=2.
                      and distance[i]["rolling"]>=.30 and distance[i]["stepping"]>=.15)
            leg_error_report = {
                key: {
                    name: {"mean_signed_rad": float(total[j] / stand_checks[i][key]["frames"]),
                           "peak_abs_rad": float(peak[j])}
                    for j, name in enumerate(JOINTS[:12])
                }
                for key, (total, peak) in stand_leg_errors[i].items()
            }
            rows.append(dict(passed=bool(passed),phases=phases[i],failure=failures[i],
                             fixed_stand_violation_frames=stand_checks[i], fixed_stand_joint_errors=leg_error_report,
                             push_times_s=push_times[i],
                             final_stand_s=fixed_s,distances_m=distance[i],body_forward_m=signed_body_distance[i],
                             native_position_pd=args.native_position_pd,expanded=args.expanded,
                             observation_delay_s=int(sensor_lag[i])*dt,target_delay_s=int(target_lag[i])*dt,
                             min_knee_surface_m=float(clearance_min[i]),
                             max_effort_fraction_of_guard=float(max_effort[i])))
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"switch-diagnostic.json").write_text(json.dumps(rows,indent=2)+"\n")
        contract = dict(backend="isaacsim_physx",expanded=args.expanded,seed=args.seed,
                        native_position_pd=args.native_position_pd,reverse_order=args.reverse_order,
                        recalibrate_fixed_exit=args.recalibrate_fixed_exit, motion_substep_audit=physics_audit,
                        observe_after_push_s=args.observe_after_push_s, fixed_leg_kp=fixed_kp,
                        weights={name:actor.provenance for name,actor in policies.items()})
        (args.out/"switch-contract.json").write_text(json.dumps(contract,indent=2)+"\n")
        print("AS2W_PHYSICAL_SWITCH_DIAGNOSTIC",sum(r["passed"] for r in rows),len(rows),flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("usd","urdf","rolling","stepping","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--num-envs",type=int,default=2)
    parser.add_argument("--seed",type=int,default=48001)
    parser.add_argument("--reverse-order",action="store_true")
    parser.add_argument("--native-position-pd",action="store_true")
    parser.add_argument("--expanded",action="store_true")
    parser.add_argument("--recalibrate-fixed-exit",action="store_true")
    parser.add_argument("--observe-after-push-s",type=float,choices=(0.,3.),default=0.)
    parser.add_argument("--fixed-leg-kp",type=float,choices=(55.,75.),default=55.)
    add_launcher_args(parser)
    args = parser.parse_args()
    if (args.expanded or args.fixed_leg_kp != 55.) and not args.native_position_pd:
        parser.error("Expanded switch and alternate fixed gain require native position PD")
    with launch_simulation(SimulationCfg(device=args.device,dt=.005),args) as physics:
        evaluate(args,physics)


if __name__=="__main__":
    main()
