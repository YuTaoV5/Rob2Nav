"""Closed-loop MuJoCo diagnostic for rolling, fixed stand, and stepping switches."""
import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from as2w_knee_envelope import KNEE_ENVELOPE_CORNERS, MIN_KNEE_SURFACE_M
from as2w_mode_switch import Feedback, ModeSwitch, Phase, fixed_stand_checks
from as2w_predictive_joint_guard import guarded_target
from as2w_shadow_replay import Policy
from as2w_switch_targets import SwitchTargets, diagnostic_duration, request_at
from eval_as2w_mujoco_stepping import contact_forces
from train_stair_low_speed_course import OmniEnvironment


def geometry(data, knees, corners):
    rot = data.xmat[knees].reshape(4,3,3)
    points = data.xpos[knees,None,:]+np.einsum("kij,cj->kci",rot,corners)
    return float(points[...,2].min())


@torch.inference_mode()
def trial(assets, policies, seed, predictive_guard, fade_stand_bias, expanded, reverse, recalibrate_fixed_exit, fixed_leg_kp, fixed_wheel_kd, observe_after_push_s):
    env = OmniEnvironment(assets,1,stop_training=True)
    env.reset(seed=seed,options={"scene_seed":seed,"target":(0.,0.,0.)})
    model,data = env.model,env.data
    dt = env.dt
    rng = np.random.default_rng(seed)
    if expanded:
        q_start = data.qpos.copy()
        model.body_mass[env.base] *= rng.uniform(.95,1.05)
        model.body_ipos[env.base,:2] += rng.uniform(-.01,.01,2)
        model.geom_friction[env.ground,0] = rng.uniform(.58,.85)
        mujoco.mj_setConst(model,data)
        data.qpos[:] = q_start
        data.qvel[:] = 0.
        data.qvel[env.va[:12]] = rng.uniform(-.12,.12,12)
        mujoco.mj_forward(model,data)
    lag = int(rng.integers(2)) if expanded else 0
    delay = round(.005*rng.integers(5)/model.opt.timestep) if expanded else 0
    command_ring = [None]*(delay+1)
    cursor = 0
    push_tick = round(rng.uniform(4.,8.)/dt)
    prior_sensor = delivered = None
    dropped = False
    q0 = env.defaults[:12]
    lo = q0+.90*(env.leg_low-q0)
    hi = q0+.90*(env.leg_high-q0)
    effort = .90*env.torque_high
    knees = [model.body(f"{leg}_calf").id for leg in ("FL","FR","RL","RR")]
    corners = np.asarray(KNEE_ENVELOPE_CORNERS)
    wheel_bodies = sorted(env.wheel_index,key=env.wheel_index.get)
    twist,wrench = np.empty(6),np.empty(6)
    supervisor = ModeSwitch()
    controller = SwitchTargets(policies,q0,lo,hi,env.wheel_limit,dt,data.qpos[env.qa[12:]])
    phases = []
    final_stand_violations = Counter()
    speed_violation_times = []
    push_times = []
    distances = {"rolling":0.,"stepping":0.}
    guard_fault = None
    guard_axis = []
    interventions = 0
    min_surface = float("inf")
    max_effort = 0.
    unload_s = 0.
    last_x = float(data.qpos[0])
    fixed_leg_kd = 2. * np.sqrt(fixed_leg_kp / 55.)
    push_end_s = diagnostic_duration(reverse)
    for tick in range(round((push_end_s + observe_after_push_s)/dt)+1):
        t = tick*dt
        requested = request_at(t,reverse)
        rotation = data.xmat[env.base].reshape(3,3)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,env.base,twist,1)
        loads,nonwheel = contact_forces(env,wrench,strict=True)
        clearance = geometry(data,knees,corners)
        q = data.qpos[env.qa]
        dq = data.qvel[env.va]
        q_error = q[:12]-q0
        speed = np.linalg.norm(twist[3:])
        stick_moving = (supervisor.phase==Phase.ACTIVE and requested==supervisor.policy)
        feedback = Feedback(t,0.,float(speed),float(twist[2]),float(rotation[2,2]),
                            float(np.max(np.abs(q_error))),int(np.sum(loads>2.)),
                            not stick_moving,not nonwheel and clearance>=MIN_KNEE_SURFACE_M,
                            guard_ok=guard_fault is None)
        current_sensor = (q.copy(),dq.copy(),twist[:3].copy(),-rotation[2].copy())
        selected = prior_sensor if lag and prior_sensor is not None else current_sensor
        prior_sensor = current_sensor
        if expanded:
            dropped = rng.random()<.02 and not dropped and delivered is not None
            if not dropped:
                sample_q,sample_dq,sample_g,sample_gravity = selected
                q_noise = rng.normal(0.,.004,16)
                dq_noise = rng.normal(0.,.04,16)
                g_noise = rng.normal(0.,.015,3)
                delivered = (sample_q+q_noise,sample_dq+dq_noise,sample_g+g_noise,sample_gravity)
            selected = delivered
        if requested is None and t >= (29. if reverse else 28.) and supervisor.phase == Phase.STAND:
            final_stand_violations["frames"] += 1
            checks = fixed_stand_checks(feedback)
            final_stand_violations.update(name for name, ok in checks.items() if not ok)
            if not checks["speed"]:
                speed_violation_times.append(round(t, 6))
        decision = supervisor.step(feedback,requested)
        tag = (decision.phase.value,decision.policy)
        if not phases or tag!=tuple(phases[-1]["state"]):
            phases.append(dict(time_s=t,state=tag,fault=decision.fault))
        if decision.fault:
            break
        fixed_target,fixed_wheel_velocity,leg_target,wheel_target = controller.update(
            *selected,decision,stick_moving,t,fade_stand_bias=fade_stand_bias,
            recalibrate_fixed_exit=recalibrate_fixed_exit)
        mode = decision.policy
        rolling_active = mode=="rolling" and decision.weight>.99 and np.linalg.norm(controller.filtered)>.02
        if expanded and tick==push_tick and t <= push_end_s:
            push_times.append(round(t, 6))
            data.qvel[:2] += rng.uniform((-.08,-.06),(.08,.06))
            data.qvel[5] += rng.uniform(-.08,.08)
            push_tick += round(rng.uniform(4.,8.)/dt)
        controls = (fixed_target,fixed_wheel_velocity,leg_target,wheel_target,
                    controller.wheel_anchor.copy(),decision.weight,mode)
        for _ in range(env.substeps):
            command_ring[cursor] = controls
            delayed = command_ring[(cursor-delay)%len(command_ring)] or controls
            cursor = (cursor+1)%len(command_ring)
            fixed_q,fixed_vel,leg,wheel,anchor,weight,policy = delayed
            q = data.qpos[env.qa]
            dq = data.qvel[env.va]
            fixed_tau = np.r_[fixed_leg_kp*(fixed_q-q[:12])-fixed_leg_kd*dq[:12],fixed_wheel_kd*(fixed_vel-dq[12:])]
            if policy=="stepping":
                target = guarded_target(leg,q[:12],dq[:12],lo,hi) if predictive_guard else leg
                interventions += int(np.max(np.abs(target-leg))>1e-4)
                policy_tau = np.r_[85.*(target-q[:12])-3.2*dq[:12],45.*(anchor-q[12:])-4.*dq[12:]]
            else:
                policy_tau = np.r_[55.*(leg-q[:12])-2.*dq[:12],2.*(wheel-dq[12:])]
            combined = (1.-weight)*fixed_tau+weight*policy_tau
            data.ctrl[env.motors] = np.clip(combined,-effort,effort)
            mujoco.mj_step(model,data)
            max_effort = max(max_effort,float(np.max(np.abs(data.actuator_force[env.motors])/effort)))
            q_act = data.qpos[env.qa[:12]]
            clear = geometry(data,knees,corners)
            loads,nonwheel = contact_forces(env,wrench,strict=True)
            min_surface = min(min_surface,clear)
            unload_s = unload_s+model.opt.timestep if rolling_active and np.any(loads<2.) else 0.
            yaw = np.arctan2(data.xmat[env.base,3],data.xmat[env.base,0])
            lateral = data.xpos[wheel_bodies,:2] @ np.array((-np.sin(yaw),np.cos(yaw)))
            rear = q_act[6:8]
            bad_stance = np.any(lateral[[0,2]]-lateral[[1,3]]<.06) or np.any((rear<.18) | (rear>1.40))
            if not np.all((lo<=q_act)&(q_act<=hi)):
                guard_fault = "joint_angle"
                guard_axis = [env.contract["joint_names"][j] for j in np.flatnonzero((q_act<lo)|(q_act>hi))]
            elif nonwheel or clear<MIN_KNEE_SURFACE_M or np.any(data.xpos[knees,2]<.045):
                guard_fault = "nonwheel_or_knee_contact"
            elif bad_stance:
                guard_fault = "stance"
            elif unload_s>.060001:
                guard_fault = "rolling_wheel_unload"
            elif data.xmat[env.base,8]<.80:
                guard_fault = "tilt"
            elif max_effort>1.00001:
                guard_fault = "effort"
            if guard_fault:
                break
        x = float(data.qpos[0])
        if mode and decision.phase==Phase.ACTIVE:
            distances[mode] += x-last_x
        last_x = x
    final_stand_s = 0. if supervisor.fixed_since is None else t-supervisor.fixed_since
    passed = (guard_fault is None and not decision.fault and decision.phase==Phase.STAND and final_stand_s>=2.
              and distances["rolling"]>=.30 and distances["stepping"]>=.15)
    return dict(seed=seed,passed=bool(passed),phases=phases,fault=decision.fault,
                expanded=expanded,reverse_order=reverse,observation_delay_s=lag*dt,
                target_delay_s=delay*model.opt.timestep,physics_step_s=model.opt.timestep,
                physical_guard_fault=guard_fault,physical_guard_axes=guard_axis,
                predictive_interventions=interventions,fade_stand_bias=fade_stand_bias,
                final_stand_s=final_stand_s,distances_m=distances,
                final_stand_violation_frames=dict(final_stand_violations),
                final_stand_speed_violation_times_s=speed_violation_times, push_times_s=push_times,
                min_knee_surface_m=min_surface,max_effort_fraction_of_guard=max_effort)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets",type=Path)
    parser.add_argument("rolling",type=Path)
    parser.add_argument("stepping",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--trials",type=int,default=2)
    parser.add_argument("--predictive-guard",action="store_true")
    parser.add_argument("--fade-stand-bias",action="store_true")
    parser.add_argument("--expanded",action="store_true")
    parser.add_argument("--recalibrate-fixed-exit",action="store_true")
    parser.add_argument("--reverse-order",action="store_true")
    parser.add_argument("--fixed-leg-kp",type=float,choices=(55.,75.),default=55.)
    parser.add_argument("--fixed-wheel-kd", type=float, choices=(2., 4.), default=2.)
    parser.add_argument("--observe-after-push-s",type=float,choices=(0.,3.),default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policies = {"rolling":Policy(args.rolling,(65,16)),"stepping":Policy(args.stepping,(47,12))}
    rows = [trial(args.assets,policies,48001+i,args.predictive_guard,args.fade_stand_bias,args.expanded,args.reverse_order,
                  args.recalibrate_fixed_exit,args.fixed_leg_kp,args.fixed_wheel_kd,args.observe_after_push_s) for i in range(args.trials)]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"switch-diagnostic.json").write_text(json.dumps(rows,indent=2)+"\n")
    contract = dict(backend="mujoco",expanded=args.expanded,seed=48001,reverse_order=args.reverse_order,
                    recalibrate_fixed_exit=args.recalibrate_fixed_exit, fixed_leg_kp=args.fixed_leg_kp, fixed_wheel_kd=args.fixed_wheel_kd, observe_after_push_s=args.observe_after_push_s,
                    motion_substep_audit={"physics_dt_s": rows[0]["physics_step_s"]},
                    weights={name:actor.provenance for name,actor in policies.items()})
    (args.output/"switch-contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print("AS2W_PHYSICAL_SWITCH_DIAGNOSTIC",sum(r["passed"] for r in rows),len(rows),flush=True)


if __name__=="__main__":
    main()
