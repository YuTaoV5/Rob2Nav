"""Independent MuJoCo rolling diagnostics with guarded actuation."""
import argparse
import json
from pathlib import Path

from as2w_rolling_brake_limits import rolling_brake_limits
from as2w_policy_provenance import checkpoint_record

import mujoco
from as2w_mujoco_pd import configure_native_pd
from as2w_knee_envelope import KNEE_ENVELOPE_CORNERS, MIN_KNEE_SURFACE_M
import numpy as np
import torch
from as2w_zero_memory import ZeroWheelMemory
from export_as2w_candidate_onnx import actor_from_state
from train_stair_low_speed_course import OmniEnvironment

KNEES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")


def residual_command(raw, filtered):
    command = raw.copy()
    command[:12] = .34*np.clip(raw[:12],-1.,1.)
    side = np.array((1.,-1.,1.,-1.))
    wheel = np.clip(raw[12:],-1.,1.)
    turn = min(1.,abs(filtered[2])/.20)
    steering = float(np.mean(wheel*side))
    gain = .66*(1.+.25*steering)
    linear = filtered[0]*(1.+.25*turn*float(wheel.mean()))-gain*filtered[2]*side
    command[12:] = np.clip(linear/(.086*16.)+.35*(1.-turn)*wheel,-6.,6.)
    return command


def advance_native_pd(env, action, monitor=None):
    data = env.data
    env.previous = action.copy()
    data.ctrl[env.motors] = np.r_[
        np.clip(env.defaults[:12]+.35*action[:12],env.leg_low,env.leg_high),
        np.clip(16.*action[12:],-env.wheel_limit,env.wheel_limit)]
    peak = 0.
    for _ in range(env.substeps):
        mujoco.mj_step(env.model,data)
        peak = max(peak,float(np.max(np.abs(data.actuator_force[env.motors])/env.torque_high)))
        if monitor is not None:
            monitor()
    env.tick += 1
    env.update_flat_command()
    done = data.xmat[env.base,8]<.80 or data.qpos[2]<.20
    return env.observe(),done,peak


@torch.inference_mode()
def trial(args, actor, seed, name, target, replay):
    env = OmniEnvironment(args.assets, 1, stop_training=True, flat_command_max=max(.10,args.speed))
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": (0.,0.,0.)})
    rng = np.random.default_rng(seed)
    if args.acceptance:
        model, data = env.model, env.data
        q_start = data.qpos.copy()
        model.body_mass[env.base] *= rng.uniform(.95,1.05)
        model.body_ipos[env.base,:2] += rng.uniform(-.01,.01,2)
        mujoco.mj_setConst(model,data)
        data.qpos[:] = q_start
        data.qvel[:] = 0.
        data.qvel[env.va[:12]] = rng.uniform(-.12,.12,12)
        mujoco.mj_forward(model,data)
        obs = env.observe()
    env.apply_motion_prior = False
    env.next_command_tick = 10**9
    center = env.defaults[:12]
    env.leg_low = center + .90*(env.leg_low-center)
    env.leg_high = center + .90*(env.leg_high-center)
    env.torque_low = .90*env.torque_low
    env.torque_high = .90*env.torque_high
    if args.native_pd:
        configure_native_pd(env.model,env.motors,env.kp,env.kd,
                            np.r_[env.leg_low,-env.wheel_limit],np.r_[env.leg_high,env.wheel_limit],
                            env.torque_high,velocity_axes=4)
    knee = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KNEES]
    if min(knee)<0:
        raise ValueError("Missing knee collider bodies")
    wheel_bodies = sorted(env.wheel_index, key=env.wheel_index.get)
    q_ref = np.r_[(env.data.qpos[env.qa[:12]]-center)/.35, np.zeros(4)]
    memory = ZeroWheelMemory()
    history = np.zeros(16,dtype=np.float32)
    last_obs = obs.copy()
    next_action = q_ref.copy()
    delayed = seed >= 47000
    packet_rng = np.random.default_rng(seed+7919)
    delivered = None
    dropped = False
    response = []
    park = []
    min_clearance = float("inf")
    min_surface = float("inf")
    corners = np.asarray(KNEE_ENVELOPE_CORNERS)
    nonwheel = limit = bad_stance = False
    first_nonwheel_contact = None
    max_torque_ratio = 0.
    unload = np.zeros(4,dtype=int)
    max_unload = 0
    wrench = np.zeros(6)
    heading0 = None
    heading_error = 0.
    brake_distance = None
    push_tick = round(rng.uniform(3.5,8.)/env.dt)
    stand_bias = torch.zeros((1,12))
    filtered_legs = torch.zeros(12)
    previous_legs = np.zeros(12, dtype=np.float32)
    wheel_reference = torch.zeros((1,4))
    hold_frames = torch.zeros(1,dtype=torch.long)
    recenter_frames = torch.zeros_like(hold_frames)
    park_lock_events = []
    park_odometry_samples = []
    park_transient = []
    fixed_entry = 0
    fixed_final = 0
    trace = []
    fixed_exit_checks = []
    sub_unload = np.zeros(4,dtype=int)
    sub_max_unload = 0

    def stance_invalid(q, heading):
        rear = q[6:8]
        xy = env.data.xpos[wheel_bodies,:2]
        lateral = xy @ np.array((-np.sin(heading),np.cos(heading)))
        crossed = np.any(lateral[[0,2]]-lateral[[1,3]]<.06)
        return bool(crossed or np.any((rear<.18) | (rear>1.40)))

    def contact_event(link):
        phase = "entry" if elapsed < start else "motion" if elapsed < release else "braking_or_hold"
        return dict(time_s=float(env.data.time - trial_start_time), phase=phase,
                    link=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(link)))

    def monitor_substep():
        nonlocal min_surface,min_clearance,nonwheel,limit,bad_stance,sub_unload,sub_max_unload

        nonlocal first_nonwheel_contact
        d = env.data
        q_step = d.qpos[env.qa[:12]]
        limit |= bool(np.any(q_step<env.leg_low) or np.any(q_step>env.leg_high))
        heading = np.arctan2(d.xmat[env.base,3],d.xmat[env.base,0])
        bad_stance |= stance_invalid(q_step,heading)
        loads = np.zeros(4)
        for idx,c in enumerate(d.contact):
            if env.ground not in c.geom:
                continue
            geom = c.geom[1] if c.geom[0]==env.ground else c.geom[0]
            link = env.model.geom_bodyid[geom]
            if link in env.wheel_index:
                mujoco.mj_contactForce(env.model,d,idx,wrench)
                loads[env.wheel_index[link]] += abs(wrench[0]*c.frame[2])
            else:
                nonwheel = True
                if first_nonwheel_contact is None:
                    first_nonwheel_contact = contact_event(link)
        if elapsed >= start:
            sub_unload = np.where(loads<2.,sub_unload+1,0)
            sub_max_unload = max(sub_max_unload,int(sub_unload.max()))
            rot = d.xmat[knee].reshape(4,3,3)
            world = d.xpos[knee,None,:]+np.einsum("kij,cj->kci",rot,corners)
            min_surface = min(min_surface,float(world[...,2].min()))
            min_clearance = min(min_clearance,float(d.xpos[knee,2].min()))

    start = 4. if args.rolling_wheel_hold else 2.
    release = start+6.
    trial_start_time = env.data.time
    raw_lateral_frames = 0
    effective_lateral_peak = 0.
    for tick in range(round((release+15.)/env.dt)):
        elapsed = tick*env.dt
        if elapsed < start or elapsed >= release:
            amount = max(0., 1.-1.2*max(0.,elapsed-release)/max(args.speed,.1)) if elapsed>=release else 0.
        else:
            amount = min(1., 1.2*(elapsed-start)/max(args.speed,.1))
        requested = np.asarray(target)*amount
        if name == "pure_vy" and elapsed >= start and not args.vy_neutral_control:
            requested[1] = .25
        if name == "forward_then_vy" and elapsed >= release and not args.vy_neutral_control:
            requested[1] = .25
        raw_lateral_frames += int(requested[1] != 0.)
        requested[1] = 0.
        env.target_goal = requested
        effective_lateral_peak = max(effective_lateral_peak,abs(float(obs[1])))
        if args.turn_yaw_slew and target[2]:
            rise = max(0.,elapsed-start)*args.turn_yaw_slew
            fall = abs(target[2])-max(0.,elapsed-release)*args.turn_yaw_slew
            env.target_goal[2] = np.sign(target[2])*max(0.,min(abs(target[2]),rise,fall))
        sampled = last_obs.copy() if delayed else obs.copy()
        sampled[:3] = obs[:3]
        sampled[42:60] = obs[42:60]
        sampled[42:58] = history
        if args.acceptance:
            sampled[3:6] += rng.normal(0.,.015,3)
            sampled[10:26] += rng.normal(0.,.004,16)
            sampled[26:42] += rng.normal(0.,.04,16)
        if args.packet_loss:
            dropped = packet_rng.random()<args.packet_loss and not dropped and delivered is not None
            if dropped:
                sampled[3:9],sampled[10:42] = delivered
            else:
                delivered = sampled[3:9].copy(),sampled[10:42].copy()
        frame = torch.from_numpy(memory.push(sampled))
        if tick % 20 == 0:
            replay.append(frame.numpy().copy())
        current = actor(frame)
        if args.mirror_rolling:
            from as2w_rolling_fixed_exit import mirror_rolling_joints, mirror_rolling_observation

            reflected = actor(mirror_rolling_observation(frame[None]))
            current = .5*(current + mirror_rolling_joints(reflected)[0])
        if args.rolling_leg_filter_hz:
            from as2w_rolling_filtered import filter_leg_residual
            filtered_legs = filter_leg_residual(filtered_legs, current[:12], env.dt, args.rolling_leg_filter_hz)
            current = torch.cat((filtered_legs, current[12:]))
        current = current.numpy()
        if args.rolling_fixed_exit:
            from as2w_rolling_fixed_exit import (parking_odometry, rolling_fixed_exit, rolling_loaded_stance, rolling_precalibrated_stand,
                                                rolling_stand_bias, rolling_wheel_hold)
            cmd = torch.from_numpy(obs[:3].copy()).to(torch.float32)
            cmd[0] += .4
            legs, wheels = rolling_fixed_exit(torch.from_numpy(current)[None], cmd[None],
                                             pure_turn_common_scale=args.pure_turn_common_scale)
            if args.rolling_stand_compensation:
                settled = None
                adapt_before_latch = args.park_adapt_before_latch and elapsed >= release
                if adapt_before_latch:
                    still = (float(cmd.abs().max()) < .005 and float(frame[38:42].abs().max()) < .50
                             and float(frame[26:38].abs().max()) < .50 and float(frame[3:6].norm()) < .15)
                    recenter_frames = recenter_frames+1 if still else torch.zeros_like(recenter_frames)
                    near_stand = float(frame[10:22].abs().max()) < .14
                    arm = (recenter_frames*env.dt >= .30) & near_stand
                else:
                    arm = None
                if args.rolling_wheel_hold:
                    wheel_angle = frame[None,22:26]
                    if args.park_odometry:
                        wheel_angle = parking_odometry(wheel_angle,frame[None,10:22],-frame[None,6:9])
                    latch_time = .35 if args.park_settle_joints else .12
                    was_latched = float(hold_frames[0]) * env.dt >= latch_time
                    wheel_reference,hold_frames,hold = rolling_wheel_hold(
                        wheel_reference,hold_frames,cmd[None],wheel_angle,
                        frame[None,38:42],frame[None,3:6],env.dt,common_feedback=args.park_odometry,
                        leg_speed=frame[None,26:38] if args.park_settle_joints else None,arm=arm, common_gain_scale=args.park_common_gain_scale,
                        common_speed_damping=args.park_common_speed_damping)
                    is_latched = float(hold_frames[0]) * env.dt >= latch_time
                    if elapsed >= release and not was_latched and is_latched:
                        park_lock_events.append((tick, env.data.qpos[:2].copy(), float(np.arctan2(env.data.xmat[env.base,3], env.data.xmat[env.base,0]))))
                    if args.park_odometry and is_latched and park_lock_events:
                        _, lock_xy, lock_heading = park_lock_events[-1]
                        forward_axis = np.array((np.cos(lock_heading), np.sin(lock_heading)))
                        actual_m = float((env.data.qpos[:2] - lock_xy) @ forward_axis)
                        angle_error = (wheel_angle - wheel_reference + 2.*torch.pi).remainder(4.*torch.pi) - 2.*torch.pi
                        estimated_m = .086 * float(angle_error.mean())
                        park_odometry_samples.append((estimated_m, actual_m))
                    wheels = wheels + hold
                    settled = hold_frames*env.dt>=.65
                signals = (stand_bias,cmd[None],frame[None,10:22],frame[None,3:6],
                           -frame[None,8],frame[None,38:42])
                if args.park_precalibrated:
                    stand_bias,stand = rolling_precalibrated_stand(
                        *signals,torch.maximum(recenter_frames,hold_frames) if adapt_before_latch else hold_frames,
                        frame.new_tensor([elapsed]),env.dt,continuous=args.park_continuous_bias,
                        straight_only=args.park_straight_bias,
                        recalibrate=(args.park_recalibrate_stop and elapsed >= release and
                                     bool(settled[0]) and float(frame[26:38].abs().max()) < .5)
                                    or (adapt_before_latch and float(hold_frames[0])*env.dt < .12))
                else:
                    stand_bias,stand = rolling_stand_bias(*signals,env.dt,enabled=settled)
                legs = legs+stand
                if args.rolling_loaded_stance:
                    legs = legs + rolling_loaded_stance(stand_bias,cmd[None],-frame[None,6:9])
            current = torch.cat((legs, wheels), dim=1)[0].numpy()
        elif args.rolling_residual:
            current = residual_command(current, obs[:3]+np.array((.4,0.,0.)))
        step_limit = .9 * env.dt / .35
        previous_legs += np.clip(current[:12]-previous_legs,-step_limit,step_limit)
        current[:12] = previous_legs
        history = current.copy()
        blend = np.clip(elapsed/2.,0.,1.)
        command = (1.-blend)*q_ref + blend*current
        last_obs = obs.copy()
        if args.acceptance and tick == push_tick:
            delta = rng.uniform((-.08,-.06),(.08,.06))
            rotation_push = rng.uniform(-.08,.08)
            if args.push_schedule=="all" or (args.push_schedule=="motion-only" and elapsed<release):
                env.data.qvel[:2] += delta
                env.data.qvel[5] += rotation_push
            push_tick += round(rng.uniform(3.5,8.)/env.dt)
        applied = next_action if delayed else command
        if args.native_pd:
            obs, done, peak = advance_native_pd(env,applied,monitor_substep if args.substep_safety else None)
        else:
            obs, _, done, _, _ = env.step(applied)
            peak = float(np.max(np.abs(env.data.ctrl[env.motors])/env.torque_high))
        next_action = command
        max_torque_ratio = max(max_torque_ratio,peak)
        rotation = env.data.xmat[env.base].reshape(3,3)
        v = rotation.T @ env.data.qvel[:3]
        yaw = env.data.qvel[5]
        loads = np.zeros(4)
        body = False
        for index, contact in enumerate(env.data.contact):
            if env.ground not in contact.geom:
                continue
            link = env.model.geom_bodyid[contact.geom[1] if contact.geom[0] == env.ground
                                         else contact.geom[0]]
            mujoco.mj_contactForce(env.model,env.data,index,wrench)
            if link in env.wheel_index:
                loads[env.wheel_index[link]] += abs(wrench[0]*contact.frame[2])
            else:
                body = True
                if first_nonwheel_contact is None:
                    first_nonwheel_contact = contact_event(link)
        nonwheel |= body
        heading = np.arctan2(rotation[1,0],rotation[0,0])
        if heading0 is None and elapsed >= 2.:
            heading0 = heading
        if elapsed >= 2.:
            unload = np.where(loads<2.,unload+1,0)
            max_unload = max(max_unload,int(unload.max()))
        q = env.data.qpos[env.qa[:12]]
        limit |= bool(np.any((q<env.leg_low) | (q>env.leg_high)))
        bad_stance |= stance_invalid(q,heading)
        fixed_checks = dict(
            wheel_support=bool(np.all(loads > 2.)),
            upright=bool(rotation[2,2] > .95),
            joint_pose=bool(np.max(np.abs(q-center)) < .18),
            planar_speed=bool(np.linalg.norm(v[:2]) < .06),
            yaw_rate=bool(abs(yaw) < .12),
        )
        fixed = all(fixed_checks.values())
        if elapsed < start:
            fixed_entry = fixed_entry+1 if fixed else 0
        elif elapsed >= release:
            fixed_exit_checks.append(fixed_checks)
            fixed_final = fixed_final+1 if fixed else 0
        if 2. <= elapsed:
            rot = env.data.xmat[knee].reshape(4,3,3)
            world = env.data.xpos[knee,None,:]+np.einsum("kij,cj->kci",rot,corners)
            min_surface = min(min_surface,float(world[...,2].min()))
            min_clearance = min(min_clearance,float(env.data.xpos[knee,2].min()))
        if release-2. <= elapsed < release:
            response.append((float(v[0]),float(v[1]),float(yaw)))
            heading_error = max(heading_error,abs(np.arctan2(np.sin(heading-heading0),
                                                            np.cos(heading-heading0))))
        if args.park_odometry and park_lock_events and is_latched:
            _, lock_xy, _ = park_lock_events[-1]
            wheel_target = env.data.ctrl[env.motors[12:]] if args.native_pd else 16. * applied[12:]
            park_transient.append(dict(
                time_since_release_s=elapsed - release,
                displacement_m=float(np.linalg.norm(env.data.qpos[:2] - lock_xy)),
                forward_velocity_m_s=float(v[0]),
                leg_position_error_rad=(q-center).tolist(),
                wheel_vertical_load_n=loads.tolist(),
                lateral_velocity_m_s=float(v[1]),
                hold_target_rad_s=(16. * hold[0]).tolist(),
                applied_wheel_target_rad_s=wheel_target.tolist(),
                wheel_speed_rad_s=env.data.qvel[env.va[12:]].tolist(),
            ))
        if elapsed >= release:
            park.append(np.r_[env.data.qpos[:2], heading, np.linalg.norm(v[:2]),abs(yaw)])
        if args.trace and seed==46001 and name=="forward" and (elapsed>=release-1. or elapsed<start):
            trace.append(dict(time_s=elapsed,command=env.command.tolist(),
                              body_velocity=v.tolist(),wheel_speed=env.data.qvel[env.va[12:]].tolist(),
                              target_wheel_speed=(16.*applied[12:]).tolist(), wheel_load=loads.tolist(),
                              q_error=float(np.max(np.abs(q-center))),
                              held_frames=int(hold_frames[0]),bias=stand_bias[0].tolist()))
        if done:
            break
    if trace:
        (args.output/"forward-trace.json").write_text(json.dumps(trace)+"\n")
    mean = np.mean(response,axis=0).tolist() if response else None
    stop = None
    held = False
    hold_drift = None
    hold_heading_deg = None
    hold_speed_p95 = None
    for i in range(25,len(park)):
        window = np.asarray(park[i-25:i])
        if np.all((window[:,3] <= .025) & (window[:,4] <= .06)):
            stop = i*env.dt
            braking = np.asarray(park[:i])
            brake_distance = float(np.linalg.norm(braking[:,:2]-braking[0,:2],axis=1).max())
            if len(park) >= i+501:
                end = np.asarray(park[i:i+501])
                hold_drift = float(np.linalg.norm(end[:,:2]-end[0,:2],axis=1).max())
                angle = float(np.max(np.abs(np.unwrap(end[:,2])-end[0,2])))
                hold_heading_deg = float(np.rad2deg(angle))
                hold_speed_p95 = float(np.quantile(end[:,3],.95))
                drift_cap, yaw_cap, speed_cap = (.05,5.,.035) if args.speed>=1. else (.02,3.,.02)
                held = (hold_drift<=drift_cap and angle<=np.deg2rad(yaw_cap)
                        and hold_speed_p95<=speed_cap)
            break
    unload_s = max_unload*env.dt
    if args.substep_safety:
        unload_s = max(unload_s,sub_max_unload*env.model.opt.timestep)
    stable = (not done and not nonwheel and not limit and not bad_stance and min_surface>=MIN_KNEE_SURFACE_M
              and min_clearance>=.045 and unload_s<=.06 and max_torque_ratio<=1.00001)
    tracked = False
    if mean is not None:
        x,y,yaw = mean
        if name == "pure_vy":
            tracked = np.hypot(x,y)<=.025 and abs(yaw)<=.06
        elif target[2]:
            translation_ok = (x>=.035 and abs(y)<=.04) if target[0] else np.hypot(x,y)<=.05
            tracked = yaw*np.sign(target[2])>=.40*abs(target[2]) and translation_ok
        else:
            lower = 2.95 if args.speed>=3. else .80*args.speed
            tracked = (lower<=x*np.sign(target[0])<=1.2*args.speed and abs(y)<=.2*args.speed+.01
                       and heading_error<=np.deg2rad(10.) and abs(yaw)<=.15)
    speed = abs(target[0])
    brake_time_limit, brake_distance_limit = rolling_brake_limits(speed)
    brake_ok = (stop is not None and stop<=brake_time_limit
                and brake_distance<=brake_distance_limit)
    passed = stable and tracked and brake_ok and held and fixed_entry*env.dt>=.40 and fixed_final*env.dt>=2.
    exit_window = fixed_exit_checks[-round(2. / env.dt):]
    exit_failures = {name: sum(not frame[name] for frame in exit_window)
                     for name in exit_window[0]} if exit_window else {}
    park_lock = None
    if park_lock_events and park:
        lock_tick, lock_xy, _ = park_lock_events[0]
        park_xy = np.asarray(park)[:, :2]
        release_tick = round(release / env.dt)
        lock_offset = max(0, lock_tick - release_tick)
        post_lock_xy = park_xy[lock_offset:]
        park_lock = dict(
            first_lock_s=(lock_tick - release_tick) * env.dt,
            lock_count=len(park_lock_events),
            release_to_lock_m=float(np.linalg.norm(lock_xy - park_xy[0])),
            post_lock_peak_m=float(np.linalg.norm(post_lock_xy - lock_xy, axis=1).max()),
        )
    if park_lock is not None and park_transient:
        peak_frame = max(range(len(park_transient)), key=lambda i: park_transient[i]["displacement_m"])
        park_lock["at_peak_displacement"] = park_transient[peak_frame]
        if args.trace:
            context_frames = round(1. / env.dt)
            park_lock["peak_context"] = park_transient[max(0, peak_frame-context_frames):peak_frame+context_frames+1]
    if park_lock is not None and park_odometry_samples:
        estimated, actual = np.asarray(park_odometry_samples).T
        park_lock.update(
            odometry_final_longitudinal_m=float(estimated[-1]),
            actual_final_longitudinal_m=float(actual[-1]),
            odometry_longitudinal_error_peak_m=float(np.abs(estimated - actual).max()),
        )
    return dict(seed=seed, scenario=name, command=target, mean_velocity=mean, park_lock=park_lock,
                raw_lateral_frames=raw_lateral_frames,effective_lateral_peak_m_s=effective_lateral_peak,
                passed=bool(passed), stable=bool(stable), tracked=bool(tracked),
                brake_distance_m=brake_distance, brake_time_limit_s=brake_time_limit,
                brake_distance_limit_m=brake_distance_limit, max_wheel_unload_s=unload_s,
                fixed_entry_s=fixed_entry*env.dt, fixed_final_s=fixed_final*env.dt,
                fixed_exit_window_frames=len(exit_window), fixed_exit_failed_frames=exit_failures,
                final_leg_position_rad=q.tolist(),
                final_leg_position_error_rad=(q-center).tolist(),
                final_leg_target_rad=env.data.ctrl[env.motors[:12]].tolist() if args.native_pd else None,
                final_stand_bias_rad=stand_bias[0].tolist() if args.rolling_stand_compensation and args.rolling_fixed_exit else None,
                final_leg_error_rad=float(np.max(np.abs(q-center))),
                final_wheel_load_n=loads.tolist(), final_body_speed_m_s=float(np.linalg.norm(v[:2])),
                first_nonwheel_contact=first_nonwheel_contact,
                terminated=bool(done), nonwheel=nonwheel, angle_violation=limit, stance_violation=bad_stance,
                max_sim_effort_ratio=max_torque_ratio,
                knee_surface_ok=min_surface>=MIN_KNEE_SURFACE_M,
                min_knee_surface_m=min_surface if np.isfinite(min_surface) else None,
                min_knee_clearance_m=min_clearance if np.isfinite(min_clearance) else None,
                stop_s=stop, hold_displacement_m=hold_drift, hold_heading_deg=hold_heading_deg,
                hold_speed_p95_m_s=hold_speed_p95, held=bool(held))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--speed",type=float,default=.30)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--rolling-residual",action="store_true")
    mode.add_argument("--rolling-fixed-exit",action="store_true")
    p.add_argument("--mirror-rolling",action="store_true")
    p.add_argument("--rolling-stand-compensation",action="store_true")
    p.add_argument("--rolling-wheel-hold",action="store_true")
    p.add_argument("--park-precalibrated",action="store_true")
    p.add_argument("--park-recalibrate-stop",action="store_true")
    p.add_argument("--park-adapt-before-latch",action="store_true")
    p.add_argument("--rolling-loaded-stance",action="store_true")
    p.add_argument("--park-odometry",action="store_true")
    p.add_argument("--park-common-gain-scale",type=float,default=1.)
    p.add_argument("--park-common-speed-damping", type=float, default=0.)
    p.add_argument("--park-settle-joints",action="store_true")
    p.add_argument("--park-continuous-bias",action="store_true")
    p.add_argument("--park-straight-bias",action="store_true")
    p.add_argument("--acceptance",action="store_true")
    p.add_argument("--turn-yaw-slew",type=float,default=0.)
    p.add_argument("--packet-loss",type=float,default=0.)
    p.add_argument("--push-schedule",choices=("all","motion-only","off"),default="all")
    p.add_argument("--native-pd",action="store_true")
    p.add_argument("--substep-safety",action="store_true")
    p.add_argument("--rolling-leg-filter-hz", type=float, default=0.)
    p.add_argument("--trace",action="store_true")
    p.add_argument("--trials",type=int,default=20)
    p.add_argument("--suppress-pure-turn-common",action="store_true")
    p.add_argument("--pure-turn-common-scale",type=float,default=1.)
    p.add_argument("--vy-diagnostic",action="store_true")
    p.add_argument("--vy-neutral-control",action="store_true")
    args=p.parse_args()
    if args.suppress_pure_turn_common:
        if args.pure_turn_common_scale != 1.:
            p.error("Choose one pure-turn common-mode option")
        args.pure_turn_common_scale = 0.
    if not 0. <= args.pure_turn_common_scale <= 1. or (
            args.pure_turn_common_scale != 1. and not args.rolling_fixed_exit):
        p.error("Pure-turn common-mode scale must be 0–1 and requires rolling fixed exit")
    if not 0. <= args.packet_loss <= .10:
        p.error("Packet loss must be between 0 and .10")
    if args.substep_safety and not args.native_pd:
        p.error("--substep-safety requires --native-pd")
    if not 0. <= args.turn_yaw_slew <= 3.:
        p.error("Turn yaw slew must be 0–3 rad/s²")
    if not 0. <= args.rolling_leg_filter_hz <= 20. or (args.rolling_leg_filter_hz and not args.rolling_fixed_exit):
        p.error("Leg filtering requires fixed-exit mode and a cutoff in (0, 20] Hz")
    if args.trials < 1:
        p.error("--trials must be positive")
    if (args.park_recalibrate_stop or args.park_adapt_before_latch) and not (args.park_precalibrated and args.rolling_wheel_hold):
        p.error("Stop recalibration requires precalibrated wheel hold")
    if args.park_recalibrate_stop and args.park_adapt_before_latch:
        p.error("Choose one stop recalibration timing")
    if args.park_adapt_before_latch and args.park_settle_joints:
        p.error("Pre-latch adaptation already requires quiet leg motion")
    if args.park_straight_bias and not args.park_continuous_bias:
        p.error("--park-straight-bias requires continuous bias")
    if args.park_continuous_bias and (not args.park_precalibrated or args.rolling_loaded_stance):
        p.error("Continuous bias requires precalibration without loaded-stance compensation")
    if args.park_settle_joints and not args.rolling_wheel_hold:
        p.error("--park-settle-joints requires --rolling-wheel-hold")
    if not 0. <= args.park_common_speed_damping <= 1. or (args.park_common_speed_damping and not args.park_odometry):
        p.error("Park speed damping must be in [0, 1] and requires odometry when enabled")
    if not 0. < args.park_common_gain_scale <= 1. or (args.park_common_gain_scale != 1. and not args.park_odometry):
        p.error("Park common gain scale must be in (0, 1] and requires odometry when changed")
    if args.park_odometry and not args.park_precalibrated:
        p.error("--park-odometry requires --park-precalibrated")
    if args.rolling_loaded_stance and not args.park_precalibrated:
        p.error("--rolling-loaded-stance requires --park-precalibrated")
    if args.park_precalibrated and not args.rolling_wheel_hold:
        p.error("--park-precalibrated requires --rolling-wheel-hold")
    if args.rolling_wheel_hold and not (args.rolling_fixed_exit and args.rolling_stand_compensation):
        p.error("--rolling-wheel-hold requires fixed-exit mode and stand compensation")
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=True)["actor_state_dict"]
    actor=actor_from_state(state,6.)
    cases = {"forward":(args.speed,0.,0.),"back":(-args.speed,0.,0.),
             "turn_left":(0.,0.,.10),"turn_right":(0.,0.,-.10)}
    seeds = (46005, 46006)
    if args.acceptance:
        cases.update(arc_left=(.05,0.,.08),arc_right=(.05,0.,-.08))
    else:
        seeds = (46001,46002,47001,47002)
    if args.vy_diagnostic:
        cases = {"pure_vy":(0.,0.,0.),"forward_then_vy":(args.speed,0.,0.)}
    args.output.mkdir(parents=True,exist_ok=True)
    replay = []
    rows=[trial(args,actor,seed,name,target,replay) for seed in seeds
          for name,target in cases.items()]
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/"policy-observations.npy",np.asarray(replay))
    settings = vars(args).copy()
    settings.pop("checkpoint")
    settings.pop("output")
    settings["assets"] = str(args.assets.resolve())
    contract = dict(**checkpoint_record(args.checkpoint),backend="mujoco",wrapped_leg_slew_rad_s=.9,**settings)
    (args.output/"mujoco-contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    (args.output/"mujoco-rolling-diagnostic.json").write_text(json.dumps(rows,indent=2)+"\n")
    if args.acceptance:
        scores = {name:sum(row["passed"] for row in rows if row["scenario"]==name) for name in cases}
        safety_gate = all(row["stable"] for row in rows)
        gate = (args.push_schedule=="all" and args.substep_safety and safety_gate
                and all(count>=.8*len(seeds) for count in scores.values()))
        report = dict(gate=gate,safety_gate=safety_gate,push_schedule=args.push_schedule,
                      trials_per_case=len(seeds), passed=scores, cap_m_s=args.speed,
                      park_recalibrate_stop=args.park_recalibrate_stop,
                      park_adapt_before_latch=args.park_adapt_before_latch,
                      native_pd=args.native_pd,substep_safety=args.substep_safety,park_precalibrated=args.park_precalibrated,
                      rolling_loaded_stance=args.rolling_loaded_stance,park_odometry=args.park_odometry,
                      park_continuous_bias=args.park_continuous_bias,park_straight_bias=args.park_straight_bias,
                      rolling_leg_filter_hz=args.rolling_leg_filter_hz)
        report_name = "mujoco-rolling-acceptance.json" if args.push_schedule=="all" else "mujoco-rolling-diagnostic-summary.json"
        (args.output/report_name).write_text(json.dumps(report,indent=2)+"\n")
    print("AS2W_MUJOCO_ROLLING",len(rows),sum(r["held"] for r in rows),flush=True)


if __name__=="__main__":
    main()