"""Replay guarded AS2-W stepping in MuJoCo with independent terrain checks."""
import argparse
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record

import mujoco
import numpy as np
import torch

from as2w_mujoco_pd import configure_native_pd
from as2w_predictive_joint_guard import guarded_target
from as2w_knee_envelope import DISTAL_CALF_ENVELOPE_CORNERS, KNEE_ENVELOPE_CORNERS, MIN_KNEE_SURFACE_M, calf_hull_vertices
from phase_gate_control import guarded_sequence_action, sequence_swing
from export_as2w_candidate_onnx import actor_from_state
from train_stair_low_speed_course import OmniEnvironment

FLAT = dict(forward=(.10,0.,0.), back=(-.10,0.,0.), left=(0.,.10,0.), right=(0.,-.10,0.),
            front_left=(.071,.071,0.), front_right=(.071,-.071,0.),
            back_left=(-.071,.071,0.), back_right=(-.071,-.071,0.),
            turn_left=(0.,0.,.30), turn_right=(0.,0.,-.30))




def surface(env, xy, margin=0.):
    result = np.zeros(xy.shape[:-1])
    for geom in env.surfaces:
        if geom == env.ground:
            continue
        center = env.data.geom_xpos[geom]
        extent = env.model.geom_size[geom]
        inside = (np.abs(xy-center[:2]) <= extent[:2]+margin).all(axis=-1)
        result = np.maximum(result,np.where(inside,center[2]+extent[2],0.))
    return result


def contact_forces(env, wrench, *, strict=False, offending_contacts=None):
    loads = np.zeros(4)
    nonwheel = False
    for i, contact in enumerate(env.data.contact):
        g1,g2 = contact.geom
        if g1 in env.surfaces:
            terrain_geom, robot_geom = g1, g2
        elif g2 in env.surfaces:
            terrain_geom, robot_geom = g2, g1
        else:
            continue
        link = env.model.geom_bodyid[robot_geom]
        mujoco.mj_contactForce(env.model,env.data,i,wrench)
        if link in env.wheel_index:
            loads[env.wheel_index[link]] += abs(wrench[0]*contact.frame[2])
        else:
            touch = wrench[0] > 4. or (strict and contact.dist <= 0.)
            nonwheel |= touch
            if touch and offending_contacts is not None:
                offending_contacts.append(dict(
                    link=env.model.body(link).name,
                    robot_geom=env.model.geom(robot_geom).name,
                    point_link_m=(env.data.xmat[link].reshape(3, 3).T @
                                  (contact.pos - env.data.xpos[link])).tolist(),
                    terrain=env.model.geom(terrain_geom).name,
                    point_world_m=contact.pos.tolist(),
                    normal_world=contact.frame[:3].tolist(),
                    separation_m=float(contact.dist),
                    normal_force_n=float(wrench[0]),
                ))
    return loads,nonwheel


@torch.inference_mode()
def trial(args, actor, name, command, seed, replay):
    rng = np.random.default_rng(seed)
    stair = name in ("up","down")
    env = OmniEnvironment(args.assets,0 if stair else 1,stop_training=True,
                          stair_rise=.08 if args.wheel_teacher_replay else .04 + .01*(seed % 3))
    env.reset(seed=seed,options={"scene_seed":seed,"target":(0.,0.,0.)})
    m,d = env.model,env.data
    q0 = env.defaults[:12]
    lo,hi = q0 + .90*(env.leg_low-q0),q0 + .90*(env.leg_high-q0)
    predictive_lo, predictive_hi = lo.copy(), hi.copy()
    if args.rear_stance_predictor:
        predictive_lo[6:8] = np.maximum(predictive_lo[6:8], .18)
        predictive_hi[6:8] = np.minimum(predictive_hi[6:8], 1.40)
    effort = .90*env.torque_high
    m.body_mass[env.base] *= rng.uniform(.95,1.05)
    m.body_ipos[env.base,:2] += rng.uniform(-.01,.01,2)
    m.geom_friction[list(env.surfaces),0] = rng.uniform(.64,.80)
    mujoco.mj_setConst(m,d)
    d.qpos[:2] = (0.,0.)
    d.qpos[2] = .38+rng.uniform(-.012,.012)
    d.qpos[3:7] = (1.,0.,0.,0.)
    if name == "down":
        top = max(env.surfaces,key=env.surfaces.get)
        d.qpos[0] = d.geom_xpos[top,0]-m.geom_size[top,0]+args.down_approach
        d.qpos[2] += d.geom_xpos[top,2]+m.geom_size[top,2]
        d.qpos[3:7] = (0.,0.,0.,1.)
    d.qpos[env.qa[:12]] = q0+rng.uniform(-.10,.10,12)
    d.qpos[env.qa[12:]] = 0.
    d.qvel[:] = 0.
    d.qvel[env.va[:12]] = rng.uniform(-.12,.12,12)
    mujoco.mj_forward(m,d)
    wheel = list(env.wheel_index)
    knees = [m.body(f"{leg}_calf").id for leg in ("FL","FR","RL","RR")]
    twist,wrench = np.empty(6),np.empty(6)
    noise = np.r_[[.015]*3,[.006]*3,[.004]*12,[.04]*12]
    old_sensor = None
    delivered = None
    missed = False
    filtered = np.zeros(3,dtype=np.float32)
    history_action = np.zeros(12,dtype=np.float32)
    if args.go2_transfer:
        actor.reset()
        joint_names = env.contract["joint_names"][:12]
        go2_order = [joint_names.index(f"{leg}_{axis}") for leg in ("FL","FR","RL","RR")
                     for axis in ("hip","thigh","calf")]
        as2w_order = np.argsort(go2_order)
        if args.go2_foot_retarget:
            from as2w_go2_retarget import Go2WheelRetarget
            retarget = Go2WheelRetarget(q0[go2_order].reshape(4,3))

    lag = round(.005*rng.integers(5)/m.opt.timestep)
    obs_lag = rng.integers(2)
    ring = np.repeat(d.qpos[env.qa[:12]][None],lag+1,axis=0)
    cursor = 0
    leg_gains = np.tile((55., 2.), (12, 1))
    leg_groups = ("hip", "thigh", "calf")
    if args.leg_pd_group == "all":
        leg_gains[:] = args.leg_pd
    else:
        group = leg_groups.index(args.leg_pd_group)
        leg_gains[4*group:4*(group+1)] = args.leg_pd
    kp = np.r_[leg_gains[:, 0], [45.]*4]*rng.uniform(.90,1.10,16)
    kd = np.r_[leg_gains[:, 1], [4.]*4]*rng.uniform(.90,1.10,16)
    if args.native_pd:
        configure_native_pd(m,env.motors,kp,kd,np.r_[lo,[-1.]*4],np.r_[hi,[1.]*4],effort)
    dt = env.dt
    motion_end = 27.
    if args.wheel_teacher_replay:
        names = args.teacher_data["joint_names"].tolist()
        teacher_order = [names.index(name) for name in env.contract["joint_names"][:12]]
        teacher_q = args.teacher_data["as2w_joint_target_rad"].reshape(-1,12)[:,teacher_order]
        teacher_time = (args.teacher_data["time_s"] - args.teacher_data["time_s"][0]) / args.teacher_rate
        motion_end = 2. + float(teacher_time[-1])
    push = round(rng.uniform(3.5,8.)/dt)
    lifts,air,peak = np.zeros(4,dtype=int),np.zeros(4,dtype=int),np.zeros(4)
    origin = d.qpos[:2].copy()
    height0 = float(surface(env,origin))
    fixed = 0
    entry = 0
    contact_frames = 0
    min_surface = float("inf")
    motion_base_heights = []
    min_distal_calf = np.full(4, np.inf)
    distal_calf_warning_s = [None] * 4
    distal_calf_corners = np.asarray(DISTAL_CALF_ENVELOPE_CORNERS)
    calf_hulls = calf_hull_vertices(args.assets / "official-model/robots/as2w_description/meshes") if args.calf_mesh_diagnostic else None
    min_calf_hull = np.full(4, np.inf)
    calf_hull_warning_s = [None] * 4
    calf_collision_geoms = [np.flatnonzero((m.geom_bodyid == knee) &
                                           ((m.geom_contype | m.geom_conaffinity) != 0)) for knee in knees]
    if args.calf_mesh_diagnostic and any(not len(geoms) for geoms in calf_collision_geoms):
        raise ValueError("Missing enabled calf collision geometry")
    min_calf_geom_distance = np.full(4, np.inf)
    calf_geom_warning_s = [None] * 4
    max_effort = 0.
    reason = None
    failure_joints = []
    failure_up = None
    failure_detail = None
    offending_contacts = []
    progress = []
    gait_time = wait_time = 0.
    gate_frames = short_steps = 0
    short_slot = -1
    for tick in range(round((motion_end+4.)/dt)):
        age = tick*dt
        target_command = command if 2. <= age < motion_end else np.zeros(3)
        filtered += .12*(target_command-filtered)
        if age >= motion_end:
            filtered[:] = 0.
        rot = d.xmat[env.base].reshape(3,3)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,env.base,twist,1)
        q = d.qpos[env.qa[:12]]
        sensed = np.r_[twist[:3],-rot[2],q-q0,d.qvel[env.va[:12]]] + rng.normal(size=30)*noise
        sensor = old_sensor if obs_lag and old_sensor is not None else sensed
        old_sensor = sensed
        if args.packet_loss:
            missed = rng.random()<args.packet_loss and not missed and delivered is not None
            if missed:
                sensor = delivered
            else:
                delivered = sensor
        if 2. < age < motion_end:
            candidate = gait_time + dt
            slot_length = args.sequence_period / 4.
            slot_index = int(candidate / slot_length)
            swing_start = slot_index * slot_length + .85 * args.sequence_period / 12.
            blocked = False
            if gait_time < swing_start <= candidate:
                swing_leg = (0, 3, 1, 2)[slot_index % 4]
                load, _ = contact_forces(env, wrench)
                underloaded = np.any(np.delete(load, swing_leg) < 8.)
                blocked = underloaded and wait_time < .30
                if underloaded and not blocked:
                    short_slot = slot_index
                    short_steps += 1
            if blocked:
                wait_time += dt
                gate_frames += 1
            else:
                gait_time = candidate
                wait_time = 0.
        phase = 2.*np.pi*gait_time/args.sequence_period
        slot_phase = gait_time * 12. / args.sequence_period
        short_phase = slot_phase - 3. * short_slot
        short_envelope = min(np.clip((short_phase-.85)/.5,0.,1.),
                             np.clip((3.-short_phase)/.5,0.,1.)) if short_slot >= 0 else 0.
        stride_gain = args.sequence_stride * (1. - .35 * short_envelope)
        if args.wheel_teacher_replay:
            blend = np.clip(2.*(age-2.),0.,1.)*np.clip(motion_end+1.-age,0.,1.)
            q_ref = np.array([np.interp(age-2.,teacher_time,axis) for axis in teacher_q.T])
            desired = np.clip(q0+blend*(q_ref-q0),lo+args.target_margin,hi-args.target_margin)
            if tick % 10 == 0:
                replay.append(sensor.copy())
        elif args.go2_transfer:
            relative_q = sensor[6:18][go2_order]
            joint_velocity = sensor[18:30][go2_order]
            if args.go2_foot_retarget:
                as2w_q = (q0[go2_order]+relative_q).reshape(4,3)
                delta = .0001*joint_velocity.reshape(4,3)
                relative_q = retarget.observation(as2w_q).ravel()
                joint_velocity = (
                    retarget.observation(as2w_q+delta)-retarget.observation(as2w_q-delta)
                ).ravel()/.0002
            obs = np.r_[sensor[:3]*.25,sensor[3:6],filtered*np.array([2.,2.,.25]),
                        relative_q,joint_velocity*.05,
                        history_action].astype(np.float32)
            output = actor(torch.from_numpy(obs[None]))
            if isinstance(output,tuple):
                output = output[0]
            history_action = output[0].numpy()
            blend = np.clip(age-2.,0.,1.)*np.clip(motion_end+1.-age,0.,1.)
            target = q0+.25*blend*history_action[as2w_order]
            if args.go2_foot_retarget:
                mapped = retarget.target(history_action.reshape(4,3)).ravel()[as2w_order]
                target = q0+blend*(mapped-q0)
            if tick % 10 == 0:
                replay.append(obs)
            desired = np.clip(target,lo+args.target_margin,hi-args.target_margin)
        else:
            obs = np.r_[filtered,sensor,history_action,np.sin(phase),np.cos(phase)].astype(np.float32)
            if tick % 10 == 0:
                replay.append(obs)
            action = actor(torch.from_numpy(obs[None]))
            sensed_t = torch.from_numpy(sensor[None].astype(np.float32))
            history_action = guarded_sequence_action(torch.from_numpy(filtered[None]),
                torch.tensor([age]),action,sensed_t[:,12:14],motion_end,period=args.sequence_period,
                stride_gain=stride_gain,gravity=sensed_t[:,3:6] if args.sequence_attitude else None,
                gyro=sensed_t[:,:3] if args.sequence_attitude else None,
                lateral_shift_scale=args.lateral_shift_scale,front_lift_m=args.front_lift_m,
                front_reach_scale=args.front_reach_scale,
                damp_lateral_shift=args.damp_lateral_shift, body_raise_m=args.body_raise_m,
                symmetric_front_support=args.symmetric_front_support, phase_time=torch.tensor([gait_time]))[0].numpy()
            desired = np.clip(q0+.30*history_action,lo+args.target_margin,hi-args.target_margin)
        if tick == push:
            d.qvel[:2] += rng.uniform((-.08,-.06),(.08,.06))
            d.qvel[5] += rng.uniform(-.08,.08)
            push += round(rng.uniform(3.5,8.)/dt)
        for substep in range(env.substeps):
            ring[cursor] = desired
            applied = ring[(cursor-lag)%len(ring)]
            cursor = (cursor+1)%len(ring)
            if args.predictive_guard:
                applied = guarded_target(applied,d.qpos[env.qa[:12]],d.qvel[env.va[:12]],
                                         predictive_lo,predictive_hi)
            q_target = np.r_[applied,np.zeros(4)]
            if args.native_pd:
                d.ctrl[env.motors] = q_target
            else:
                tau = kp*(q_target-d.qpos[env.qa])-kd*d.qvel[env.va]
                d.ctrl[env.motors] = np.clip(tau,-effort,effort)
            mujoco.mj_step(m,d)
            max_effort = max(max_effort,float(np.max(np.abs(d.actuator_force[env.motors])/effort)))
            if max_effort > 1.00001:
                reason = "effort"
                break
            q_step = d.qpos[env.qa[:12]]
            invalid = np.flatnonzero((q_step<lo) | (q_step>hi))
            if len(invalid):
                reason = "joint_angle"
                failure_joints = [
                    dict(name=env.contract["joint_names"][j], angle_rad=float(q_step[j]),
                         target_rad=float(applied[j]), lower_rad=float(lo[j]),
                         upper_rad=float(hi[j]), velocity_rad_s=float(d.qvel[env.va[j]]),
                         actuator_effort_nm=float(d.actuator_force[env.motors[j]]),
                         effort_limit_nm=float(effort[j]))
                    for j in invalid
                ]
                loads, simultaneous_contact = contact_forces(
                    env, wrench, strict=True, offending_contacts=offending_contacts)
                failure_detail = dict(wheel_vertical_load_n=loads.tolist(),
                                      simultaneous_nonwheel_contact=bool(simultaneous_contact))
                break
            rotation = d.xmat[knees].reshape(4,3,3)
            points = d.xpos[knees,None,:]+np.einsum("kij,cj->kci",rotation,KNEE_ENVELOPE_CORNERS)
            clearance = float(np.min(points[...,2]-surface(env,points[...,:2],.02)))
            min_surface = min(min_surface,clearance)
            distal_points = d.xpos[knees, None, :] + np.einsum("kij,cj->kci", rotation, distal_calf_corners)
            distal_gap = np.min(distal_points[..., 2] - surface(env, distal_points[..., :2]), axis=1)
            min_distal_calf = np.minimum(min_distal_calf, distal_gap)
            for leg in np.flatnonzero(distal_gap < MIN_KNEE_SURFACE_M):
                if distal_calf_warning_s[leg] is None:
                    distal_calf_warning_s[leg] = float(d.time)
            if calf_hulls is not None:
                calf_geom_distances = np.empty(4)
                for leg, vertices in enumerate(calf_hulls):
                    world = vertices @ rotation[leg].T + d.xpos[knees[leg]]
                    gap = float(np.min(world[:, 2] - surface(env, world[:, :2])))
                    min_calf_hull[leg] = min(min_calf_hull[leg], gap)
                    if gap < MIN_KNEE_SURFACE_M and calf_hull_warning_s[leg] is None:
                        calf_hull_warning_s[leg] = float(d.time)
                    distance = min(mujoco.mj_geomDistance(m, d, int(geom), int(terrain), .10, None)
                                   for geom in calf_collision_geoms[leg] for terrain in env.surfaces)
                    calf_geom_distances[leg] = distance
                    min_calf_geom_distance[leg] = min(min_calf_geom_distance[leg], distance)
                    if distance < MIN_KNEE_SURFACE_M and calf_geom_warning_s[leg] is None:
                        calf_geom_warning_s[leg] = float(d.time)
            if contact_forces(env,wrench,strict=True,offending_contacts=offending_contacts)[1]:
                contact_frames += 1
                reason = "nonwheel_contact"
                _, swing_lift, _ = sequence_swing(torch.tensor([phase/(2.*np.pi)]))
                hub = d.xpos[wheel]
                failure_detail = {
                    "links": sorted({contact["link"] for contact in offending_contacts}),
                    "contacts": offending_contacts,
                    "swing_lift_weight": swing_lift[0].tolist(),
                    "wheel_tread_clearance_m": (hub[:,2]-surface(env,hub[:,:2])-.086).tolist(),
                    "min_knee_envelope_m": clearance,
                    "calf_geom_distances_m": [float(distance) if np.isfinite(distance) else None for distance in calf_geom_distances] if calf_hulls is not None else None,
                }
                break
            knee_center = d.xpos[knees]
            center_clearance = np.min(knee_center[:,2]-surface(env,knee_center[:,:2],.02))
            if clearance < MIN_KNEE_SURFACE_M or center_clearance < .045:
                reason = "knee_surface"
                failure_detail = {"envelope_m": clearance, "center_m": float(center_clearance)}
                break
            yaw = np.arctan2(d.xmat[env.base,3],d.xmat[env.base,0])
            lateral = d.xpos[wheel] @ np.array((-np.sin(yaw),np.cos(yaw),0.))
            width = lateral[[0,2]]-lateral[[1,3]]
            crossed = np.any(width<.06)
            rear = q_step[6:8]
            if crossed or np.any(rear<.18) or np.any(rear>1.40):
                reason = "stance"
                failure_detail = {"lateral_width_m": width.tolist(), "rear_thigh_rad": rear.tolist()}
                break
            if d.xmat[env.base,8] < .80:
                reason = "tilt"
                break
        if reason is not None:
            failure_up = d.xmat[env.base,6:9].tolist()
            break
        loads,_ = contact_forces(env,wrench)
        supported = loads > 4.
        if 2. <= age < motion_end:
            motion_base_heights.append(float(d.qpos[2] - surface(env, d.qpos[:2])))
            wheel_height = d.xpos[wheel,2]-surface(env,d.xpos[wheel,:2])-.086
            lifts += supported & (air>=3) & (peak>=.02)
            air = np.where(supported,0,air+1)
            peak = np.where(supported,0.,np.maximum(peak,wheel_height))
            if command[2]:
                progress.append(float(twist[2]*np.sign(command[2])))
            else:
                progress.append(float(np.dot(twist[3:5],command[:2])/np.linalg.norm(command[:2])))
        else:
            steady = (supported.all() and d.xmat[env.base,8]>.95 and np.max(np.abs(q_step-q0))<.18
                      and np.linalg.norm(twist[3:])<.06 and np.linalg.norm(twist[:3])<.12)
            if age < 2.:
                entry = entry+1 if steady else 0
            else:
                fixed = fixed+1 if steady else 0
    distance = float(np.linalg.norm(d.qpos[:2]-origin))
    forward_progress = float(d.qpos[0]-origin[0]) * (-1. if name=="down" else 1.)
    elevation = float(surface(env,d.qpos[:2]))-height0
    signed = float(np.mean(progress)) if progress else None
    required = .5*(abs(command[2]) if command[2] else np.linalg.norm(command[:2]))
    moved = signed is not None and signed>=required
    if stair:
        moved = forward_progress>=1. and elevation*(1. if name=="up" else -1.)>=.045
    passed = reason is None and fixed*dt>=2. and entry*dt>=.40 and np.all(lifts>=2) and moved
    return dict(support_gate_wait_frames=gate_frames, support_gate_short_steps=short_steps, passed=bool(passed), failure=reason, failure_time_s=age+(substep+1)*m.opt.timestep if reason else None,
                failure_joints=failure_joints, failure_up_body=failure_up, failure_detail=failure_detail,
                seed=seed, lifts=lifts.tolist(), signed_progress=signed, fixed_exit_s=fixed*dt,
                fixed_entry_s=entry*dt,
                travel_m=distance, forward_progress_m=forward_progress, signed_elevation_m=elevation,
                motion_base_height_m=float(np.median(motion_base_heights)) if motion_base_heights else None,
                nonwheel_frames=contact_frames,
                min_distal_calf_surface_m=[float(gap) if np.isfinite(gap) else None for gap in min_distal_calf],
                distal_calf_warning_s=distal_calf_warning_s,
                min_calf_hull_vertex_gap_m=[float(gap) if np.isfinite(gap) else None for gap in min_calf_hull],
                calf_hull_warning_s=calf_hull_warning_s,
                min_calf_geom_distance_m=[float(distance) if np.isfinite(distance) else None for distance in min_calf_geom_distance],
                calf_geom_warning_s=calf_geom_warning_s,
                min_knee_surface_m=min_surface, max_effort_fraction_of_guard=max_effort)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--seed",type=int,default=8296)
    p.add_argument("--trials",type=int,default=2)
    p.add_argument("--sequence-period",type=float,default=12.)
    p.add_argument("--sequence-stride",type=float,default=.10)
    p.add_argument("--packet-loss",type=float,default=0.)
    p.add_argument("--native-pd",action="store_true")
    p.add_argument("--calf-mesh-diagnostic", action="store_true")
    p.add_argument("--leg-pd", nargs=2, type=float, default=(85., 3.2), metavar=("KP", "KD"))
    p.add_argument("--leg-pd-group", choices=("all", "hip", "thigh", "calf"), default="all")
    p.add_argument("--target-margin",type=float,default=0.)
    p.add_argument("--sequence-attitude",action="store_true")
    p.add_argument("--predictive-guard",action="store_true")
    p.add_argument("--rear-stance-predictor",action="store_true")
    p.add_argument("--front-lift-m",type=float,default=0.)
    p.add_argument("--body-raise-m", type=float, default=0.)
    p.add_argument("--symmetric-front-support", action="store_true")
    p.add_argument("--front-reach-scale",type=float,default=1.)
    p.add_argument("--down-approach",type=float,default=1.20)
    p.add_argument("--lateral-shift-scale",type=float,default=1.)
    p.add_argument("--case",action="append",choices=(*FLAT,"up","down"))
    p.add_argument("--damp-lateral-shift",action="store_true")
    p.add_argument("--go2-transfer",action="store_true")
    p.add_argument("--go2-foot-retarget",action="store_true")
    p.add_argument("--wheel-teacher-replay",action="store_true")
    p.add_argument("--teacher-rate",type=float,choices=(.5,1.),default=1.)
    a = p.parse_args()
    if not 0. <= a.body_raise_m <= .04:
        p.error("--body-raise-m must be within 0–0.04 m")
    if a.body_raise_m and (a.go2_transfer or a.go2_foot_retarget or a.wheel_teacher_replay):
        p.error("--body-raise-m requires the guarded stepping controller")
    if not all(np.isfinite(gain) and gain > 0 for gain in a.leg_pd):
        p.error("--leg-pd requires two finite positive gains")
    if a.teacher_rate != 1. and not a.wheel_teacher_replay:
        p.error("Teacher rate requires wheel teacher replay")
    if a.wheel_teacher_replay and (a.go2_transfer or a.case not in (["up"],["down"])):
        p.error("Teacher replay requires exactly one stair direction and an NPZ trace")
    if a.go2_foot_retarget and not a.go2_transfer:
        p.error("Go2 foot retargeting requires a Go2 transfer checkpoint")
    if a.damp_lateral_shift and not a.sequence_attitude:
        p.error("Lateral shift damping requires sequence attitude feedback")
    if not .60 <= a.front_reach_scale <= 1.:
        p.error("Front forward reach scale must be .60–1")
    if not 0. <= a.front_lift_m <= .05:
        p.error("Front lift addition must be 0–.05 m")
    if a.rear_stance_predictor and not a.predictive_guard:
        p.error("Rear stance prediction requires the predictive guard")
    if not .35 <= a.lateral_shift_scale <= 1.:
        p.error("Lateral shift scale must be .35–1")
    if not .45 <= a.down_approach <= 1.35:
        p.error("Down approach must be .45–1.35 m from the top edge")
    if not 0. <= a.target_margin <= .12:
        p.error("Target margin must be 0–.12 rad")
    if not 0. <= a.packet_loss <= .10:
        p.error("Packet loss must be 0–.10")
    if not 3. <= a.sequence_period <= 12. or not .10 <= a.sequence_stride <= .65:
        p.error("Sequence period must be 3–12 s and stride gain .10–.65 s")
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if a.wheel_teacher_replay:
        with np.load(a.checkpoint,allow_pickle=False) as data:
            a.teacher_data = {name:data[name] for name in ("joint_names","as2w_joint_target_rad","time_s")}
        actor = None
    elif a.go2_transfer:
        actor = torch.jit.load(str(a.checkpoint),map_location="cpu").eval()
    else:
        state = torch.load(a.checkpoint,map_location="cpu",weights_only=True)
        actor = actor_from_state(state["actor_state_dict"],2.)
    replay,results = [],{}
    cases = {**FLAT,"up":(.16,0.,0.),"down":(.16,0.,0.)}
    if a.case:
        cases = {name:cases[name] for name in a.case}
    for name, command in cases.items():
        cmd = np.array(command,dtype=np.float32)
        rows = [trial(a,actor,name,cmd,a.seed+i,replay) for i in range(a.trials)]
        results[name] = dict(passed=sum(r["passed"] for r in rows), total=len(rows),rows=rows)
        print(name,results[name]["passed"],len(rows),flush=True)
    np.save(a.output/"policy-observations.npy",np.asarray(replay))
    (a.output/"contract.json").write_text(json.dumps(dict(
        **checkpoint_record(a.checkpoint), support_gate_force_n=8., support_gate_stride_reduction=.35, support_gate_wait_cap_s=.30, wheel_teacher_replay=a.wheel_teacher_replay, teacher_rate=a.teacher_rate, go2_transfer=a.go2_transfer, go2_foot_retarget=a.go2_foot_retarget, trials_per_case=a.trials, seed=a.seed,
        motion_substep_audit=True, min_knee_center_m=.045,
        rear_stance_predictor=a.rear_stance_predictor, front_lift_m=a.front_lift_m, front_reach_scale=a.front_reach_scale,
        period_s=a.sequence_period, stride_gain=a.sequence_stride, observation_packet_loss=a.packet_loss,
        calf_mesh_diagnostic=a.calf_mesh_diagnostic,
        body_raise_m=a.body_raise_m,
        symmetric_front_support=a.symmetric_front_support,
        leg_pd_group=a.leg_pd_group,leg_pd=a.leg_pd,native_pd=a.native_pd,target_margin_rad=a.target_margin,sequence_attitude=a.sequence_attitude,predictive_guard=a.predictive_guard,
        down_approach_m=a.down_approach, damp_lateral_shift=a.damp_lateral_shift,lateral_shift_scale=a.lateral_shift_scale,cases=list(cases)),indent=2)+"\n")
    (a.output/"mujoco-stepping-diagnostic.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__":
    main()
