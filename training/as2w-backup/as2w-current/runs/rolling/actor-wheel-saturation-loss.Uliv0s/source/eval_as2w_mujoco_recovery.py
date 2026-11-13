"""Independent MuJoCo recovery diagnostics with the shared AS2-W controller."""
import argparse
from collections import deque
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record

import mujoco
import numpy as np
import torch

from as2w_knee_envelope import KNEE_ENVELOPE_CORNERS, MIN_KNEE_SURFACE_M
from as2w_predictive_joint_guard import guarded_target
from as2w_recovery_control import (latch_handoff, mirror_legs, mirror_observation,
                                    recovery_roll_brace, staged_target, stand_integral)
from export_as2w_candidate_onnx import actor_from_state
from train_stair_low_speed_course import OmniEnvironment


def contacts(env, *, strict=False, snapshot=None):
    loads = np.zeros(4)
    body = False
    wrench = np.zeros(6)
    for index, contact in enumerate(env.data.contact):
        if env.ground not in contact.geom:
            continue
        other = contact.geom[1] if contact.geom[0] == env.ground else contact.geom[0]
        link = env.model.geom_bodyid[other]
        mujoco.mj_contactForce(env.model, env.data, index, wrench)
        if link in env.wheel_index:
            loads[env.wheel_index[link]] += abs(wrench[0] * contact.frame[2])
        elif wrench[0] > 4. or (strict and contact.dist <= 0.):
            body = True
            if snapshot is not None:
                snapshot.setdefault("nonwheel_contact_links", []).append(env.model.body(link).name)
    if snapshot is not None:
        snapshot["wheel_vertical_load_n"] = loads.tolist()
    return bool(np.all(loads > 2.)), body


@torch.inference_mode()
def trial(env, actor, start, seed, observations, low_handoff, roll_brace, packet_loss, predictive_guard, anchor_wheels, fast_side_entry, observe_after_push=False):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, options={"scene_seed": seed, "target": (0.,0.,0.)})
    m, d = env.model, env.data
    defaults = env.defaults[:12]
    lo = defaults + .90*(env.leg_low-defaults)
    hi = defaults + .90*(env.leg_high-defaults)
    effort = .90*env.torque_high
    m.body_mass[env.base] *= rng.uniform(.95,1.05)
    m.body_ipos[env.base,:2] += rng.uniform(-.01,.01,2)
    for link in env.wheel_index:
        m.geom_friction[m.geom_bodyid == link,0] *= rng.uniform(.85,1.15)
    mujoco.mj_setConst(m,d)
    d.qpos[2] = start["height_m"]
    d.qpos[3:7] = start["root_wxyz"]
    d.qpos[env.qa[:12]] = np.clip(start["leg_q"],lo+.12,hi-.12)
    d.qpos[env.qa[12:]] = 0.
    d.qvel[:] = 0.
    d.qvel[env.va[:12]] = rng.uniform(-.10,.10,12)
    mujoco.mj_forward(m,d)
    knees = [m.body(f"{leg}_calf").id for leg in ("FL","FR","RL","RR")]
    initial = torch.tensor(((d.qpos[env.qa[:12]]-defaults)/.35)[None],dtype=torch.float32)
    fast_entry = torch.tensor([abs(d.xmat[env.base,7]) > .42]) if fast_side_entry else None
    bias = torch.zeros_like(initial)
    frames = torch.zeros(1,dtype=torch.int64)
    steady = frames.clone()
    previous_action = np.zeros(12,dtype=np.float32)
    old_sensor = None
    delivered = None
    missed = False
    obs_lag = rng.integers(2)
    actuator_lag = round(.005*rng.integers(5)/m.opt.timestep)
    history = np.repeat(d.qpos[env.qa[:12]][None],actuator_lag+1,axis=0)
    cursor = 0
    wheel_target = np.zeros(4)
    wheel_anchored = False
    kp = np.r_[[85.]*12,[25.]*4]*rng.uniform(.90,1.10,16)
    kd = np.r_[[3.2]*12,[4.]*4]*rng.uniform(.90,1.10,16)
    dt = env.dt
    push_tick = round(rng.uniform(3.5,8.)/dt)
    twist = np.empty(6)
    noise = np.r_[[.015]*3,[.006]*3,[.004]*12,[.04]*12]
    min_surface = float("inf")
    max_effort_ratio = 0.
    window = longest = 0
    fixed_gate_names = ("height", "upright", "wheel_support", "body_clear",
                        "joint_error", "linear_speed", "angular_speed")
    fixed_gate_history = deque(maxlen=150)
    final_speed_history = deque(maxlen=150)
    push_times_s = []
    regained = body_after = initial_body = False
    survived = True
    first_failure = None
    failure_state = None
    wheel_bodies = sorted(env.wheel_index, key=env.wheel_index.get)
    joint_ids = np.asarray(env.va)[:12]

    def audit_state(time):
        nonlocal regained, body_after, min_surface, max_effort_ratio, first_failure, failure_state
        q = d.qpos[env.qa[:12]]
        h, up = d.qpos[2], d.xmat[env.base,8]
        regained |= h >= .31 and up >= .95
        invalid_joints = np.flatnonzero((q < lo) | (q > hi))
        effort_ratio = float(np.max(np.abs(d.actuator_force[env.motors])/effort))
        max_effort_ratio = max(max_effort_ratio,effort_ratio)
        failure = []
        if len(invalid_joints):
            failure.append("joint_angle")
        if effort_ratio > 1.00001:
            failure.append("effort")
        if up < -.10:
            failure.append("overturned")
        if regained or (h > .29 and up > .85):
            _, body = contacts(env,strict=True)
            body_after |= regained and body
            if body:
                failure.append("nonwheel_contact")
            rot = d.xmat[knees].reshape(4,3,3)
            points = d.xpos[knees,None,:]+np.einsum("kij,cj->kci",rot,KNEE_ENVELOPE_CORNERS)
            surface = float(points[...,2].min())
            min_surface = min(min_surface,surface)
            if surface < MIN_KNEE_SURFACE_M or np.any(d.xpos[knees,2] < .045):
                failure.append("knee_surface")
            yaw = np.arctan2(d.xmat[env.base,3],d.xmat[env.base,0])
            lateral = d.xpos[wheel_bodies,:2] @ np.array((-np.sin(yaw),np.cos(yaw)))
            rear = q[6:8]
            if np.any((rear < .18) | (rear > 1.40)) or np.any(lateral[[0,2]]-lateral[[1,3]]<.06):
                failure.append("stance")
        if failure:
            first_failure = failure
            failure_state = dict(time_s=time, base_height_m=float(h), up_z=float(up),
                                 angular_velocity_rad_s=d.qvel[3:6].tolist(),
                                 joint=[env.contract["joint_names"][i] for i in invalid_joints],
                                 angle_rad=q[invalid_joints].tolist(),
                                 velocity_rad_s=d.qvel[joint_ids[invalid_joints]].tolist())
            contacts(env, strict=True, snapshot=failure_state)
        return not failure

    for tick in range(700 if observe_after_push else 550):
        rotation = d.xmat[env.base].reshape(3,3)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,env.base,twist,1)
        q = d.qpos[env.qa[:12]]
        h, up = d.qpos[2], rotation[2,2]
        sensed = np.r_[twist[:3], -rotation[2], q-defaults, d.qvel[env.va[:12]]]
        sensed += rng.normal(size=30)*noise
        sensor = old_sensor if obs_lag and old_sensor is not None else sensed
        old_sensor = sensed
        if packet_loss:
            missed = rng.random()<packet_loss and not missed and delivered is not None
            if missed:
                sensor = delivered
            else:
                delivered = sensor
        obs = np.r_[sensor,previous_action].astype(np.float32)
        if tick % 10 == 0:
            observations.append(obs)
        action = actor(torch.from_numpy(obs[None]))
        wheels, body = contacts(env)
        initial_body |= tick < 12 and body
        established = h >= .31 and up >= .95
        speed, spin = np.linalg.norm(twist[3:]), np.linalg.norm(twist[:3])
        error = float(np.max(np.abs(q-defaults)))
        fixed_gates = (h >= .31, up >= .95, wheels, not body, error < .18, speed < .06, spin < .12)
        fixed_gate_history.append(fixed_gates)
        final_speed_history.append((tick * dt, float(speed)))
        fixed = all(fixed_gates)
        window = window+1 if fixed else 0
        longest = max(longest,window)
        stable = h>(.27 if low_handoff else .31) and up>.95 and wheels and speed<.08 and spin<.18
        steady = steady+1 if stable else torch.zeros_like(steady)
        height, upright = torch.tensor([h]), torch.tensor([up])
        frames, blend = latch_handoff(steady,frames,height,upright,dt,min_height=.25 if low_handoff else .28)
        if anchor_wheels:
            if frames.item() == 0:
                wheel_target.fill(0.)
                wheel_anchored = False
            elif not wheel_anchored and blend.item() >= .80:
                wheel_target[:] = d.qpos[env.qa[12:]]
                wheel_anchored = True
        bias = stand_integral(bias,torch.from_numpy(sensor[None,6:18]),blend,height,upright,dt,
                             min_height=.25 if low_handoff else .29)
        age = torch.tensor([tick*dt])
        target = staged_target(initial,action,age,fast_entry=fast_entry)
        if roll_brace:
            target = target + recovery_roll_brace(torch.tensor([sensor[4]]),age)
        output = target*(1.-blend[:,None])+blend[:,None]*bias/.35
        previous_action = output[0].to(torch.float32).numpy()
        command = np.clip(defaults+.35*previous_action,lo+.06,hi-.06)
        if tick == push_tick and tick < 550:
            push_times_s.append(tick * dt)
            d.qvel[:2] += rng.uniform((-.08,-.06),(.08,.06))
            d.qvel[5] += rng.uniform(-.08,.08)
            push_tick += round(rng.uniform(3.5,8.)/dt)
        for substep in range(env.substeps):
            history[cursor] = command
            delayed = history[(cursor-actuator_lag)%len(history)]
            cursor = (cursor+1)%len(history)
            if predictive_guard:
                delayed = guarded_target(delayed,d.qpos[env.qa[:12]],d.qvel[env.va[:12]],lo,hi,margin=.06)
            tau = kp*(np.r_[delayed,wheel_target]-d.qpos[env.qa])-kd*d.qvel[env.va]
            applied = np.clip(tau,-effort,effort)
            d.ctrl[env.motors] = applied
            mujoco.mj_step(m,d)
            survived = audit_state(tick*dt + (substep+1)*m.opt.timestep)
            if not survived:
                break
        if not survived:
            break
    passed = survived and window>=100 and not body_after and min_surface>=MIN_KNEE_SURFACE_M
    return dict(passed=bool(passed), survived=survived, first_failure=first_failure,
                failure_state=failure_state,
                initial_body_contact=bool(initial_body),
                last_height_m=float(h), last_joint_error_rad=error, final_wheel_support=wheels,
                fixed_final_window_s=window*dt, longest_fixed_window_s=longest*dt,
                final_150_frames_failed=dict(zip(fixed_gate_names,
                    np.count_nonzero(~np.asarray(fixed_gate_history, dtype=bool), axis=0).tolist())),
                fixed_gate_frames_observed=len(fixed_gate_history),
                push_times_s=push_times_s,
                final_speed_gate_events=[{"time_s": time, "speed_m_s": speed}
                                         for time, speed in final_speed_history if speed >= .06],
                min_knee_surface_m=min_surface if np.isfinite(min_surface) else None,
                nonwheel_after_upright=bool(body_after), max_effort_fraction_of_guard=max_effort_ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("pose_bank",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--seed",type=int,default=7294)
    parser.add_argument("--low-support-handoff", action="store_true")
    parser.add_argument("--mirror-policy", action="store_true")
    parser.add_argument("--roll-brace", action="store_true")
    parser.add_argument("--paired-tilt", action="store_true")
    parser.add_argument("--side-pose-bank",type=Path)
    parser.add_argument("--packet-loss",type=float,default=0.)
    parser.add_argument("--predictive-guard",action="store_true")
    parser.add_argument("--anchor-wheels-on-handoff",action="store_true")
    parser.add_argument("--observe-after-push", action="store_true",
                        help="Diagnostic: observe 3 extra seconds after the original push schedule")
    parser.add_argument("--fast-side-entry",action="store_true")
    parser.add_argument("--family",action="append",choices=("prone","left","right","near_stand"))
    args = parser.parse_args()
    if not 0. <= args.packet_loss <= .10:
        parser.error("Packet loss must be between 0 and .10")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    base_actor = actor_from_state(state["actor_state_dict"],2.)

    def symmetric_actor(obs):
        return .5*(base_actor(obs) + mirror_legs(base_actor(mirror_observation(obs))))

    actor = symmetric_actor if args.mirror_policy else base_actor
    poses = json.loads(args.pose_bank.read_text())
    env = OmniEnvironment(args.assets,1,stop_training=True)
    rng = np.random.default_rng(args.seed)
    from prepare_as2w_supported_getup import quaternion
    poses["near_stand"] = [
        dict(height_m=.39,root_wxyz=quaternion(*rng.uniform(-.10,.10,2)),
             leg_q=env.defaults[:12]+rng.uniform(-.09,.09,12)) for _ in range(8)]
    for family, sign in (("left", -1.), ("right", 1.)):
        poses[family] = []
        for _ in range(8):
            roll = sign*rng.uniform(.38,.96)
            pitch = rng.uniform(-.10,.10)
            q = env.defaults[:12]+rng.uniform(-.09,.09,12)
            q[4:8] += rng.uniform(.34,.46)
            q[8:12] -= rng.uniform(.58,.72)
            height = .39-.125+.15*abs(np.sin(roll))+.10*abs(np.sin(pitch))
            poses[family].append(dict(height_m=height,root_wxyz=quaternion(roll,pitch),leg_q=q))
    if args.side_pose_bank:
        sides = json.loads(args.side_pose_bank.read_text())
        poses.update({family:sides[family] for family in ("left","right")})
    if args.paired_tilt:
        for left, right in zip(poses["left"],poses["right"]):
            w,x,y,z = left["root_wxyz"]
            right["root_wxyz"] = np.array((w,-x,y,-z))
            right["leg_q"] = mirror_legs(torch.as_tensor(left["leg_q"])[None])[0].numpy()
            right["height_m"] = left["height_m"]
    pair_seeds = rng.integers(10**8,size=8)
    result = {}
    observations = []
    for family, bank in poses.items():
        paired = args.paired_tilt and family in ("left", "right")
        samples = range(8) if paired else rng.choice(len(bank),min(8,len(bank)),replace=False)
        seeds = pair_seeds if paired else rng.integers(10**8,size=len(samples))
        if args.family and family not in args.family:
            continue
        rows = [trial(env,actor,bank[i],int(seed),observations,
                      args.low_support_handoff,args.roll_brace,args.packet_loss,args.predictive_guard,args.anchor_wheels_on_handoff,args.fast_side_entry,args.observe_after_push) for i,seed in zip(samples,seeds)]
        result[family] = dict(total=len(rows),passed=sum(r["passed"] for r in rows),
                              actual_start_support=sum(r["initial_body_contact"] for r in rows),rows=rows)
    np.save(args.output/"policy-observations.npy",np.array(observations))
    (args.output/"contract.json").write_text(json.dumps({
        **checkpoint_record(args.checkpoint), "seed": args.seed,
        "low_support_handoff": args.low_support_handoff,
        "mirror_policy": args.mirror_policy, "paired_tilt": args.paired_tilt,
        "roll_brace": args.roll_brace,
        "observation_packet_loss": args.packet_loss,
        "side_pose_bank": str(args.side_pose_bank) if args.side_pose_bank else None,
        "policy_step_s": env.dt, "physics_step_s": env.model.opt.timestep,
        "motion_substep_audit": True, "min_knee_center_m": .045,
        "predictive_guard": args.predictive_guard,
        "anchor_wheels_on_handoff": args.anchor_wheels_on_handoff,
        "observe_after_push_diagnostic": args.observe_after_push,
        "fast_side_entry": args.fast_side_entry, "evaluated_families": list(result),
        "actuator_delay_ms": [0, 5, 10, 15, 20], "observation_delay_ms": [0, 20],
        "families": {"prone": "heldout torso support", "left": "random noninverted tilt",
                     "right": "random noninverted tilt", "near_stand": "random standing offset"},
    },indent=2)+"\n")
    (args.output/"mujoco-recovery-diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:(v["passed"],v["total"],v["actual_start_support"]) for k,v in result.items()},flush=True)


if __name__ == "__main__":
    main()
