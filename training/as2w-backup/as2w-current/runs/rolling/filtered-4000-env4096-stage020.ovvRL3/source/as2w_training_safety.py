"""AS2-W simulation safety contract shared across the skill runners."""
import xml.etree.ElementTree as ET

import torch
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg

from as2w_isaac_flat_env_all import JOINTS

DEFAULT = (0.,)*4 + (.8,)*4 + (-1.6,)*4
FRACTION = .90


def contract_limits(urdf):
    joints = {j.attrib["name"]: j.find("limit").attrib
              for j in ET.parse(urdf).getroot().findall("joint") if j.find("limit") is not None}
    angle = {}
    for name, stand in zip(JOINTS[:12], DEFAULT):
        lower = float(joints[name]["lower"])
        upper = float(joints[name]["upper"])
        if not lower < stand < upper:
            raise ValueError(f"Invalid standing reference for {name}")
        angle[name] = (stand + FRACTION*(lower-stand), stand + FRACTION*(upper-stand))
    torque = {name: FRACTION*float(joints[name]["effort"]) for name in JOINTS}
    wheel = {name: float(joints[name]["velocity"]) for name in JOINTS[12:]}
    return dict(angle_rad=angle, effort_nm=torque, wheel_rad_s=wheel,
                fraction=FRACTION, source="simulation_urdf_not_hardware_calibration")


def apply_safety(cfg, urdf, *, floor_guard=True, perturb=True, surface_guard=False):
    limits = contract_limits(urdf)
    cfg.as2w_safety = limits
    cfg.as2w_floor_guard = floor_guard or cfg.scene.terrain.terrain_type == "plane"
    cfg.as2w_min_knee_clearance_m = .045
    cfg.actions.joint_pos.clip = limits["angle_rad"]
    if cfg.actions.wheel_vel is not None:
        cfg.actions.wheel_vel.clip = {k: (-v, v) for k, v in limits["wheel_rad_s"].items()}
    for actuator in cfg.scene.robot.actuators.values():
        if not isinstance(actuator.effort_limit_sim, dict):
            raise TypeError("Expected per-axis simulated actuator effort limits")
        actuator.effort_limit_sim = {k: limits["effort_nm"][k] for k in actuator.effort_limit_sim}
    cfg.terminations.as2w_angle_violation = TerminationTermCfg(func=angle_violation)
    cfg.terminations.as2w_stance_violation = TerminationTermCfg(func=stance_violation)
    cfg.terminations.as2w_nonwheel_contact = TerminationTermCfg(func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"^(?!.*_foot$).*"),
                "threshold": 4.})
    if surface_guard:
        from as2w_knee_envelope import MIN_KNEE_SURFACE_M
        limits["min_knee_surface_m"] = MIN_KNEE_SURFACE_M
        cfg.terminations.as2w_knee_surface = TerminationTermCfg(func=knee_surface_violation)
    if perturb:
        cfg.events.bounded_push = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=(3.5, 8.), params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-.08,.08), "y": (-.06,.06), "yaw": (-.08,.08)},
            })
    return limits


def angle_violation(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_guard_limits"):
        env._guard_limits = torch.tensor(list(env.cfg.as2w_safety["angle_rad"].values()),
                                         device=env.device)
    q = robot.data.joint_pos.torch[:, env.as2w_joint_ids[:12]]
    lo, hi = env._guard_limits.unbind(dim=1)
    invalid = (q < lo) | (q > hi)
    if getattr(env.cfg, "as2w_guard_diagnostics", False) and not hasattr(env, "_angle_report"):
        ids = torch.nonzero(invalid, as_tuple=False)[:8]
        if len(ids):
            env._angle_report = True
            print("AS2W_ANGLE_SAMPLE", [(int(env.episode_length_buf[i]), JOINTS[j],
                  round(float(q[i,j]),3), round(float(lo[j]),3), round(float(hi[j]),3))
                  for i,j in ids.tolist()], flush=True)
    return invalid.any(dim=1)


def stance_geometry(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_guard_body"):
        env._guard_body = [robot.body_names.index(j) for j in
                           (*JOINTS[12:], *JOINTS[8:12])]
    position = robot.data.body_pos_w.torch[:, env._guard_body]
    x, y, z, w = robot.data.root_quat_w.torch.unbind(dim=1)
    heading = torch.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    pairs = position[:, [0, 2]] - position[:, [1, 3]]
    width = -heading.sin()[:, None]*pairs[..., 0] + heading.cos()[:, None]*pairs[..., 1]
    rear_thigh = robot.data.joint_pos.torch[:, env.as2w_joint_ids[6:8]]
    return width, rear_thigh, knee_clearance(env, position[:, 4:])


def stepping_stance_margin_cost(env):
    width, rear_thigh, _ = stance_geometry(env)
    width_shortfall = ((.10 - width) / .04).clamp_min(0.)
    thigh_shortfall = ((.28 - rear_thigh) / .10).clamp_min(0.)
    thigh_excess = ((rear_thigh - 1.30) / .10).clamp_min(0.)
    return width_shortfall.square().mean(dim=1) + (
        thigh_shortfall.square() + thigh_excess.square()).mean(dim=1)


def stance_violation(env, *, components=False):
    width, rear_thigh, knee_z = stance_geometry(env)
    crossed = (width < .06).any(dim=1)
    rear_overextended = ((rear_thigh < .18) | (rear_thigh > 1.40)).any(dim=1)
    low_knee = (knee_z < env.cfg.as2w_min_knee_clearance_m).any(dim=1)
    if getattr(env.cfg, "as2w_guard_diagnostics", False):
        env._guard_stance_detail = torch.stack((crossed, rear_overextended, low_knee), dim=1)
        env._guard_stance_metrics = torch.cat((width, rear_thigh, knee_z), dim=1)
    if not hasattr(env, "_guard_report"):
        env._guard_report = True
        print("AS2W_GUARD_GEOMETRY", {
            "width": width[:4].tolist(), "rear_thigh": rear_thigh[:4].tolist(),
            "knee_clearance": knee_z[:4].tolist(), "crossed": int(crossed.sum()),
            "rear_overextended": int(rear_overextended.sum()), "low_knee": int(low_knee.sum()),
        }, flush=True)
    if components:
        return crossed, rear_overextended, low_knee
    return crossed | rear_overextended | low_knee


def knee_clearance(env, knees, *, padded=True):
    if env.cfg.as2w_floor_guard:
        return knees[..., 2] - env.scene.env_origins[:, None, 2]
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    if not hasattr(env, "_guard_terrain"):
        import numpy as np
        from isaaclab import sim
        from pxr import Usd, UsdGeom

        terrain_path = env.scene.terrain.terrain_prim_paths[0]
        mesh = UsdGeom.Mesh(sim.get_first_matching_child_prim(
            terrain_path, lambda p: p.GetTypeName() == "Mesh"))
        transform = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())).T
        vertices = np.asarray(mesh.GetPointsAttr().Get()) @ transform[:3,:3].T + transform[:3,3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        env._guard_terrain = convert_to_warp_mesh(vertices, faces, device=env.device)
    offsets = knees.new_tensor(((0.,0.,1.),(.02,0.,1.),(-.02,0.,1.),(0.,.02,1.),(0.,-.02,1.)))
    if not padded:
        offsets = offsets[:1]
    starts = knees[:,:,None,:] + offsets
    rays = torch.zeros_like(starts)
    rays[..., 2] = -1.
    hits, _, _, _ = raycast_mesh(starts, rays, env._guard_terrain, max_dist=3.)
    return knees[..., 2] - hits[..., 2].amax(dim=2)


def knee_surface_clearance(env):
    """Minimum clearance of the four oriented knee envelopes to local terrain."""
    from isaaclab.utils.math import quat_apply
    from as2w_knee_envelope import KNEE_ENVELOPE_CORNERS

    robot = env.scene["robot"]
    if not hasattr(env, "_knee_surface_ids"):
        env._knee_surface_ids = [robot.body_names.index(j) for j in JOINTS[8:12]]
    pos = robot.data.body_pos_w.torch[:,env._knee_surface_ids]
    quat = robot.data.body_quat_w.torch[:,env._knee_surface_ids]
    corners = pos.new_tensor(KNEE_ENVELOPE_CORNERS)
    points = pos[:,:,None,:] + quat_apply(quat[:,:,None,:],corners)
    gaps = knee_clearance(env,points.flatten(1,2))
    return gaps.reshape(env.num_envs,4,len(KNEE_ENVELOPE_CORNERS)).amin(dim=2)


def rolling_knee_margin_cost(env, margin_m):
    """Penalize loss of clearance before the hard contact guard trips."""
    shortfall = (margin_m - knee_surface_clearance(env)).clamp_min(0.0)
    return (shortfall / margin_m).square().mean(dim=1)


def knee_surface_violation(env):
    from as2w_knee_envelope import MIN_KNEE_SURFACE_M

    return (knee_surface_clearance(env) < MIN_KNEE_SURFACE_M).any(dim=1)


MOTION_SUBSTEP_TERMS = ("joint_angle", "stance", "knee_surface", "nonwheel_contact", "effort", "overturned",
                        "rolling_wheel_unload", "stance_crossed_wheels",
                        "stance_rear_thigh", "stance_knee_center")


def motion_substep_violation(env):
    if not hasattr(env, "_motion_substep_bad"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._motion_substep_last.copy_(env._motion_substep_bad)
    result = env._motion_substep_bad.any(dim=1)
    env._motion_substep_bad.zero_()
    return result


def configure_motion_substep_audit(cfg):
    cfg.scene.contact_forces.update_period = 0.
    cfg.terminations.as2w_motion_substep = TerminationTermCfg(func=motion_substep_violation)


def install_motion_substep_audit(env, *, recovery_phase=None, rolling_active=None):
    """Latch a failure from any PhysX substep until the termination manager reads it."""
    if env._physics_handles_decimation:
        raise RuntimeError("AS2-W motion audit requires individual physics steps")
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    nonwheel = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_foot")]
    if not nonwheel:
        raise RuntimeError("AS2-W motion audit requires non-wheel contact sensors")
    effort = torch.tensor(
        [env.cfg.as2w_safety["effort_nm"][name] for name in JOINTS], device=env.device
    )
    env._motion_substep_bad = torch.zeros(
        (env.num_envs, len(MOTION_SUBSTEP_TERMS)), dtype=torch.bool, device=env.device
    )
    env._motion_substep_last = torch.zeros_like(env._motion_substep_bad)
    env._motion_first_contact_body = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    nonwheel_ids = torch.tensor(nonwheel, device=env.device)
    update_scene = env.scene.update
    if rolling_active is not None:
        wheels = [sensor.body_names.index(name) for name in JOINTS[12:]]
        unloaded_steps = torch.zeros(env.num_envs,dtype=torch.long,device=env.device)

    def update_and_audit(dt):
        if abs(dt - env.physics_dt) > 1.e-8:
            raise RuntimeError("AS2-W motion audit missed a physics substep")
        update_scene(dt)
        forces = sensor.data.net_forces_w.torch
        contact, contact_index = forces[:, nonwheel].norm(dim=-1).max(dim=1)
        torque = robot.data.applied_torque.torch[:, env.as2w_joint_ids].abs()
        crossed, rear, low_knee = stance_violation(env,components=True)
        stance = crossed | rear | low_knee
        knee = knee_surface_violation(env)
        body = contact > 1.e-3
        inverted = torch.zeros_like(stance)
        if recovery_phase is not None:
            standing, inverted = recovery_phase(env)
            stance &= standing
            crossed &= standing
            rear &= standing
            low_knee &= standing
            knee &= standing
            body &= standing
        first_contact = env._motion_first_contact_body
        first_contact.copy_(torch.where(body & (first_contact < 0),
                                        nonwheel_ids[contact_index], first_contact))
        unload = torch.zeros_like(stance)
        if rolling_active is not None:
            unsupported = (forces[:,wheels,2] < 2.).any(dim=1)
            unloaded_steps.copy_(torch.where(rolling_active & unsupported,unloaded_steps+1,0))
            unload = unloaded_steps * dt > .060001
        env._motion_substep_bad |= torch.stack((
            angle_violation(env), stance, knee, body, (torque > effort * 1.00001).any(dim=1), inverted, unload,
            crossed, rear, low_knee
        ),dim=1)

    env.scene.update = update_and_audit
    return {"physics_dt_s": env.physics_dt, "nonwheel_force_tolerance_n": .001}
