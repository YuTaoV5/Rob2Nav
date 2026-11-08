"""Paired flat-terrain MuJoCo screening of AS2-W recovery checkpoints."""
import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import mujoco as mj
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "fall-baseline"
POSES = {"back": np.pi, "left": np.pi / 2, "right": -np.pi / 2, "belly": 0.}


class DynamicsStress(NamedTuple):
    friction: float
    mass_scale: np.ndarray
    gain_scale: np.ndarray
    push_time: float
    push_delta_xy: np.ndarray


def sample_dynamics_stress(seed, body_count, joint_count):
    rng = np.random.default_rng([seed, 804])
    return DynamicsStress(
        rng.uniform(.45, .95), rng.uniform(.9, 1.1, body_count),
        rng.uniform(.85, 1.15, (2, joint_count)), rng.uniform(2.8, 4.5),
        rng.uniform(-.15, .15, 2),
    )


def evaluate(model, contract, state, seed, roll, capture=None, limits=None, action_clip=None,
             wheel_scale=6., reference_step=None, full_leg_range=False, drive_speed_limit=False,
             dynamics_stress=None, wheel_angle_period=None, terrain=None, stair_gyro_heading_gain=0.,
             stair_heading_feedback=False, native_policy=None):
    rng = np.random.default_rng(seed)
    data = mj.MjData(model)
    joints = [model.joint(name) for name in contract["joint_names"]]
    qa = np.array([j.qposadr[0] for j in joints])
    va = np.array([j.dofadr[0] for j in joints])
    names = [name for term in contract["action_terms"] for name in term["joints"]]
    if names != contract["joint_names"]:
        raise ValueError("Action order differs from the observation joint order")
    motors = np.array([model.actuator(name).id for name in names])
    wheel_bodies = {model.jnt_bodyid[j.id] for j in joints[12:]}
    default = np.array(contract["default_joint_pos"])
    if full_leg_range:
        leg_upper_span = limits[:12, 1] - default[:12]
        leg_lower_span = default[:12] - limits[:12, 0]
    ground = model.geom("ground").id
    stairs = terrain == "stairs"
    stair_geoms = {model.geom(f"stair_{level}").id: level for level in range(1, 6)} if stairs else {}
    surfaces = {ground, *stair_geoms}
    ground_friction = rng.uniform(.55, .85)
    if dynamics_stress is not None:
        ground_friction = dynamics_stress.friction
        model.body_mass[1:] *= dynamics_stress.mass_scale
        model.body_inertia[1:] *= dynamics_stress.mass_scale[:, None]
        mj.mj_setConst(model, data)
    model.geom_friction[list(surfaces), 0] = ground_friction
    if terrain:
        ground_mask = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        down = np.array([0., 0., -1.])
        hit_geom = np.empty(1, dtype=np.int32)

        def surface_height(x, y):
            distance = mj.mj_ray(model, data, np.array([x, y, 5.]), down, ground_mask,
                                 True, -1, hit_geom)
            return 5. - distance if distance >= 0. and hit_geom[0] in surfaces else np.nan

        mj.mj_forward(model, data)
    data.qpos[2] = .37 if stairs else 1.1
    if terrain:
        data.qpos[2] += surface_height(0., 0.)
    if stairs:
        data.qpos[:2] = rng.uniform((-.12, -.15), (.12, .15))
        yaw = rng.uniform(-.08, .08)
        data.qpos[3:7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        data.qpos[qa] = default + rng.uniform(-.12, .12, size=16)
    else:
        angle = roll + rng.uniform(-.12, .12)
        data.qpos[3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0., 0.]
        data.qpos[qa] = default + rng.uniform(-.55, .55, size=16)
    for j in joints[:12]:
        data.qpos[j.qposadr[0]] = np.clip(data.qpos[j.qposadr[0]], *j.range)
    mj.mj_forward(model, data)
    dt = contract["control_dt"]
    substeps = round(dt / model.opt.timestep)
    settle = 0. if stairs else contract["settle_seconds"]
    recovery = 9. if stairs else contract["recovery_seconds"]
    passive_ticks = round(settle / dt)
    kp = np.r_[np.full(12, 55.), np.zeros(4)]
    kd = np.full(16, 2.)
    push_tick = -1
    if dynamics_stress is not None:
        kp *= dynamics_stress.gain_scale[0]
        kd *= dynamics_stress.gain_scale[1]
        push_tick = round(dynamics_stress.push_time / dt)
    last = np.zeros(16)
    streak = longest = 0
    peak_impact = 0.
    max_upright = -1.
    upright_samples = 0
    highest_upright_height = None
    handoff_cosine = None
    clipped_leg_actions = clipped_wheel_actions = active_ticks = 0
    wheel_contact_ticks = 0
    best_upright_ground_load = {}
    peak_joint_speed = np.zeros(len(joints))
    overspeed_ticks = 0
    level_first_contact_s = {}
    level_wheel_first_contact_s = {} if stairs else None
    max_progress = float(data.qpos[0])
    start_y = y_at_farthest = float(data.qpos[1])
    max_lateral_excursion = 0.
    heading_error = 0.
    failed_while_upright = dict.fromkeys(("height", "linear_speed", "angular_speed", "wheel_support", "nonwheel_contact"), 0)
    force = np.zeros(6)
    base = model.body("base_link").id
    for tick in range(round((settle + recovery) / dt)):
        t = tick * dt
        active = tick >= passive_ticks
        if tick == push_tick:
            data.qvel[:2] += dynamics_stress.push_delta_xy
            mj.mj_forward(model, data)
        rotation = data.xmat[base].reshape(3, 3)
        if tick == passive_ticks:
            handoff_cosine = float(rotation[2, 2])
        # Isaac Lab advances action history even while its actuators are passive.
        joint_positions = data.qpos[qa].copy()
        if wheel_angle_period is not None:
            joint_positions[12:] = (
                joint_positions[12:] + wheel_angle_period / 2.
            ) % wheel_angle_period - wheel_angle_period / 2.
        command = .4 if stairs else max(0., (t - settle) / recovery)
        linear_velocity_obs = np.zeros(3) if stairs else rotation.T @ data.qvel[:3]
        if stair_gyro_heading_gain or stair_heading_feedback:
            # Integrate relative yaw from the gyro and projected gravity.
            up_yz = rotation[2, 1:]
            yaw_rate = (data.qvel[4:6] @ up_yz) / max(up_yz @ up_yz, .2)
            heading_error = (heading_error + dt * yaw_rate + np.pi) % (2 * np.pi) - np.pi
        if stair_heading_feedback:
            linear_velocity_obs[:2] = [np.cos(heading_error) - 1., np.sin(heading_error)]
        obs = np.r_[linear_velocity_obs, data.qvel[3:6], -rotation[2],
                    command, joint_positions - default, data.qvel[va], last]
        x = torch.as_tensor(obs, dtype=torch.float32)
        for layer in (0, 2, 4, 6):
            x = F.linear(x, state[f"actor.{layer}.weight"], state[f"actor.{layer}.bias"])
            if layer != 6:
                x = F.elu(x)
        last = x.numpy()
        if active:
            active_ticks += 1
            if action_clip is not None:
                clipped_leg_actions += np.count_nonzero(np.abs(last[:12]) > action_clip)
                clipped_wheel_actions += np.count_nonzero(np.abs(last[12:]) > action_clip)
        if action_clip is not None:
            last = np.clip(last, -action_clip, action_clip)
        leg_target = default[:12] + .35 * last[:12]
        if full_leg_range:
            span = np.where(last[:12] >= 0., leg_upper_span, leg_lower_span)
            leg_target = default[:12] + last[:12] * span / 6.
        wheel_target = wheel_scale * last[12:]
        if stair_gyro_heading_gain:
            correction = np.clip(stair_gyro_heading_gain * heading_error, -4., 4.)
            wheel_target += correction * np.array([1., -1., 1., -1.])
        if limits is not None:
            leg_target = np.clip(leg_target, *limits[:12, :2].T)
            wheel_target = np.clip(wheel_target, -limits[12:, 2], limits[12:, 2])
        if native_policy is not None:
            sensors = obs.copy()
            sensors[10:26] = data.qpos[qa]
            last, leg_target, wheel_target, native_active = native_policy(sensors, t)
            if bool(native_active) != active:
                raise RuntimeError("Native policy activation differs from the simulation contract")
        for _ in range(substeps):
            if active:
                leg = kp[:12] * (leg_target - data.qpos[qa[:12]]) - kd[:12] * data.qvel[va[:12]]
                wheel = kd[12:] * (wheel_target - data.qvel[va[12:]])
                drive = np.r_[leg, wheel]
                if drive_speed_limit:
                    speed = data.qvel[va]
                    # Suppress propulsive drive above the limit, preserving braking and contact dynamics.
                    drive[(np.abs(speed) >= limits[:, 2]) & (speed * drive > 0.)] = 0.
                data.ctrl[motors] = np.clip(drive, *model.actuator_ctrlrange[motors].T)
            expected_time = data.time + model.opt.timestep
            mj.mj_step(model, data)
            if abs(data.time - expected_time) > 1e-9 or not np.isfinite(data.qacc).all():
                return {"passed": False, "valid": False, "failure": "unstable_physics",
                        "stable_s": longest * dt, "peak_impact_n": peak_impact}
        if capture is not None:
            capture(data)
        if not active:
            continue
        joint_speed = np.abs(data.qvel[va])
        peak_joint_speed = np.maximum(peak_joint_speed, joint_speed)
        if limits is not None:
            overspeed_ticks += bool(np.any(joint_speed > limits[:, 2]))
        wheel_load = dict.fromkeys(wheel_bodies, 0.)
        top_wheel_load = dict.fromkeys(wheel_bodies, 0.) if stairs else None
        ground_load = {}
        impact = 0.
        for i, contact in enumerate(data.contact):
            surface, other = contact.geom
            if surface not in surfaces:
                surface, other = other, surface
            if surface not in surfaces or other in surfaces:
                continue
            body = model.geom_bodyid[other]
            mj.mj_contactForce(model, data, i, force)
            ground_load[body] = ground_load.get(body, 0.) + float(force[0])
            if body in wheel_load:
                wheel_load[body] += force[0]
                if stairs and surface in stair_geoms and abs(contact.frame[2]) >= .7:
                    level = stair_geoms[surface]
                    if force[0] >= 8.:
                        contact_s = (tick + 1) * dt
                        level_first_contact_s.setdefault(level, contact_s)
                        level_wheel_first_contact_s.setdefault(level, {}).setdefault(body, contact_s)
                    if level == 5:
                        top_wheel_load[body] += force[0]
            else:
                impact = max(impact, force[0])
        peak_impact = max(peak_impact, impact)
        upright_cosine = float(data.xmat[base, 8])
        if reference_step is not None:
            reference_step(obs, leg_target, wheel_target, upright_cosine)
        if upright_cosine > max_upright:
            max_upright = upright_cosine
            best_upright_ground_load = ground_load
        wheel_contact_ticks += any(load >= 8. for load in wheel_load.values())
        upright = upright_cosine >= np.cos(np.deg2rad(20))
        height = float(data.qpos[2])
        if terrain:
            height -= surface_height(*data.qpos[:2])
        checks = {
            "height": height >= .29,
            "linear_speed": np.linalg.norm(data.qvel[:3]) <= .2,
            "angular_speed": np.linalg.norm(data.qvel[3:6]) <= .4,
            "wheel_support": min(wheel_load.values()) >= 8.,
            "nonwheel_contact": impact < 1.,
        }
        if upright:
            upright_samples += 1
            if np.isfinite(height) and (highest_upright_height is None or height > highest_upright_height):
                highest_upright_height = height
            for name, passed in checks.items():
                failed_while_upright[name] += int(not passed)
        stable = upright and all(checks.values())
        if stairs:
            if data.qpos[0] > max_progress:
                max_progress = float(data.qpos[0])
                y_at_farthest = float(data.qpos[1])
            max_lateral_excursion = max(max_lateral_excursion, abs(float(data.qpos[1]) - start_y))
            stable = upright and height >= .25 and impact < 1. and min(top_wheel_load.values()) >= 8.
        streak = streak + 1 if stable else 0
        longest = max(longest, streak)
    result = {"passed": longest * dt >= 1., "valid": True,
            "stable_s": longest * dt, "peak_impact_n": peak_impact, "max_upright_cosine": max_upright,
            "upright_samples": upright_samples, "failed_while_upright": failed_while_upright,
            "handoff_upright_cosine": handoff_cosine, "sampled_ground_friction": ground_friction,
            "clipped_leg_action_fraction": clipped_leg_actions / (12 * active_ticks),
            "clipped_wheel_action_fraction": clipped_wheel_actions / (4 * active_ticks),
            "wheel_contact_s": wheel_contact_ticks * dt,
            "best_upright_ground_load_n": {
                model.body(body).name: load for body, load in best_upright_ground_load.items()
            },
            "final_ground_load_n": {model.body(body).name: load for body, load in ground_load.items()},
            "final_wheel_speed_rad_s": data.qvel[va[12:]].tolist(),
            "peak_joint_speed_rad_s": dict(zip(names, peak_joint_speed.tolist())),
            "urdf_joint_overspeed_s": overspeed_ticks * dt if limits is not None else None,
            "highest_upright_height_m": highest_upright_height,
            "final_base_height_m": float(data.qpos[2]),
            "final_upright_cosine": float(data.xmat[base, 8])}
    if stairs:
        visited_levels = sorted(level_first_contact_s)
        ordered = len(visited_levels) == 5 and all(
            level_first_contact_s[level] <= level_first_contact_s[level + 1] for level in range(1, 5)
        )
        first_step = model.geom("stair_1")
        result.update(
            passed=ordered and longest * dt >= .5 and peak_impact < 50.,
            visited_levels=visited_levels, level_first_contact_s=level_first_contact_s,
            level_wheel_first_contact_s={
                level: {model.body(body).name: time_s for body, time_s in wheel_times.items()}
                for level, wheel_times in level_wheel_first_contact_s.items()
            },
            top_support_s=longest * dt, stair_rise_m=float(2 * first_step.size[2]),
            stair_tread_m=float(2 * first_step.size[0]),
            max_forward_x_m=max_progress, final_forward_x_m=float(data.qpos[0]),
            lateral_excursion_m=max_lateral_excursion, y_at_farthest_x_m=y_at_farthest,
            final_y_m=float(data.qpos[1]),
        )
    return result


def load_flat_model(disable_self_collision, ground_friction_priority, terrain=None, seed=0):
    path = ROOT / "local-step-diagnostic/as2w_flat.xml"
    if terrain is None:
        model = mj.MjModel.from_xml_path(str(path))
    else:
        xml = ET.parse(path).getroot()
        for asset in xml.findall("./asset/mesh"):
            asset.set("file", (path.parent / asset.attrib["file"]).resolve().as_posix())
        ground_geom = xml.find("./worldbody/geom[@name='ground']")
        ground_geom.set("group", "5")
        if terrain == "slope":
            angle = np.deg2rad(8. if seed % 2 else -8.)
            ground_geom.set("quat", f"{np.cos(angle/2)} 0 {np.sin(angle/2)} 0")
        elif terrain == "rough":
            axis = np.linspace(-3., 3., 81)
            x, y = np.meshgrid(axis, axis)
            phases = np.random.default_rng([seed, 611]).uniform(0., 2 * np.pi, size=3)
            elevation = (np.sin(11 * x + phases[0]) + np.cos(9 * y + phases[1])
                         + .5 * np.sin(7 * (x + y) + phases[2]))
            ET.SubElement(xml.find("asset"), "hfield", name="rough_ground",
                          nrow="81", ncol="81", size="3 3 .035 .1",
                          elevation=" ".join(map(str, elevation.ravel())))
            ground_geom.set("type", "hfield")
            ground_geom.set("hfield", "rough_ground")
            del ground_geom.attrib["size"]
        else:
            rise = (.06, .08, .10)[seed % 3]
            tread = (.31, .35, .39)[(seed // 3) % 3]
            for level in range(1, 6):
                start = .7 + (level - 1) * tread
                length = tread if level < 5 else 2.
                top = level * rise
                ET.SubElement(xml.find("worldbody"), "geom", name=f"stair_{level}", type="box",
                              size=f"{length/2} 1.25 {top/2}",
                              pos=f"{start+length/2} 0 {top/2}", group="5",
                              friction=ground_geom.attrib["friction"])
        model = mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    ground = model.geom("ground").id
    surface_ids = [ground]
    if terrain == "stairs":
        surface_ids.extend(model.geom(f"stair_{level}").id for level in range(1, 6))
    if ground_friction_priority:
        model.geom_priority[surface_ids] = int(model.geom_priority.max()) + 1
    if disable_self_collision:
        model.geom_contype[:] = 1
        model.geom_conaffinity[:] = 0
        model.geom_contype[surface_ids] = 0
        model.geom_conaffinity[surface_ids] = 1
    return model


def load_target_limits(contract):
    urdf = ET.parse(ROOT / "official-model/robots/as2w_description/as2w.urdf")
    by_name = {joint.get("name"): joint.find("limit") for joint in urdf.findall("joint")}
    return np.array([
        [float(by_name[name].get(key, "0")) for key in ("lower", "upper", "velocity")]
        for name in contract["joint_names"]
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a single checkpoint")
    parser.add_argument("--video", action="store_true", help="Record seed 12001 for each pose on the selected checkpoint or model_4999")
    parser.add_argument("--video-seed", type=int, action="append",
                        help="Select replay seeds when recording video (repeatable, default: 12001)")
    screens = parser.add_mutually_exclusive_group()
    screens.add_argument("--confirmation", action="store_true",
                         help="Use the independent 64-case confirmation set for checkpoint comparisons")
    screens.add_argument("--flat-dynamics-stress", action="store_true",
                         help="Screen an independent 64-case friction, mass, drive gain and push set")
    screens.add_argument("--flat-dynamics-control", action="store_true",
                         help="Use the stress set's initial poses with nominal dynamics")
    screens.add_argument("--terrain", choices=("slope", "rough", "stairs"),
                         help="Evaluate recovery on a nonflat surface with dynamics stress")
    parser.add_argument("--stair-confirmation", action="store_true",
                        help="Use 64 disjoint blind-stair cases with the same acceptance criteria")
    parser.add_argument("--stair-gyro-heading-gain", type=float, default=0.,
                        help="Diagnose a blind differential wheel correction based on integrated gyro yaw")
    parser.add_argument("--stair-heading-feedback", action="store_true",
                        help="Supply the actor with relative yaw derived from gyro and gravity")
    parser.add_argument("--limit-targets", action="store_true", help="Clip actuator targets to official URDF limits")
    parser.add_argument("--disable-self-collision", action="store_true",
                        help="Preserve ground contacts while disabling robot self-collision")
    parser.add_argument("--ground-friction-priority", action="store_true",
                        help="Use the seeded ground friction for robot-ground contacts")
    parser.add_argument("--action-clip", type=float, help="Match the policy wrapper action limit")
    parser.add_argument("--wheel-scale", type=float, default=6., help="Match the training wheel target scale")
    parser.add_argument("--wheel-angle-period", type=float,
                        help="Match the training simulator's wheel angle observation period in radians")
    parser.add_argument("--full-leg-range", action="store_true",
                        help="Match training with the full official leg joint range")
    args = parser.parse_args()
    if args.stair_confirmation and args.terrain != "stairs":
        parser.error("--stair-confirmation requires --terrain stairs")
    if not 0. <= args.stair_gyro_heading_gain < float("inf") or (
        args.stair_gyro_heading_gain and args.terrain != "stairs"
    ):
        parser.error("--stair-gyro-heading-gain requires a finite nonnegative gain and stair terrain")
    if args.stair_heading_feedback and (args.terrain != "stairs" or args.stair_gyro_heading_gain):
        parser.error("--stair-heading-feedback requires stair terrain and no wheel correction overlay")
    if args.confirmation and args.video:
        parser.error("Record videos on the standard screen before running confirmation")
    if (args.flat_dynamics_stress or args.flat_dynamics_control) and (
        args.video or not args.ground_friction_priority
    ):
        parser.error("Dynamics screens require --ground-friction-priority and do not record videos")
    if args.terrain and (not args.checkpoint or not args.ground_friction_priority):
        parser.error("Terrain screens require --checkpoint and --ground-friction-priority")
    if not 0. < args.wheel_scale < float("inf"):
        parser.error("--wheel-scale must be positive and finite")
    if args.wheel_angle_period is not None and not 0. < args.wheel_angle_period < float("inf"):
        parser.error("--wheel-angle-period must be positive and finite")
    if args.full_leg_range and (not args.limit_targets or args.action_clip != 6.):
        parser.error("--full-leg-range requires --limit-targets and --action-clip 6")
    first_stair_seed = 28001 if args.stair_confirmation else 27001
    video_seeds = args.video_seed or ([first_stair_seed] if args.terrain == "stairs" else [12001])
    contract = json.loads((BASE / "fall-recovery-contract.json").read_text())
    model = load_flat_model(args.disable_self_collision, args.ground_friction_priority)
    limits = load_target_limits(contract) if args.limit_targets else None
    variant = "limited" if args.limit_targets else "history"
    if args.disable_self_collision:
        variant += "-no-self"
    if args.ground_friction_priority:
        variant += "-seeded-friction"
    if args.action_clip is not None:
        if args.action_clip <= 0:
            parser.error("--action-clip must be positive")
        variant += f"-clip{args.action_clip:g}"
    if args.wheel_scale != 6.:
        variant += f"-wheel{args.wheel_scale:g}"
    if args.full_leg_range:
        variant += "-full-leg-range"
    if args.wheel_angle_period is not None:
        variant += f"-angle{args.wheel_angle_period:g}"
    seeds = range(12001, 12005)
    if args.confirmation:
        variant += "-confirmation"
        seeds = range(22001, 22017)
    if args.flat_dynamics_stress:
        variant += "-flat-dynamics-stress"
        seeds = range(23001, 23017)
    if args.flat_dynamics_control:
        variant += "-flat-dynamics-control"
        seeds = range(23001, 23017)
    if args.terrain:
        variant += f"-{args.terrain}-dynamics-stress"
    poses = POSES
    if args.terrain == "stairs":
        if args.stair_heading_feedback:
            variant += "-heading-feedback"
        if args.stair_gyro_heading_gain:
            variant += f"-gyro-heading{args.stair_gyro_heading_gain:g}"
        if args.stair_confirmation:
            variant += "-confirmation"
        count = 64 if args.stair_confirmation else 16
        seeds = range(first_stair_seed, first_stair_seed + count)
        poses = {"upright": 0.}
    torch.set_num_threads(1)
    checkpoints = [args.checkpoint] if args.checkpoint else sorted(BASE.glob("model_*.pt"))
    report_suffix = f"-{args.checkpoint.stem}" if args.checkpoint else ""
    with (BASE / f"flat-screen-{variant}{report_suffix}.jsonl").open("w", encoding="utf-8") as report, torch.inference_mode():
        for checkpoint in checkpoints:
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
            for pose, roll in poses.items():
                for seed in seeds:
                    stress = None
                    if args.flat_dynamics_stress or args.terrain:
                        model = load_flat_model(args.disable_self_collision, args.ground_friction_priority,
                                                args.terrain, seed)
                        stress = sample_dynamics_stress(seed, model.nbody - 1, len(contract["joint_names"]))
                    if args.video and (args.checkpoint or checkpoint.stem == "model_4999") and seed in video_seeds:
                        import imageio_ffmpeg

                        camera = mj.MjvCamera()
                        camera.distance = 2.4
                        camera.azimuth = 135
                        camera.elevation = -22
                        scene_option = mj.MjvOption()
                        if args.terrain:
                            scene_option.geomgroup[5] = 1
                        path = BASE / f"mujoco-{variant}-{checkpoint.stem}-{pose}-{seed}.mp4"
                        with mj.Renderer(model, height=480, width=640) as renderer, subprocess.Popen(
                            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                             "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x480",
                             "-framerate", "50", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE,
                        ) as video:
                            def capture(data):
                                camera.lookat[:] = data.qpos[:3]
                                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                                video.stdin.write(renderer.render().tobytes())

                            result = evaluate(model, contract, state, seed, roll, capture, limits,
                                              args.action_clip, args.wheel_scale,
                                              full_leg_range=args.full_leg_range,
                                              wheel_angle_period=args.wheel_angle_period,
                                              dynamics_stress=stress, terrain=args.terrain,
                                              stair_gyro_heading_gain=args.stair_gyro_heading_gain,
                                              stair_heading_feedback=args.stair_heading_feedback)
                            video.stdin.close()
                            if video.wait():
                                raise RuntimeError(f"Video encoding failed: {path}")
                    else:
                        result = evaluate(model, contract, state, seed, roll, limits=limits,
                                          action_clip=args.action_clip, wheel_scale=args.wheel_scale,
                                          full_leg_range=args.full_leg_range, dynamics_stress=stress,
                                          wheel_angle_period=args.wheel_angle_period, terrain=args.terrain,
                                          stair_gyro_heading_gain=args.stair_gyro_heading_gain,
                                          stair_heading_feedback=args.stair_heading_feedback)
                    if args.terrain:
                        result["terrain"] = args.terrain
                    if stress is not None:
                        result["dynamics_stress"] = {
                            "mass_scale": stress.mass_scale.tolist(), "gain_scale": stress.gain_scale.tolist(),
                            "push_time_s": stress.push_time, "push_delta_xy_m_s": stress.push_delta_xy.tolist(),
                        }
                    print(json.dumps(dict(result, checkpoint=checkpoint.name, pose=pose, seed=seed)),
                          file=report, flush=True)


if __name__ == "__main__":
    main()
