"""Evaluate the joint flat driving and blind stair policy in MuJoCo."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

import mujoco_screen as sim
from train_recovery_student import make_actor, numpy_actor


FLAT_COMMANDS = {
    "stop": (0., 0., 0.),
    "forward_slow": (.2, 0., 0.),
    "forward": (.4, 0., 0.),
    "forward_fast": (.6, 0., 0.),
    "reverse": (-.4, 0., 0.),
    "left": (0., .3, 0.),
    "right": (0., -.3, 0.),
    "turn_left": (0., 0., .65),
    "turn_right": (0., 0., -.65),
    "diagonal_left": (.35, .25, 0.),
    "diagonal_right": (.35, -.25, 0.),
    "arc_left": (.35, 0., .5),
    "arc_right": (.35, 0., -.5),
}


def gait_clock(elapsed):
    angle = 2. * np.pi * elapsed / .62
    return (np.sin(angle), np.cos(angle))


def prior_flat_planner(actor, dt, observe=None):
    from omni_motion_prior import OmniMotionPrior

    prior = OmniMotionPrior().eval()

    def plan(model, data, obs, command, step, advance):
        raw_action = actor(obs)
        if observe is not None:
            observe(obs, raw_action)
        inputs = (raw_action[None, :], command[None, :], obs[None, 5:6],
                  obs[None, 7:8], [step * dt])
        return prior(*(torch.as_tensor(value, dtype=torch.float32) for value in inputs))[0].numpy()

    return plan


def flat_episode(actor, contract, limits, target, seed, observation_suffix=None, planner=None, capture=None, model=None):
    if model is None:
        model = sim.load_flat_model(True, True, seed=seed)
    data = sim.mj.MjData(model)
    joints = [model.joint(name) for name in contract["joint_names"]]
    q_ids = np.array([joint.qposadr[0] for joint in joints])
    dq_ids = np.array([joint.dofadr[0] for joint in joints])
    motors = [model.actuator(name).id for name in contract["joint_names"]]
    defaults = np.asarray(contract["default_joint_pos"])
    rng = np.random.default_rng(seed)
    model.geom_friction[model.geom("ground").id, 0] = rng.uniform(.55, .85)
    base = model.body("base_link").id
    data.qpos[2] = .37
    data.qpos[q_ids] = defaults + rng.uniform(-.08, .08, size=len(q_ids))
    sim.mj.mj_forward(model, data)
    dt = contract["control_dt"]
    substeps = round(dt / model.opt.timestep)
    previous = np.zeros(16)
    samples = []
    fallen = False

    def advance(sim_state, action):
        leg_target = np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T)
        wheel_target = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
        for _ in range(substeps):
            q, dq = sim_state.qpos[q_ids], sim_state.qvel[dq_ids]
            torque = np.r_[55. * (leg_target - q[:12]) - 2. * dq[:12], 2. * (wheel_target - dq[12:])]
            sim_state.ctrl[motors] = np.clip(torque, *model.actuator_ctrlrange[motors].T)
            sim.mj.mj_step(model, sim_state)

    for step in range(round(6. / dt)):
        elapsed = step * dt
        command = target if elapsed < 4. else np.zeros(3)
        rotation = data.xmat[base].reshape(3, 3)
        q = data.qpos[q_ids].copy()
        q[12:] = (q[12:] + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
        obs = np.r_[command - (.4, 0., 0.), data.qvel[3:6], -rotation[2],
                    .4, q - defaults, data.qvel[dq_ids], previous]
        if observation_suffix is not None:
            obs = np.r_[obs, observation_suffix(elapsed)]
        previous = actor(obs) if planner is None else planner(model, data, obs, command, step, advance)
        advance(data, previous)
        if capture is not None:
            capture(model, data, command)
        rotation = data.xmat[base].reshape(3, 3)
        actual = np.r_[(rotation.T @ data.qvel[:3])[:2], data.qvel[5]]
        samples.append(actual)
        fallen |= data.qpos[2] < .20 or rotation[2, 2] < .80
    samples = np.asarray(samples)
    moving = samples[round(1.5 / dt):round(4. / dt)]
    stopped = samples[round(5. / dt):]
    xy_error = np.linalg.norm(moving[:, :2] - target[:2], axis=1).mean()
    yaw_error = np.abs(moving[:, 2] - target[2]).mean()
    stop_speed = np.linalg.norm(stopped[:, :2], axis=1).mean()
    stop_yaw_rate = np.abs(stopped[:, 2]).mean()
    return {"passed": bool(not fallen and xy_error <= .15 and yaw_error <= .22
                           and stop_speed <= .12 and stop_yaw_rate <= .15),
            "fallen": bool(fallen), "xy_mae_m_s": float(xy_error),
            "yaw_mae_rad_s": float(yaw_error), "stop_speed_m_s": float(stop_speed),
            "stop_yaw_rate_rad_s": float(stop_yaw_rate),
            "mean_body_twist": moving.mean(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirmation", action="store_true",
                        help="Use disjoint flat commands and the 64-scene stair confirmation set")
    parser.add_argument("--motion-prior", action="store_true",
                        help="Evaluate an actor with the command and IMU gait correction enabled")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = sim.load_target_limits(contract)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    observation_size = state["actor.0.weight"].shape[1]
    if observation_size not in (58, 60):
        raise ValueError(f"Unsupported observation width: {observation_size}")
    suffix = gait_clock if observation_size == 60 else None
    actor = numpy_actor(make_actor(state))
    manifest = args.checkpoint.with_name("run-config.json")
    motion_prior = args.motion_prior or (
        manifest.is_file() and json.loads(manifest.read_text()).get("omni_motion_prior", False)
    )
    planner = None
    if motion_prior:
        if observation_size != 60:
            raise ValueError("The motion prior requires the 60-input actor with gait clock")
        planner = prior_flat_planner(actor, contract["control_dt"])

    if args.confirmation:
        generator = np.random.default_rng(91291)
        random_targets = generator.uniform((-.6, -.35, -.8), (.6, .35, .8), size=(48, 3))
        flat_cases = [(f"mixed_{index:02d}", target, 91101 + index)
                      for index, target in enumerate(random_targets)]
        flat_cases.extend(("stop", np.zeros(3), seed) for seed in range(91201, 91205))
        stair_seeds = range(28001, 28065)
    else:
        flat_cases = [(name, np.asarray(target), seed)
                      for name, target in FLAT_COMMANDS.items() for seed in range(81201, 81205)]
        stair_seeds = range(27001, 27017)
    with torch.inference_mode():
        flat = [dict(flat_episode(actor, contract, limits, target, seed, suffix, planner),
                     command=name, target=target.tolist(), seed=seed)
                for name, target, seed in flat_cases]
        stairs = []
        for seed in stair_seeds:
            model = sim.load_flat_model(True, True, "stairs", seed)
            stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
            stairs.append(dict(sim.evaluate(model, contract, state, seed, 0., limits=limits,
                                            action_clip=6., wheel_scale=16., wheel_angle_period=4 * np.pi,
                                            dynamics_stress=stress, terrain="stairs",
                                            observation_suffix=suffix), seed=seed))
    for row in stairs:
        row["passed"] = bool(row["passed"])
    args.output.write_text(json.dumps({"flat": flat, "stairs": stairs,
                                      "omni_motion_prior": motion_prior}, indent=2) + "\n")
    print(json.dumps({"flat": [sum(row["passed"] for row in flat), len(flat)],
                      "stairs": [sum(row["passed"] for row in stairs), len(stairs)],
                      "stair_highest_levels": Counter(max(row.get("visited_levels", []), default=0)
                                                     for row in stairs)}))


if __name__ == "__main__":
    main()
