"""Replay Isaac AS2-W policies on their native PhysX flat environment."""
import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.physics = "isaacsim_physx"
    args.num_envs = 8
    args.command_stage = "all"
    args.direction_focus = True
    args.zero_focus = False
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        report = evaluate(args, physics)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def evaluate(args, physics):
    import torch
    from torch.nn import functional as F

    ALL_FLAT_COMMANDS = (
        ("forward", (.10, 0., 0.)), ("yaw-left", (0., 0., .30)),
        ("reverse", (-.10, 0., 0.)), ("yaw-right", (0., 0., -.30)),
        ("left", (0., .10, 0.)), ("right", (0., -.10, 0.)),
        ("diagonal-fl", (.071, .071, 0.)), ("diagonal-fr", (.071, -.071, 0.)),
        ("diagonal-rl", (-.071, .071, 0.)), ("diagonal-rr", (-.071, -.071, 0.)),
        ("stop", (0., 0., 0.)),
    )

    from as2w_isaac_flat_env_all import FlatEnvironment, REAR, configuration

    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)["actor_state_dict"]
    args.zero_memory = state["mlp.0.weight"].shape[1] == 65
    sequences = {
        "dynamic": tuple((cmd, 10. if name == "stop" else 2.) for name, cmd in ALL_FLAT_COMMANDS),
        "static": (((0., 0., 0.), 10.),),
    }
    report = {}
    args.seed = 28101
    cfg = configuration(args, physics)
    cfg.episode_length_s = 40.
    env = FlatEnvironment(cfg)
    try:
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene["contact_forces"]
        nonwheel = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_foot")]
        rear_ids = [env.as2w_joint_ids[i] for i in REAR]
        for seed in (28101, 38101):
            report[str(seed)] = {}
            for scenario, commands in sequences.items():
                observations, _ = env.reset(seed=seed)
                term.time_left.fill_(100.)
                alive = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
                rows = []
                for cmd, duration in commands:
                    moving = any(cmd)
                    term.targets[:, :] = term.targets.new_tensor(cmd)
                    samples, contacts = [], []
                    with torch.inference_mode():
                        for tick in range(round(duration / env.step_dt)):
                            action = observations["policy"]
                            for layer in (0, 2, 4, 6):
                                action = F.linear(action, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"])
                                if layer != 6:
                                    action = F.elu(action)
                            observations, _, terminated, truncated, _ = env.step(action.clamp(-6., 6.))
                            alive &= ~(terminated | truncated)
                            contact = sensor.data.net_forces_w.torch[:, nonwheel].norm(dim=-1).amax(dim=-1) > 4.
                            contacts.append(contact)
                            if tick >= round((.5 if moving else 1.) / env.step_dt):
                                w, x, y, z = robot.data.root_quat_w.torch.unbind(dim=-1)
                                heading = torch.atan2(2. * (w*z+x*y), 1.-2.*(y.square()+z.square()))
                                rear = robot.data.joint_pos.torch[:, rear_ids] - robot.data.default_joint_pos.torch[:, rear_ids]
                                samples.append(torch.cat((robot.data.root_pos_w.torch[:, :2],
                                    robot.data.root_lin_vel_b.torch[:, :2], heading[:, None], robot.data.root_ang_vel_b.torch[:, 2:3], rear.abs()), dim=-1))
                    values = torch.stack(samples)
                    delta_xy = values[..., :2] - values[0, :, :2]
                    displacement = delta_xy.norm(dim=-1).amax(dim=0)
                    heading_delta = (values[..., 4] - values[0, :, 4] + torch.pi) % (2. * torch.pi) - torch.pi
                    drift_deg = torch.rad2deg(heading_delta.abs().amax(dim=0))
                    residual = values[..., 2:4].norm(dim=-1).quantile(.95, dim=0)
                    mean_velocity = values[..., [2, 3, 5]].mean(dim=0)
                    rear_p95 = values[..., 6:].permute(1, 0, 2).flatten(1).quantile(.95, dim=-1)
                    touch = torch.stack(contacts).sum(dim=0)
                    held = alive & (touch == 0) & (displacement <= .02) & (drift_deg <= 3.) & (residual <= .02)
                    rows.append(dict(command=cmd, alive=alive.tolist(), mean_velocity=mean_velocity.tolist(),
                                     drift_m=displacement.tolist(), yaw_drift_deg=drift_deg.tolist(),
                                     residual_p95_m_s=residual.tolist(), rear_p95_rad=rear_p95.tolist(),
                                     nonwheel_frames=touch.tolist(), held=held.tolist() if not moving else None))
                report[str(seed)][scenario] = rows
    finally:
        env.close()
    return report


if __name__ == "__main__":
    main()
