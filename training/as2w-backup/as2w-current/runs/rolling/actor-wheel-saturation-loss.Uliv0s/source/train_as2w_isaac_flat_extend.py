"""Launch checkpoint-free flat AS2-W PPO on Isaac Sim PhysX."""
import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--extend-from", type=Path, help="Continue from a checkpoint created in this same run")
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--command-stage", choices=("sagittal", "heading", "all"), default="sagittal")
    parser.add_argument("--zero-focus", action="store_true", help="Tighten the zero-command training objective")
    parser.add_argument("--zero-memory", action="store_true", help="Expose wheel displacement from the zero-command entry")
    add_launcher_args(parser)
    args = parser.parse_args()
    args.physics = "isaacsim_physx"
    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        from as2w_isaac_flat_env_extend import train

        train(args, physics)


if __name__ == "__main__":
    main()
