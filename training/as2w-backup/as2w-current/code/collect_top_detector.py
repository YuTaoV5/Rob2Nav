"""Collect privileged stair-top labels for a velocity-free recurrent detector."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
import mujoco_screen as sim
from train_recovery_student import make_actor, numpy_actor


ROOT = None
CONTRACT = None
LIMITS = None
STATE = None
ACTOR = None
DEFAULT = None
STAIR_RISE = None


def initialize(results, checkpoint, stair_rise):
    global ROOT, CONTRACT, LIMITS, STATE, ACTOR, DEFAULT, STAIR_RISE
    torch.set_num_threads(1)
    ROOT = Path(results)
    sim.ROOT = ROOT
    CONTRACT = json.loads((ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    LIMITS = sim.load_target_limits(CONTRACT)
    STATE = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    ACTOR = numpy_actor(make_actor(STATE))
    DEFAULT = np.asarray(CONTRACT["default_joint_pos"])
    STAIR_RISE = stair_rise


def collect(seed):
    model = sim.load_flat_model(True, True, "stairs", seed, stair_rise=STAIR_RISE)
    stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
    stair = model.geom("stair_5")
    fifth_leading = float(model.geom_pos[stair.id, 0] - model.geom_size[stair.id, 0])
    observations = []
    labels = []
    relative_x = []

    def planner(data, obs, elapsed, advance):
        observations.append(obs.astype(np.float32, copy=True))
        x = float(data.qpos[0] - fifth_leading)
        relative_x.append(x)
        labels.append(x >= .55)
        action = np.clip(ACTOR(obs), -6., 6.)
        leg = np.clip(DEFAULT[:12] + .35 * action[:12], *LIMITS[:12, :2].T)
        wheel = np.clip(16. * action[12:], -LIMITS[12:, 2], LIMITS[12:, 2])
        return action, leg, wheel

    row = sim.evaluate(
        model, CONTRACT, STATE, seed, 0., limits=LIMITS, action_clip=6., wheel_scale=16.,
        dynamics_stress=stress, terrain="stairs", wheel_angle_period=4 * np.pi,
        observation_suffix=evaluation.gait_clock, native_planner=planner,
    )
    return {
        "seed": seed,
        "observations": np.asarray(observations, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.uint8),
        "relative_x": np.asarray(relative_x, dtype=np.float32),
        "visited_levels": np.asarray(row["visited_levels"], dtype=np.int8),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--stair-rise", type=float, default=.06)
    args = parser.parse_args()

    context = mp.get_context("spawn")
    with context.Pool(
        args.workers, initializer=initialize,
        initargs=(str(args.results.resolve()), str(args.checkpoint.resolve()), args.stair_rise),
    ) as pool:
        rows = list(pool.imap(collect, range(args.seed_start, args.seed_start + args.count)))
    observations = np.stack([row["observations"] for row in rows])
    labels = np.stack([row["labels"] for row in rows])
    relative_x = np.stack([row["relative_x"] for row in rows])
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    level_counts = np.asarray([len(row["visited_levels"]) for row in rows], dtype=np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, observations=observations, labels=labels, relative_x=relative_x,
        seeds=seeds, level_counts=level_counts,
    )
    print(json.dumps({
        "episodes": len(rows), "shape": list(observations.shape),
        "positive_frames": int(labels.sum()), "positive_episodes": int(np.any(labels, axis=1).sum()),
        "level5_episodes": int((level_counts == 5).sum()), "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

