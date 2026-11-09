"""Train a bounded stair residual inside the unified proprioceptive controller."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from screen_omni_estimated_lateral import EstimatedLateralController, StairPlaneActor
from search_lateral_omni_teacher import course
from train_omni_lateral_history import screen
from train_omni_mujoco import OmniEnvironment, verified_reference_resets
from train_omni_velocity_estimator import estimator_model
from train_recovery_student import make_actor, numpy_actor


RESIDUAL_LIMIT = .48
ARCH = (128, 96)


def residual_model():
    return torch.nn.Sequential(torch.nn.Linear(60, ARCH[0]), torch.nn.ELU(),
                               torch.nn.Linear(ARCH[0], ARCH[1]), torch.nn.ELU(),
                               torch.nn.Linear(ARCH[1], 16))


def arrays(policy, key):
    return [(a.numpy(), b.numpy()) for a, b in policy.get(key, ())]


def make_route(policy, anchor, primary, residual, contract, scale=None, stall_gated=None):
    settings = policy.get("stair_residual_config", {})
    if scale is None:
        scale = settings.get("scale", 1.)
    if stall_gated is None:
        stall_gated = settings.get("stall_gated", False)

    def stair(obs):
        weight = scale
        if stall_gated:
            slow_wheel_speed = .086 * np.mean(np.sort(obs[38:42])[:2])
            weight *= np.clip((.28 - slow_wheel_speed) / .22, 0., 1.)
        return (primary(obs) + weight * residual(obs).clip(-RESIDUAL_LIMIT, RESIDUAL_LIMIT)).clip(-6., 6.)

    return StairPlaneActor(policy["contact_gate"], anchor, stair, contract["control_dt"],
                           contract["default_joint_pos"], arrays(policy, "wheel_profiles"))


class StairResidualEnvironment(OmniEnvironment):
    def __init__(self, results, policy, refs):
        super().__init__(results, 0, refs, .48 if refs else 0., stair_speed_control=True)
        self.policy = policy
        self.anchor, self.primary = (numpy_actor(make_actor(policy[key])) for key in ("anchor", "stair"))
        self.action_space = gym.spaces.Box(-RESIDUAL_LIMIT, RESIDUAL_LIMIT, shape=(16,), dtype=np.float32)

    def sample_scene_seed(self):
        height_index = int(self.np_random.choice(3, p=(.24, .38, .38)))
        return 3 * int(self.np_random.integers(100_000, 300_000_000)) + height_index

    def reset(self, *, seed=None, options=None):
        self.residual = np.zeros(16, dtype=np.float32)
        self.route = make_route(self.policy, self.anchor, self.primary, lambda obs: self.residual, self.contract)
        self.had_impact = False
        self.replaying = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.replaying = False

    def step(self, action):
        self.residual = np.zeros(16, dtype=np.float32) if self.replaying else action
        executed = self.route(self.observe())
        active = self.route.switch_tick is not None and not self.route.settling
        observation, reward, terminated, truncated, info = super().step(action if self.replaying else executed)
        if active and not self.replaying:
            reward -= .018 * float(np.mean(action * action))
        info["residual_active"] = active
        return observation, reward, terminated, truncated, info

    def stair_reward(self, upright):
        reward, unsafe = super().stair_reward(upright)
        invalid_impact = self.last_stair_impact >= 50.
        if invalid_impact and not self.had_impact:
            reward -= 1.25
        self.had_impact |= invalid_impact
        return reward, unsafe


class ExportResidual(BaseCallback):
    def __init__(self, policy, output):
        super().__init__()
        self.policy, self.output = policy, output
        self.rollouts = 0
        self.active = 0
        self.frames = 0

    def _on_step(self):
        info = self.locals["infos"]
        self.active += sum(row.get("residual_active", False) for row in info)
        self.frames += len(info)
        return True

    def save(self):
        policy = self.model.policy
        residual = residual_model()
        for target, source in zip(residual[::2], (*policy.mlp_extractor.policy_net[::2], policy.action_net)):
            target.load_state_dict(source.state_dict())
        path = self.output / f"model_{self.num_timesteps}.pt"
        torch.save({**self.policy, "stair_residual": residual.state_dict()}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        print(json.dumps({"checkpoint": str(path), "active_fraction": self.active / max(self.frames, 1)}), flush=True)

    def _on_rollout_start(self):
        if self.rollouts % 10 == 0:
            self.save()
        self.rollouts += 1

    def _on_training_end(self):
        self.save()


def train(args, root, policy):
    refs = []
    for path in args.demos:
        with np.load(path) as archive:
            for seed in np.unique(archive["seed"]):
                actions = archive["actions"][archive["seed"] == seed]
                if len(actions) > 90:
                    refs.append((int(seed), actions))
    if len({seed for seed, _ in refs}) != len(refs):
        raise ValueError("Reference archives must use disjoint seeds")
    refs = verified_reference_resets(root, refs)
    if args.demos and not refs:
        raise ValueError("No provided demonstrations replayed successfully")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run-config.json").write_text(json.dumps({
        "base": str(args.policy), "training_steps": args.steps, "envs": args.envs,
        "reference_seeds": [seed for seed, _ in refs], "residual_limit": RESIDUAL_LIMIT,
        "reference_probability": .48 if refs else 0.,
    }, indent=2))
    env = SubprocVecEnv([lambda: StairResidualEnvironment(root, policy, refs) for _ in range(args.envs)],
                        start_method="fork")
    try:
        ppo = PPO("MlpPolicy", env, learning_rate=7e-5, n_steps=128, batch_size=128,
                  n_epochs=4, gamma=.99, gae_lambda=.95, clip_range=.12, target_kl=.018,
                  policy_kwargs={"activation_fn": torch.nn.ELU, "net_arch": dict(pi=list(ARCH), vf=[128, 96]),
                                 "log_std_init": float(np.log(.13))},
                  device="cpu", seed=548, verbose=1)
        with torch.no_grad():
            ppo.policy.action_net.weight.zero_()
            ppo.policy.action_net.bias.zero_()
        ppo.learn(args.steps, callback=ExportResidual(policy, args.output), log_interval=10)
    finally:
        env.close()


def evaluate(args, root, policy):
    contract, limits, geometry = course(root)
    anchor, primary = (numpy_actor(make_actor(policy[key])) for key in ("anchor", "stair"))
    estimator = estimator_model()
    estimator.load_state_dict(policy["estimator"])
    correction = numpy_actor(estimator)
    residual = residual_model()
    residual.load_state_dict(policy["stair_residual"])
    residual = numpy_actor(residual)
    params = np.asarray(policy["parameters"])
    odometry = arrays(policy, "odometry_profiles")

    def base_factory():
        return make_route(policy, anchor, primary, residual, contract, args.residual_scale, args.stall_gated)

    def controller_factory(base):
        return EstimatedLateralController(base, correction, params, contract, geometry, odometry)

    report = screen(base_factory, policy["anchor"], None, params, contract, limits,
                    controller_factory, args.confirmation)
    args.output.write_text(json.dumps(report, indent=2, default=lambda v: v.item()))
    print(json.dumps({name: [sum(row["passed"] for row in rows), len(rows)] for name, rows in report.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("train", "screen"):
        command = sub.add_parser(name)
        command.add_argument("results", type=Path)
        command.add_argument("policy", type=Path)
        command.add_argument("output", type=Path)
        if name == "train":
            command.add_argument("--demos", nargs="+", type=Path, default=())
            command.add_argument("--envs", type=int, default=10)
            command.add_argument("--steps", type=int, default=204800)
        else:
            command.add_argument("--confirmation", action="store_true")
            command.add_argument("--residual-scale", type=float)
            command.add_argument("--stall-gated", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = torch.load(args.policy, map_location="cpu", weights_only=True)
    if args.mode == "train":
        train(args, args.results.resolve(), policy)
    else:
        if args.residual_scale is not None and not 0. <= args.residual_scale <= 1.:
            parser.error("Residual scale must be in [0, 1]")
        evaluate(args, args.results.resolve(), policy)


if __name__ == "__main__":
    main()
