"""Fine-tune the blind AS2-W actor in the same dynamics as its MuJoCo screens."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import evaluate_omni_stairs as evaluation
import mujoco_screen as sim
from omni_motion_prior import OmniMotionPrior
from train_recovery_student import make_actor, numpy_actor
from warm_start_omni_stairs import benchmark_data, motion_prior_anchor, tensor_data


class OmniEnvironment(gym.Env):
    def __init__(self, results, rank, reference_resets=(), reference_probability=0.,
                 stair_speed_control=False, stair_rise=None):
        sim.ROOT = results
        self.contract = json.loads((results / "fall-baseline/fall-recovery-contract.json").read_text())
        self.limits = sim.load_target_limits(self.contract)
        self.defaults = np.asarray(self.contract["default_joint_pos"])
        self.dt = self.contract["control_dt"]
        self.rank = rank
        self.reference_resets = reference_resets
        self.reference_probability = reference_probability
        self.stair_speed_control = stair_speed_control
        self.stair_rise = stair_rise
        self.prior = OmniMotionPrior().eval()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(60,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-6., 6., shape=(16,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scene_seed = options.get("scene_seed")
        self.stairs = self.rank % 3 == 0
        reference = None
        if (scene_seed is None and self.stairs and self.reference_resets
                and self.np_random.random() < self.reference_probability):
            scene_seed, reference = self.reference_resets[self.np_random.integers(len(self.reference_resets))]
        if scene_seed is None:
            scene_seed = int(self.np_random.integers(100_000, 1_000_000_000))
        rng = np.random.default_rng(scene_seed)
        model = self.model = sim.load_flat_model(
            True, True, "stairs" if self.stairs else None, scene_seed, stair_rise=self.stair_rise)
        data = self.data = sim.mj.MjData(model)
        names = self.contract["joint_names"]
        joints = [model.joint(name) for name in names]
        self.qa = [joint.qposadr[0] for joint in joints]
        self.va = [joint.dofadr[0] for joint in joints]
        self.motors = [model.actuator(name).id for name in names]
        self.base = model.body("base_link").id
        self.wheel_index = {model.body(name).id: index for index, name in enumerate(names[12:])}
        ground = model.geom("ground").id
        self.surfaces = {ground: 0}
        self.kp = np.r_[np.full(12, 55.), np.zeros(4)]
        self.kd = np.full(16, 2.)
        self.stress = None
        if self.stairs:
            self.surfaces.update({model.geom(f"stair_{level}").id: level for level in range(1, 6)})
            self.stress = sim.sample_dynamics_stress(scene_seed, model.nbody - 1, 16)
            model.body_mass[1:] *= self.stress.mass_scale
            model.body_inertia[1:] *= self.stress.mass_scale[:, None]
            sim.mj.mj_setConst(model, data)
            model.geom_friction[list(self.surfaces), 0] = self.stress.friction
            self.kp *= self.stress.gain_scale[0]
            self.kd *= self.stress.gain_scale[1]
            # Match the benchmark's random draw order, including its nominal friction draw.
            rng.uniform(.55, .85)
            data.qpos[:2] = rng.uniform((-.12, -.15), (.12, .15))
            yaw = rng.uniform(-.08, .08)
            data.qpos[3:7] = [np.cos(yaw / 2.), 0., 0., np.sin(yaw / 2.)]
            data.qpos[self.qa] = self.defaults + rng.uniform(-.12, .12, size=16)
            self.target = np.array((.4, 0., 0.))
        else:
            model.geom_friction[ground, 0] = rng.uniform(.55, .85)
            data.qpos[self.qa] = self.defaults + rng.uniform(-.08, .08, size=16)
            if "target" in options:
                self.target = np.asarray(options["target"])
            elif self.np_random.random() < .65:
                commands = tuple(evaluation.FLAT_COMMANDS.values())
                self.target = np.asarray(commands[self.np_random.integers(len(commands))])
            else:
                self.target = self.np_random.uniform((-.6, -.35, -.8), (.6, .35, .8))
        data.qpos[2] = .37
        sim.mj.mj_forward(model, data)
        self.tick = 0
        self.previous = np.zeros(16)
        self.force = np.zeros(6)
        self.levels = np.zeros(4, dtype=int)
        self.top_streak = 0
        self.start_x = float(data.qpos[0])
        self.max_x = self.start_x
        self.substeps = round(self.dt / model.opt.timestep)
        self.leg_low, self.leg_high = self.limits[:12, :2].T
        self.wheel_limit = self.limits[12:, 2]
        self.torque_low, self.torque_high = model.actuator_ctrlrange[self.motors].T
        if reference is not None:
            # Keep full-course resets in the mixture and train difficult transfers from successful trajectories.
            anchor = self.np_random.integers(50, min(365, len(reference) - 40))
            for action in reference[:anchor]:
                _, _, terminated, truncated, _ = self.step(action)
                if terminated or truncated:
                    fallback = int(self.np_random.integers(100_000, 1_000_000_000))
                    return self.reset(options={"scene_seed": fallback})
            data.qvel[:2] += self.np_random.uniform(-.035, .035, size=2)
            self.top_streak = 0
            sim.mj.mj_forward(model, data)
        return self.observe(), {"stair_reference_reset": reference is not None}

    @property
    def command(self):
        return self.target if self.stairs or self.tick * self.dt < 4. else np.zeros(3)

    def observe(self):
        data = self.data
        q = data.qpos[self.qa].copy()
        q[12:] = (q[12:] + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
        return np.r_[self.command - (.4, 0., 0.), data.qvel[3:6], -data.xmat[self.base, 6:9],
                     .4, q - self.defaults, data.qvel[self.va], self.previous,
                     evaluation.gait_clock(self.tick * self.dt)].astype(np.float32)

    def stair_reward(self, upright):
        data = self.data
        contacts = np.zeros(4, dtype=int)
        top_loads = np.zeros(4)
        impact = 0.
        for index, contact in enumerate(data.contact):
            surface, robot_geom = contact.geom
            if surface not in self.surfaces:
                surface, robot_geom = robot_geom, surface
            if surface not in self.surfaces or robot_geom in self.surfaces:
                continue
            sim.mj.mj_contactForce(self.model, data, index, self.force)
            wheel = self.wheel_index.get(self.model.geom_bodyid[robot_geom])
            if wheel is None:
                impact = max(impact, self.force[0])
            elif abs(contact.frame[2]) >= .7:
                level = self.surfaces[surface]
                if self.force[0] >= 8.:
                    contacts[wheel] = max(contacts[wheel], level)
                if level == 5:
                    top_loads[wheel] += self.force[0]
        levels = np.maximum(self.levels, contacts)
        newly_reached = (levels - self.levels) @ np.array((.25, .25, .50, .50))
        self.levels = levels
        progress = max(0., data.qpos[0] - self.max_x)
        self.max_x = max(self.max_x, data.qpos[0])
        forward_speed = data.xmat[self.base, [0, 3, 6]] @ data.qvel[:3]
        excess_speed = 0.
        if self.stair_speed_control:
            progress = min(progress, .50 * self.dt)
            excess_speed = max(0., forward_speed - .55)
        tracking = np.exp(-((forward_speed - .40) / .15) ** 2)
        stalled = self.tick * self.dt >= 1. and self.max_x - self.start_x < .12
        top = upright >= .94 and impact < 1. and top_loads.min() >= 8.
        self.top_streak = self.top_streak + 1 if top else 0
        reward = (8. * progress + .02 * tracking + 2. * newly_reached + .20 * top
                  + 4. * (self.top_streak == round(.5 / self.dt))
                  - .10 * abs(data.qpos[1]) - .010 * max(impact - 15., 0.)
                  - .80 * excess_speed - .03 * stalled)
        self.last_stair_impact = impact
        return reward, impact > 90. or abs(data.qpos[1]) > .65

    def step(self, action):
        command = self.command
        model, data = self.model, self.data
        if (command[1] or command[2]) and getattr(self, 'apply_motion_prior', True):
            values = (action[None, :], command[None, :], data.qvel[None, 5:6],
                      -data.xmat[self.base:self.base + 1, 7:8], [self.tick * self.dt])
            with torch.inference_mode():
                action = self.prior(*(torch.as_tensor(x, dtype=torch.float32) for x in values))[0].numpy()
        self.previous = action
        leg = np.clip(self.defaults[:12] + .35 * action[:12], self.leg_low, self.leg_high)
        wheel = np.clip(16. * action[12:], -self.wheel_limit, self.wheel_limit)
        for _ in range(self.substeps):
            q, dq = data.qpos[self.qa], data.qvel[self.va]
            torque = np.r_[self.kp[:12] * (leg - q[:12]) - self.kd[:12] * dq[:12],
                           self.kd[12:] * (wheel - dq[12:])]
            data.ctrl[self.motors] = np.clip(torque, self.torque_low, self.torque_high)
            sim.mj.mj_step(model, data)
        rotation = data.xmat[self.base].reshape(3, 3)
        upright = rotation[2, 2]
        fallen = upright < .80
        if self.stairs:
            reward, unsafe = self.stair_reward(upright)
        else:
            velocity = (rotation.T @ data.qvel[:3])[:2]
            xy_error = np.linalg.norm(velocity - command[:2])
            yaw_error = abs(data.qvel[5] - command[2])
            reward = .12 * np.exp(-(xy_error / .18) ** 2) + .08 * np.exp(-(yaw_error / .25) ** 2)
            reward -= .12 * min(xy_error, 1.) + .08 * min(yaw_error, 2.)
            unsafe = data.qpos[2] < .20
        terminated = bool(fallen or unsafe)
        self.tick += 1
        truncated = self.tick * self.dt >= (9. if self.stairs else 6.)
        if self.stress and self.tick == round(self.stress.push_time / self.dt):
            data.qvel[:2] += self.stress.push_delta_xy
            sim.mj.mj_forward(model, data)
        return self.observe(), float(reward - 2. * terminated), terminated, truncated, {}


class ExportActor(BaseCallback):
    def __init__(self, source, out, every, replay):
        super().__init__()
        self.source, self.out, self.every = source, out, every
        self.rollout = 0
        self.replay = replay
        self.replay_optimizer = None

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.rollout and self.replay:
            self.train_replay()
        if self.rollout % self.every == 0:
            self.export()
        self.rollout += 1

    def train_replay(self):
        policy = self.model.policy
        actor = torch.nn.Sequential(policy.mlp_extractor.policy_net, policy.action_net)
        if self.replay_optimizer is None:
            self.replay_optimizer = torch.optim.Adam(actor.parameters(), lr=2e-5)
        for _ in range(3):
            losses = []
            for features, actions in self.replay:
                indices = torch.randint(len(features), (128,))
                losses.append(torch.nn.functional.smooth_l1_loss(actor(features[indices]), actions[indices]))
            self.replay_optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            self.replay_optimizer.step()

    def export(self):
        policy = self.model.policy
        weights = dict(self.source)
        for index, layer in enumerate((*policy.mlp_extractor.policy_net[::2], policy.action_net)):
            weights[f"actor.{2 * index}.weight"] = layer.weight.detach().cpu()
            weights[f"actor.{2 * index}.bias"] = layer.bias.detach().cpu()
        weights["std"] = policy.log_std.detach().exp().cpu()
        path = self.out / f"model_{self.num_timesteps}.pt"
        torch.save({"model_state_dict": weights, "iter": 0}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        print(f"EXPORTED {path}", flush=True)

    def _on_training_end(self):
        if self.replay:
            self.train_replay()
        self.export()


def verified_reference_resets(results, references):
    env = OmniEnvironment(results, 0)
    accepted = []
    for seed, actions in references:
        env.reset(options={"scene_seed": seed})
        longest = 0
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            longest = max(longest, env.top_streak)
            if terminated or truncated:
                break
        if env.levels.min() == 5 and longest * env.dt >= .5:
            accepted.append((seed, actions))
        else:
            print(f"Reference reset excluded after replay check: {seed}", flush=True)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--envs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=449)
    parser.add_argument("--benchmark-demos", type=Path, nargs="+",
                        help="Replay successful stair demonstrations and flat anchor actions during PPO")
    parser.add_argument("--reference-reset-probability", type=float, default=0.,
                        help="Start this fraction of stair training episodes partway through a demonstration")
    parser.add_argument("--stair-speed-control", action="store_true",
                        help="Cap the stair progress bonus and penalize forward speed above 0.55 m/s")
    parser.add_argument("--stair-rise", type=float,
                        help="Fix the five-step curriculum rise in metres instead of cycling 6/8/10 cm")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.initial, map_location="cpu", weights_only=True)["model_state_dict"]
    if source["actor.0.weight"].shape[1] != 60:
        raise ValueError("Expected a 60-input omni actor")
    if not 0. <= args.reference_reset_probability < 1. or (args.reference_reset_probability and not args.benchmark_demos):
        parser.error("Reference reset probability must be in [0, 1) and requires benchmark demonstrations")
    if args.stair_rise is not None and not .03 <= args.stair_rise <= .15:
        parser.error("Stair rise must be between 0.03 and 0.15 metres")
    args.out.mkdir(parents=True, exist_ok=True)
    replay = []
    replay_seeds = []
    anchor_commands = []
    reference_resets = []
    if args.benchmark_demos:
        sim.ROOT = args.results.resolve()
        contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
        anchor, anchor_commands = motion_prior_anchor(
            numpy_actor(make_actor(source)), contract, sim.load_target_limits(contract),
        )
        replay.append(anchor)
        for path in args.benchmark_demos:
            train, dev, seeds = benchmark_data(path, contract["control_dt"], evaluation.gait_clock)
            if set(seeds) & set(replay_seeds):
                parser.error("Benchmark teacher archives must use disjoint episode seeds")
            replay.append(tensor_data(train, mirror=True))
            replay_seeds.extend(seeds)
            if args.reference_reset_probability:
                reset_seeds = seeds[:-1] if len(dev[0]) else seeds
                with np.load(path) as archive:
                    for seed in reset_seeds:
                        actions = archive["actions"][archive["seed"] == seed]
                        if len(actions) > 90:
                            reference_resets.append((seed, actions))
        print(json.dumps({"anchor_commands": anchor_commands, "benchmark_demo_seeds": replay_seeds}), flush=True)
    if args.reference_reset_probability:
        reference_resets = verified_reference_resets(args.results.resolve(), reference_resets)
        if not reference_resets:
            parser.error("Reference resets require at least one verified complete trajectory")
    (args.out / "run-config.json").write_text(json.dumps({
        "omni_motion_prior": True, "simulator": "MuJoCo", "initial": str(args.initial),
        "training_steps": args.steps, "seed": args.seed, "envs": args.envs,
        "replay_seeds": replay_seeds, "anchor_commands": anchor_commands,
        "reference_reset_probability": args.reference_reset_probability,
        "reference_reset_seeds": [seed for seed, _ in reference_resets],
        "stair_speed_control": args.stair_speed_control,
        "stair_only": True,
        "stair_rise_m": args.stair_rise,
    }, indent=2) + "\n")
    env = SubprocVecEnv([lambda rank=rank: OmniEnvironment(
                             args.results.resolve(), rank, reference_resets,
                             args.reference_reset_probability, args.stair_speed_control, args.stair_rise)
                         for rank in [0] * args.envs], start_method="fork")
    try:
        widths = [source[f"actor.{index}.weight"].shape[0] for index in (0, 2, 4)]
        trainer = PPO(
            "MlpPolicy", env, learning_rate=3e-5, n_steps=128, batch_size=128,
            n_epochs=4, gamma=.99, gae_lambda=.95, clip_range=.10, target_kl=.015,
            policy_kwargs={"activation_fn": torch.nn.ELU, "net_arch": dict(pi=widths, vf=widths),
                           "log_std_init": float(np.log(.12))},
            device="cpu", seed=args.seed, verbose=1,
        )
        policy = trainer.policy
        with torch.no_grad():
            for index, layer in enumerate((*policy.mlp_extractor.policy_net[::2], policy.action_net)):
                layer.weight.copy_(source[f"actor.{2 * index}.weight"])
                layer.bias.copy_(source[f"actor.{2 * index}.bias"])
        trainer.learn(total_timesteps=args.steps, callback=ExportActor(source, args.out, 10, replay),
                      log_interval=10)
    finally:
        env.close()


if __name__ == "__main__":
    main()
