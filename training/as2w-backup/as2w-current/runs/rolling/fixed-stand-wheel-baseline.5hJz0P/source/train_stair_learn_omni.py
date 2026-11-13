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
                 stair_speed_control=False, stair_rise=None, terminal_penalty=2., top_reward=.20,
                 stair_rises=None, flat_posture_weight=.0, flat_pitch_weight=.0,
                 flat_contact_penalty=.0, flat_command_max=.15,
                 flat_command_interval=.8, flat_command_alpha=.12,
                 low_speed_hind_scale=1., stop_training=False, low_speed_course=False,
                 stair_approach_curriculum=False, learn_omni=False):
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
        self.stair_rises = stair_rises
        self.terminal_penalty = terminal_penalty
        self.top_reward = top_reward
        self.flat_posture_weight = flat_posture_weight
        self.flat_pitch_weight = flat_pitch_weight
        self.flat_contact_penalty = flat_contact_penalty
        self.flat_command_max = flat_command_max
        self.flat_command_interval = flat_command_interval
        self.flat_command_alpha = flat_command_alpha
        self.low_speed_hind_scale = low_speed_hind_scale
        self.stop_training = stop_training
        self.low_speed_course = low_speed_course
        self.stair_approach_curriculum = stair_approach_curriculum
        self.learn_omni = learn_omni
        self.apply_motion_prior = not learn_omni
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
        active_stair_rise = self.stair_rise
        if self.stairs and self.stair_rises:
            active_stair_rise = self.stair_rises[scene_seed % len(self.stair_rises)]
        model = self.model = sim.load_flat_model(
            True, True, "stairs" if self.stairs else None, scene_seed, stair_rise=active_stair_rise)
        data = self.data = sim.mj.MjData(model)
        names = self.contract["joint_names"]
        joints = [model.joint(name) for name in names]
        self.qa = [joint.qposadr[0] for joint in joints]
        self.va = [joint.dofadr[0] for joint in joints]
        self.motors = [model.actuator(name).id for name in names]
        self.base = model.body("base_link").id
        self.wheel_index = {model.body(name).id: index for index, name in enumerate(names[12:])}
        ground = model.geom("ground").id
        self.ground = ground
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
            self.ascent_speed = float(options.get("ascent_speed", .4))
            if self.low_speed_course and "ascent_speed" not in options:
                self.ascent_speed = float(self.np_random.choice([.10, .15, .22, .35], p=[.30, .25, .25, .20]))
            self.target = np.array((self.ascent_speed, 0., 0.))
            self.stair_seconds = max(9., 3.2 / self.ascent_speed + 3.) if self.low_speed_course else 9.
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
            linear = np.linalg.norm(self.target[:2])
            if linear > self.flat_command_max:
                self.target[:2] *= self.flat_command_max / linear
            self.target[2] = np.clip(self.target[2], -.5, .5)
            self.target_goal = self.target.copy()
            self.target = np.zeros(3)
        data.qpos[2] = .37
        sim.mj.mj_forward(model, data)
        self.tick = 0
        self.stair_hold_started = False
        self.stair_hold_until = 0
        self.stair_hold_enabled = self.stairs and self.stop_training and self.np_random.random() < .35
        self.next_command_tick = max(1, round(self.flat_command_interval / self.dt))
        self.previous = np.zeros(16)
        self.force = np.zeros(6)
        self.levels = np.zeros(4, dtype=int)
        self.top_streak = 0
        self.top_credited = False
        self.wheel_transfer_potential = 0.
        if self.stairs and self.stair_approach_curriculum:
            self.wheel_bodies = sorted(self.wheel_index, key=self.wheel_index.get)
            stairs = [model.geom(f"stair_{level}") for level in range(1, 6)]
            self.stair_edges = np.array([g.pos[0] - g.size[0] for g in stairs])
            self.stair_tops = np.array([g.pos[2] + g.size[2] for g in stairs])
            if options.get("scene_seed") is None and self.np_random.random() < .45:
                # Randomized ground start close to the first riser, with no overlap.
                data.qpos[0] = self.np_random.uniform(.20, .32)
                sim.mj.mj_forward(model, data)
                front_edge = data.xpos[self.wheel_bodies, 0].max() + .086
                data.qpos[0] -= max(0., front_edge - self.stair_edges[0] + .02)
                sim.mj.mj_forward(model, data)
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
        return self.target

    def update_flat_command(self):
        if self.stairs:
            return
        if self.tick >= self.next_command_tick:
            # A real remote produces changing low-speed commands. Include an explicit
            # stop state and concentrate the distribution around the 0.1 m/s test speed.
            if self.np_random.random() < .25:
                self.target_goal = np.zeros(3)
            elif self.learn_omni and self.np_random.random() < .30:
                # Make pure signed yaw identifiable independently of sideways motion.
                yaw = self.np_random.choice([-1., 1.]) * self.np_random.uniform(.15, .45)
                self.target_goal = np.array((0., 0., yaw))
            else:
                speed = self.np_random.uniform(.04, self.flat_command_max)
                heading = self.np_random.uniform(-np.pi, np.pi)
                self.target_goal = np.array((speed * np.cos(heading), speed * np.sin(heading),
                                             self.np_random.uniform(-.5, .5)))
            if self.stop_training and not np.any(self.target_goal):
                dwell = self.np_random.uniform(1.2, 2.5)
            else:
                dwell = self.np_random.uniform(.5, 1.5) * self.flat_command_interval
            self.next_command_tick += max(1, round(dwell / self.dt))
        self.target += self.flat_command_alpha * (self.target_goal - self.target)
        if self.stop_training and not np.any(self.target_goal) and np.max(np.abs(self.target)) < .005:
            self.target.fill(0.)

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
        reward = (8. * progress + .02 * tracking + 2. * newly_reached + self.top_reward * top
                  + 4. * (self.top_streak == round(.5 / self.dt))
                  - .10 * abs(data.qpos[1]) - .010 * max(impact - 15., 0.)
                  - .80 * excess_speed - .03 * stalled)
        if self.low_speed_course:
            requested = self.command[0]
            excess = max(0., forward_speed - requested - .07)
            lateral_speed = data.xmat[self.base, [1, 4, 7]] @ data.qvel[:3]
            stable_top = self.top_streak >= round(.5 / self.dt) and not self.top_credited
            self.top_credited |= stable_top
            speed_error = abs(forward_speed - requested)
            reward = (.16 * np.exp(-(speed_error / .09) ** 2)
                      - .4 * min(speed_error, 1.) + 4. * min(progress, requested * self.dt)
                      + .8 * newly_reached + 1.5 * stable_top
                      - .50 * excess - .20 * abs(lateral_speed)
                      - .12 * abs(data.qpos[1]) - .012 * max(impact - 12., 0.))
        if self.stair_approach_curriculum:
            next_level = np.minimum(self.levels, 4)
            wheel = data.xpos[self.wheel_bodies]
            rise = self.stair_tops[0]
            previous_top = self.stair_tops[next_level] - rise
            proximity = np.clip((wheel[:, 0] - self.stair_edges[next_level] + .20) / .24, 0., 1.)
            clearance = np.clip((wheel[:, 2] - previous_top - .086) / rise, 0., 1.)
            potential = float(np.sum(self.levels + (self.levels < 5) * proximity * clearance))
            reward += .55 * (potential - self.wheel_transfer_potential)
            self.wheel_transfer_potential = potential
        if self.stop_training and not np.any(self.command):
            # During a requested stop, movement up or down the stair is an error.
            speed = np.linalg.norm(data.qvel[:2])
            reward = (.16 * np.exp(-(speed / .045) ** 2)
                      - .20 * min(speed / .15, 1.)
                      - .010 * max(impact - 15., 0.) - .10 * (1. - upright))
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
        if not self.stairs and np.linalg.norm(command[:2]) <= .20:
            action = np.asarray(action).copy()
            action[[2, 3, 6, 7, 10, 11]] *= self.low_speed_hind_scale
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
            if self.learn_omni:
                reward += .10 * np.exp(-(yaw_error / .15) ** 2) - .18 * min(yaw_error, 1.)
                reward -= .20 * min(xy_error, 1.)
            # Keep the hind legs near the symmetric nominal stance on flat ground.
            # Joint order is grouped by type, so these are RL/RR hip, thigh and calf.
            q_error = data.qpos[self.qa] - self.defaults
            hind_error = q_error[[2, 3, 6, 7, 10, 11]]
            reward -= self.flat_posture_weight * float(np.mean(hind_error ** 2))
            reward -= self.flat_pitch_weight * float(rotation[2, 0] ** 2)
            nonwheel_contact = False
            for contact in data.contact:
                surface, robot_geom = contact.geom
                if surface != self.ground:
                    surface, robot_geom = robot_geom, surface
                if surface == self.ground and self.model.geom_bodyid[robot_geom] not in self.wheel_index:
                    nonwheel_contact = True
                    break
            reward -= self.flat_contact_penalty * nonwheel_contact
            if self.stop_training and not np.any(command):
                # Simulator velocity is used only to score residual motion.
                yaw_rate = abs(data.qvel[5])
                speed = np.linalg.norm(velocity)
                reward += .18 * np.exp(-((speed / .045) ** 2 + (yaw_rate / .10) ** 2))
                reward -= .12 * min(speed / .15, 1.) + .06 * min(yaw_rate / .3, 1.)
            unsafe = data.qpos[2] < .20
        terminated = bool(fallen or unsafe)
        self.tick += 1
        self.update_flat_command()
        if self.stair_hold_enabled:
            if (not self.stair_hold_started and self.levels.max() >= 1
                    and self.tick * self.dt < self.stair_seconds - 3.5):
                self.stair_hold_started = True
                self.stair_hold_until = self.tick + round(1.5 / self.dt)
                self.target.fill(0.)
            elif self.stair_hold_started and self.tick == self.stair_hold_until:
                self.target[0] = self.ascent_speed
        truncated = self.tick * self.dt >= (self.stair_seconds if self.stairs else 6.)
        if self.stress and self.tick == round(self.stress.push_time / self.dt):
            data.qvel[:2] += self.stress.push_delta_xy
            sim.mj.mj_forward(model, data)
        return self.observe(), float(reward - self.terminal_penalty * terminated), terminated, truncated, {}


class ExportActor(BaseCallback):
    def __init__(self, source, out, every, replay, replay_every=1, replay_lr=2e-5, replay_epochs=3):
        super().__init__()
        self.source, self.out, self.every = source, out, every
        self.rollout = 0
        self.replay = replay
        self.replay_every = replay_every
        self.replay_lr = replay_lr
        self.replay_epochs = replay_epochs
        self.replay_optimizer = None

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.rollout and self.replay and self.rollout % self.replay_every == 0:
            self.train_replay()
        if self.rollout % self.every == 0:
            self.export()
        self.rollout += 1

    def train_replay(self):
        policy = self.model.policy
        actor = torch.nn.Sequential(policy.mlp_extractor.policy_net, policy.action_net)
        if self.replay_optimizer is None:
            self.replay_optimizer = torch.optim.Adam(actor.parameters(), lr=self.replay_lr)
        for _ in range(self.replay_epochs):
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


def verified_reference_resets(results, references, stair_rise, terminal_penalty, top_reward):
    env = OmniEnvironment(results, 0, stair_speed_control=True, stair_rise=stair_rise,
                          terminal_penalty=terminal_penalty, top_reward=top_reward)
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
    parser.add_argument("--stair-rises", type=float, nargs="+",
                        help="Seeded mixture of five-step rises; repeated values set the ratio")
    parser.add_argument("--terminal-penalty", type=float, default=2.,
                        help="Reward subtracted on unsafe or fallen termination")
    parser.add_argument("--top-reward", type=float, default=.20,
                        help="Dense reward per control step with stable four-wheel top support")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--checkpoint-rollouts", type=int, default=10,
                        help="Export every N PPO rollouts")
    parser.add_argument("--replay-every", type=int, default=1,
                        help="Apply supervised replay every N PPO rollouts")
    parser.add_argument("--replay-learning-rate", type=float, default=2e-5,
                        help="Learning rate for supervised teacher replay")
    parser.add_argument("--replay-epochs", type=int, default=3,
                        help="Gradient epochs per supervised replay update")
    parser.add_argument("--flat-fraction", type=float, default=0.,
                        help="Fraction of PPO environments using flat command tracking instead of stairs")
    parser.add_argument("--flat-posture-weight", type=float, default=0.,
                        help="Flat-ground hind-joint nominal-posture penalty weight")
    parser.add_argument("--flat-pitch-weight", type=float, default=0.,
                        help="Flat-ground body-pitch penalty weight")
    parser.add_argument("--flat-contact-penalty", type=float, default=0.,
                        help="Penalty per control step for non-wheel ground contact on flat terrain")
    parser.add_argument("--flat-command-max", type=float, default=.15,
                        help="Maximum planar speed for dynamic flat-ground remote commands")
    parser.add_argument("--flat-command-interval", type=float, default=.8,
                        help="Mean seconds between dynamic flat-ground remote command goals")
    parser.add_argument("--flat-command-alpha", type=float, default=.12,
                        help="Per-control-step smoothing factor for dynamic remote commands")
    parser.add_argument("--low-speed-hind-scale", type=float, default=1.,
                        help="Low-speed flat-ground scale for hind hip/thigh/calf actions")
    parser.add_argument("--history", action="store_true",
                        help="Add a 480 ms window of past proprioception to the actor")
    parser.add_argument("--stop-training", action="store_true",
                        help="Train sustained zero-command holds on flat ground and interrupted ascents")
    parser.add_argument("--low-speed-course", action="store_true",
                        help="Sample slower ascent commands and use command-relative rewards and time limits")
    parser.add_argument("--stair-approach-curriculum", action="store_true",
                        help="Mix closer ground starts and reward wheel transfer near each riser")
    parser.add_argument("--learn-omni", action="store_true",
                        help="Disable action prior and train explicit pure-yaw command segments")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.initial, map_location="cpu", weights_only=True)["model_state_dict"]
    if source["actor.0.weight"].shape[1] != 60:
        raise ValueError("Expected a 60-input omni actor")
    if args.history:
        if args.benchmark_demos:
            parser.error("History requires sequential replay; frame replay is unsupported")
        from as2w_history import HistoryEnv, lift_actor
        source = lift_actor(source)
    if not 0. <= args.reference_reset_probability < 1. or (args.reference_reset_probability and not args.benchmark_demos):
        parser.error("Reference reset probability must be in [0, 1) and requires benchmark demonstrations")
    if args.stair_rise is not None and not .03 <= args.stair_rise <= .15:
        parser.error("Stair rise must be between 0.03 and 0.15 metres")
    if args.stair_rise is not None and args.stair_rises:
        parser.error("Use either --stair-rise or --stair-rises, not both")
    if args.stair_rises and any(not .03 <= rise <= .15 for rise in args.stair_rises):
        parser.error("Every mixed stair rise must be between 0.03 and 0.15 metres")
    if args.terminal_penalty < 0. or args.learning_rate <= 0.:
        parser.error("Terminal penalty must be non-negative and learning rate must be positive")
    if args.top_reward < 0.:
        parser.error("Top support reward must be non-negative")
    if args.checkpoint_rollouts < 1:
        parser.error("Checkpoint rollout interval must be positive")
    if args.replay_every < 1 or args.replay_epochs < 1 or args.replay_learning_rate <= 0.:
        parser.error("Replay interval, epochs, and learning rate must be positive")
    if not 0. <= args.flat_fraction <= 1.:
        parser.error("Flat fraction must be in [0, 1]")
    if min(args.flat_posture_weight, args.flat_pitch_weight, args.flat_contact_penalty) < 0.:
        parser.error("Flat posture, pitch and contact weights must be non-negative")
    if args.flat_command_max < .10 or args.flat_command_interval <= 0. or not 0. < args.flat_command_alpha <= 1.:
        parser.error("Flat command max must include 0.1 m/s; interval and alpha must be positive")
    if not 0. <= args.low_speed_hind_scale <= 1.:
        parser.error("Low-speed hind scale must be in [0, 1]")
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
        reference_resets = verified_reference_resets(
            args.results.resolve(), reference_resets, args.stair_rise,
            args.terminal_penalty, args.top_reward)
        if not reference_resets:
            parser.error("Reference resets require at least one verified complete trajectory")
    (args.out / "run-config.json").write_text(json.dumps({
        "omni_motion_prior": True, "simulator": "MuJoCo", "initial": str(args.initial),
        "training_steps": args.steps, "seed": args.seed, "envs": args.envs,
        "history": args.history,
        "stop_training": args.stop_training,
        "low_speed_course": args.low_speed_course,
        "stair_approach_curriculum": args.stair_approach_curriculum,
        "learn_omni": args.learn_omni,
        "replay_seeds": replay_seeds, "anchor_commands": anchor_commands,
        "reference_reset_probability": args.reference_reset_probability,
        "reference_reset_seeds": [seed for seed, _ in reference_resets],
        "stair_speed_control": args.stair_speed_control,
        "stair_only": args.flat_fraction == 0.,
        "flat_fraction": args.flat_fraction,
        "flat_posture_weight": args.flat_posture_weight,
        "flat_pitch_weight": args.flat_pitch_weight,
        "flat_contact_penalty": args.flat_contact_penalty,
        "flat_command_max": args.flat_command_max,
        "flat_command_interval": args.flat_command_interval,
        "flat_command_alpha": args.flat_command_alpha,
        "low_speed_hind_scale": args.low_speed_hind_scale,
        "stair_rise_m": args.stair_rise,
        "stair_rises_m": args.stair_rises,
        "terminal_penalty": args.terminal_penalty,
        "top_reward": args.top_reward,
        "learning_rate": args.learning_rate,
        "checkpoint_rollouts": args.checkpoint_rollouts,
        "replay_every": args.replay_every,
        "replay_learning_rate": args.replay_learning_rate,
        "replay_epochs": args.replay_epochs,
    }, indent=2) + "\n")
    flat_envs = round(args.envs * args.flat_fraction)
    ranks = [1] * flat_envs + [0] * (args.envs - flat_envs)
    def create_env(rank):
        env = OmniEnvironment(
                             args.results.resolve(), rank, reference_resets,
                             args.reference_reset_probability, args.stair_speed_control, args.stair_rise,
                             args.terminal_penalty, args.top_reward, args.stair_rises,
                             args.flat_posture_weight, args.flat_pitch_weight,
                             args.flat_contact_penalty, args.flat_command_max,
                             args.flat_command_interval, args.flat_command_alpha,
                             args.low_speed_hind_scale, args.stop_training, args.low_speed_course,
                             args.stair_approach_curriculum, args.learn_omni)
        return HistoryEnv(env) if args.history else env

    env = SubprocVecEnv([lambda rank=rank: create_env(rank) for rank in ranks], start_method="fork")
    try:
        widths = [source[f"actor.{index}.weight"].shape[0] for index in (0, 2, 4)]
        trainer = PPO(
            "MlpPolicy", env, learning_rate=args.learning_rate, n_steps=128, batch_size=128,
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
        trainer.learn(total_timesteps=args.steps,
                      callback=ExportActor(source, args.out, args.checkpoint_rollouts, replay,
                                           args.replay_every, args.replay_learning_rate,
                                           args.replay_epochs),
                      log_interval=10)
    finally:
        env.close()


if __name__ == "__main__":
    main()

