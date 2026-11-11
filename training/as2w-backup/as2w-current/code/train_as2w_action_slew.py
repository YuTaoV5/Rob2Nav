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
                 stair_approach_curriculum=False, learn_omni=False, directional_flat=False,
                 rear_transfer_weight=0., stair_hold_probability=.35, stair_hold_stage_max=1,
                 top_support_shaping=0., flat_hind_envelope=0., stair_pair_balance=0.,
                 action_alpha=1.):
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
        self.directional_flat = directional_flat
        self.rear_transfer_weight = rear_transfer_weight
        self.stair_hold_probability = stair_hold_probability
        self.stair_hold_stage_max = stair_hold_stage_max
        self.top_support_shaping = top_support_shaping
        self.flat_hind_envelope = flat_hind_envelope
        self.stair_pair_balance = stair_pair_balance
        self.action_alpha = action_alpha
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
        reference_rise = None
        if (scene_seed is None and self.stairs and self.reference_resets
                and self.np_random.random() < self.reference_probability):
            scene_seed, reference, reference_rise = self.reference_resets[self.np_random.integers(len(self.reference_resets))]
        if scene_seed is None:
            scene_seed = int(self.np_random.integers(100_000, 1_000_000_000))
        rng = np.random.default_rng(scene_seed)
        active_stair_rise = options.get("stair_rise", reference_rise if reference_rise is not None else self.stair_rise)
        if self.stairs and self.stair_rises and reference_rise is None and "stair_rise" not in options:
            active_stair_rise = self.stair_rises[scene_seed % len(self.stair_rises)]
        if self.stairs and "platform_length" in options:
            from as2w_platform_model import load_platform_model
            model = load_platform_model(sim, scene_seed, active_stair_rise, float(options["platform_length"]))
        else:
            model = sim.load_flat_model(
                True, True, "stairs" if self.stairs else None, scene_seed, stair_rise=active_stair_rise)
        self.model = model
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
                self.ascent_speed = (.10 if reference is not None else
                                     float(self.np_random.choice([.10, .15, .22, .35], p=[.30, .25, .25, .20])))
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
        self.stair_hold_enabled = (self.stairs and self.stop_training and
                                   self.np_random.random() < self.stair_hold_probability)
        self.stair_hold_stage = (int(self.np_random.integers(1, self.stair_hold_stage_max + 1))
                                 if self.stair_hold_enabled and self.stair_hold_stage_max > 1 else 1)
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
            if reference is None and options.get("scene_seed") is None and self.np_random.random() < .45:
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
            upper = min(round((self.stair_seconds - 3.) / self.dt), len(reference) - 40)
            if not self.low_speed_course:
                upper = min(365, upper)
            anchor = self.np_random.integers(50, upper)
            enable_hold = self.stair_hold_enabled
            self.stair_hold_enabled = False
            for action in reference[:anchor]:
                _, _, terminated, truncated, _ = self.step(action, filter_action=False)
                if terminated or truncated:
                    fallback = int(self.np_random.integers(100_000, 1_000_000_000))
                    return self.reset(options={"scene_seed": fallback})
            self.stair_hold_enabled = enable_hold
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
            elif self.directional_flat:
                mode = self.np_random.integers(4)
                sign = self.np_random.choice([-1., 1.])
                speed = self.np_random.uniform(.07, self.flat_command_max)
                if mode == 0:
                    self.target_goal = np.array((sign * speed, 0., 0.))
                elif mode == 1:
                    self.target_goal = np.array((0., sign * speed, 0.))
                elif mode == 2:
                    self.target_goal = np.array((0., 0., sign * self.np_random.uniform(.20, .45)))
                else:
                    heading = self.np_random.uniform(-np.pi, np.pi)
                    self.target_goal = np.array((speed * np.cos(heading), speed * np.sin(heading),
                                                 self.np_random.uniform(-.30, .30)))
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
        rear_pair_progress = levels[2:].min() - self.levels[2:].min()
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
            wheel_progress = self.levels + (self.levels < 5) * proximity * clearance
            potential = float(wheel_progress.sum())
            if self.stair_pair_balance:
                # Give the lagging wheel credit for catching up on each axle.
                paired = 2. * wheel_progress.reshape(2, 2).min(axis=1).sum()
                potential += self.stair_pair_balance * (paired - potential)
            reward += .55 * (potential - self.wheel_transfer_potential)
            self.wheel_transfer_potential = potential
        reward += self.rear_transfer_weight * rear_pair_progress
        if self.top_support_shaping:
            support = np.clip(top_loads / 12., 0., 1.)
            upright_quality = np.clip((upright - .86) / .08, 0., 1.)
            # Prioritize the least loaded wheel without requiring a binary transition.
            reward += self.top_support_shaping * upright_quality * (.75 * support.min() + .25 * support.mean())
        if self.stop_training and not np.any(self.command):
            # During a requested stop, movement up or down the stair is an error.
            speed = np.linalg.norm(data.qvel[:2])
            reward = (.16 * np.exp(-(speed / .045) ** 2)
                      - .20 * min(speed / .15, 1.)
                      - .010 * max(impact - 15., 0.) - .10 * (1. - upright))
        self.last_stair_impact = impact
        return reward, impact > 90. or abs(data.qpos[1]) > .65

    def step(self, action, *, filter_action=True):
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
        if filter_action and self.action_alpha < 1.:
            previous = self.previous.astype(np.float32, copy=False)
            action = previous + self.action_alpha * (action - previous)
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
            if self.directional_flat and np.any(command):
                linear = np.linalg.norm(command[:2])
                if linear >= .025:
                    along = float(velocity @ (command[:2] / linear))
                    cross = abs(float((command[0] * velocity[1] - command[1] * velocity[0]) / linear))
                    reward += .20 * np.clip(along / linear, -1., 1.)
                    reward -= .13 * min(abs(along - linear) / .10, 2.) + .10 * min(cross / .10, 2.)
                else:
                    reward -= .20 * min(np.linalg.norm(velocity) / .10, 2.)
                if abs(command[2]) >= .08:
                    measured_yaw = float(data.qvel[5])
                    reward += .15 * np.clip(measured_yaw / command[2], -1., 1.)
                    reward -= .15 * min(yaw_error / .30, 2.)
            # Keep the hind legs near the symmetric nominal stance on flat ground.
            # Joint order is grouped by type, so these are RL/RR hip, thigh and calf.
            q_error = data.qpos[self.qa] - self.defaults
            hind_error = q_error[[2, 3, 6, 7, 10, 11]]
            reward -= self.flat_posture_weight * float(np.mean(hind_error ** 2))
            if self.flat_hind_envelope:
                limits = np.repeat((.28, .40, .45), 2)
                excess = np.maximum(np.abs(hind_error) - limits, 0.)
                reward -= self.flat_hind_envelope * float(np.max(excess) + .5 * np.mean(excess))
                if abs(command[1]) < .025 and abs(command[2]) < .08:
                    mismatch = np.array((hind_error[0] + hind_error[1], hind_error[2] - hind_error[3],
                                         hind_error[4] - hind_error[5]))
                    reward -= .35 * self.flat_hind_envelope * float(np.mean(np.abs(mismatch)))
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
            if (not self.stair_hold_started and self.levels.max() >= self.stair_hold_stage
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
    def __init__(self, source, out, every, replay, replay_every=1, replay_lr=2e-5, replay_epochs=3,
                 stop_demo_weight=1., bootstrap_updates=0):
        super().__init__()
        self.source, self.out, self.every = source, out, every
        self.rollout = 0
        self.replay = replay
        self.replay_every = replay_every
        self.replay_lr = replay_lr
        self.replay_epochs = replay_epochs
        self.bootstrap_updates = bootstrap_updates
        self.replay_optimizer = None
        self.replay_probabilities = []
        for features, _ in replay:
            neutral = features[:, :3] + features.new_tensor((.4, 0., 0.))
            weight = 1. + (stop_demo_weight - 1.) * (neutral.abs().amax(dim=1) < 1e-5).float()
            self.replay_probabilities.append(None if stop_demo_weight == 1. else weight)

    def _on_step(self):
        return True

    def _on_training_start(self):
        if self.bootstrap_updates:
            self.export("model_random_init.pt")
            self.train_replay(self.bootstrap_updates)
            print(f"Completed {self.bootstrap_updates} teacher updates before PPO rollout", flush=True)

    def _on_rollout_start(self):
        if self.rollout and self.replay and self.rollout % self.replay_every == 0:
            self.train_replay()
        if self.rollout % self.every == 0:
            self.export()
        self.rollout += 1

    def train_replay(self, updates=None):
        policy = self.model.policy
        actor = torch.nn.Sequential(policy.mlp_extractor.policy_net, policy.action_net)
        if self.replay_optimizer is None:
            self.replay_optimizer = torch.optim.Adam(actor.parameters(), lr=self.replay_lr)
        for _ in range(self.replay_epochs if updates is None else updates):
            losses = []
            for (features, actions), probabilities in zip(self.replay, self.replay_probabilities):
                indices = (torch.randint(len(features), (128,)) if probabilities is None else
                           torch.multinomial(probabilities, 128, replacement=True))
                losses.append(torch.nn.functional.smooth_l1_loss(actor(features[indices]), actions[indices]))
            self.replay_optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            self.replay_optimizer.step()

    def export(self, filename=None):
        policy = self.model.policy
        weights = dict(self.source)
        for index, layer in enumerate((*policy.mlp_extractor.policy_net[::2], policy.action_net)):
            weights[f"actor.{2 * index}.weight"] = layer.weight.detach().cpu()
            weights[f"actor.{2 * index}.bias"] = layer.bias.detach().cpu()
        weights["std"] = policy.log_std.detach().exp().cpu()
        path = self.out / (filename or f"model_{self.num_timesteps}.pt")
        torch.save({"model_state_dict": weights, "iter": 0}, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        print(f"EXPORTED {path}", flush=True)

    def _on_training_end(self):
        if self.replay:
            self.train_replay()
        self.export()


def verified_reference_resets(results, references, stair_rise, terminal_penalty, top_reward, low_speed):
    env = OmniEnvironment(results, 0, stair_speed_control=True, stair_rise=stair_rise,
                          terminal_penalty=terminal_penalty, top_reward=top_reward, low_speed_course=low_speed)
    accepted = []
    for seed, actions, rise in references:
        options = {"scene_seed": seed}
        if low_speed:
            options.update(ascent_speed=.10, stair_rise=rise)
        env.reset(options=options)
        longest = 0
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            longest = max(longest, env.top_streak)
            if terminated or truncated:
                break
        if env.levels.min() == 5 and longest * env.dt >= .5:
            accepted.append((seed, actions, rise))
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
    parser.add_argument("--flat-demos", type=Path, nargs="+",
                        help="Replay vetted flat demos without using them for stair reset states")
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
    parser.add_argument("--exploration-std", type=float, default=.12)
    parser.add_argument("--action-alpha", type=float, default=1.,
                        help="Blend the policy action with the previous applied action before motor target mapping")
    parser.add_argument("--stair-pair-balance", type=float, default=0.,
                        help="Blend individual wheel progress with progress of the lagging wheel per axle")
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--stop-demo-weight", type=float, default=1.)
    parser.add_argument("--stair-hold-probability", type=float, default=.35)
    parser.add_argument("--stair-hold-stage-max", type=int, default=1)
    parser.add_argument("--top-support-shaping", type=float, default=0.,
                        help="Dense reward for upright simultaneous wheel support on the platform")
    parser.add_argument("--privileged-critic", action="store_true",
                        help="Train the value function with simulation state hidden from the actor")
    parser.add_argument("--simulation-teacher", action="store_true",
                        help="Train a simulation-only actor with privileged velocity and geometry")
    parser.add_argument("--random-actor", action="store_true",
                        help="Use the checkpoint architecture but initialize actor weights from scratch")
    parser.add_argument("--checkpoint-rollouts", type=int, default=10,
                        help="Export every N PPO rollouts")
    parser.add_argument("--replay-every", type=int, default=1,
                        help="Apply supervised replay every N PPO rollouts")
    parser.add_argument("--replay-learning-rate", type=float, default=2e-5,
                        help="Learning rate for supervised teacher replay")
    parser.add_argument("--replay-epochs", type=int, default=3,
                        help="Gradient epochs per supervised replay update")
    parser.add_argument("--bootstrap-replay-updates", type=int, default=0,
                        help="Teacher updates before the first PPO rollout of a random student")
    parser.add_argument("--flat-fraction", type=float, default=0.,
                        help="Fraction of PPO environments using flat command tracking instead of stairs")
    parser.add_argument("--flat-posture-weight", type=float, default=0.,
                        help="Flat-ground hind-joint nominal-posture penalty weight")
    parser.add_argument("--flat-hind-envelope", type=float, default=0.,
                        help="Penalize flat hind joint extremes and sagittal-command asymmetry")
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
    parser.add_argument("--directional-flat", action="store_true",
                        help="Train signed directional response and penalize cross-axis drift")
    parser.add_argument("--rear-transfer-weight", type=float, default=0.,
                        help="Bonus for advancing the lower of the two rear wheel support levels")
    args = parser.parse_args()
    if not 0. < args.action_alpha <= 1.:
        parser.error("Action alpha must be in (0, 1]")
    if not 0. <= args.stair_pair_balance <= 1. or not np.isfinite(args.exploration_std) or args.exploration_std <= 0.:
        parser.error("Pair balance must be in [0, 1] and exploration std must be finite and positive")
    if args.simulation_teacher and (args.privileged_critic or args.history or args.benchmark_demos or args.flat_demos):
        parser.error("The simulation teacher requires the canonical inputs without replay")
    torch.set_num_threads(1)
    source = torch.load(args.initial, map_location="cpu", weights_only=True)["model_state_dict"]
    if source["actor.0.weight"].shape[1] != 60:
        raise ValueError("Expected a 60-input omni actor")
    if args.simulation_teacher:
        from as2w_privileged_critic import CRITIC_EXTRA
        source["actor.0.weight"] = torch.nn.functional.pad(source["actor.0.weight"], (0, CRITIC_EXTRA))
    if args.history:
        from as2w_history import HistoryEnv, lift_actor, sequential_replay
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
    if min(args.top_reward, args.top_support_shaping) < 0.:
        parser.error("Top support reward must be non-negative")
    if args.rear_transfer_weight < 0.:
        parser.error("Rear transfer reward must be non-negative")
    if not 0. < args.gamma < 1.:
        parser.error("Gamma must be in (0, 1)")
    if args.checkpoint_rollouts < 1:
        parser.error("Checkpoint rollout interval must be positive")
    if args.replay_every < 1 or args.replay_epochs < 1 or args.replay_learning_rate <= 0.:
        parser.error("Replay interval, epochs, and learning rate must be positive")
    if args.bootstrap_replay_updates < 0 or (args.bootstrap_replay_updates and
            (not args.random_actor or not (args.benchmark_demos or args.flat_demos))):
        parser.error("Replay bootstrapping requires a random student and demonstration archives")
    if args.stop_demo_weight < 1. or not 0. <= args.stair_hold_probability <= 1.:
        parser.error("Invalid stop demo weight or stair hold probability")
    if not 1 <= args.stair_hold_stage_max <= 5:
        parser.error("Stair hold stage must be between 1 and 5")
    if not 0. <= args.flat_fraction <= 1.:
        parser.error("Flat fraction must be in [0, 1]")
    if min(args.flat_posture_weight, args.flat_hind_envelope,
           args.flat_pitch_weight, args.flat_contact_penalty) < 0.:
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
    if args.benchmark_demos or args.flat_demos:
        sim.ROOT = args.results.resolve()
        contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    if args.benchmark_demos:
        if not args.learn_omni and not args.history:
            anchor, anchor_commands = motion_prior_anchor(
                numpy_actor(make_actor(source)), contract, sim.load_target_limits(contract),
            )
            replay.append(anchor)
        for path in args.benchmark_demos:
            train, dev, seeds = benchmark_data(path, contract["control_dt"], evaluation.gait_clock)
            if set(seeds) & set(replay_seeds):
                parser.error("Benchmark teacher archives must use disjoint episode seeds")
            replay.append(sequential_replay(path, contract["control_dt"], evaluation.gait_clock, mirror=True)
                          if args.history else tensor_data(train, mirror=True))
            replay_seeds.extend(seeds)
            if args.reference_reset_probability:
                reset_seeds = seeds[:-1] if len(dev[0]) else seeds
                with np.load(path) as archive:
                    for seed in reset_seeds:
                        mask = archive["seed"] == seed
                        actions = archive["actions"][mask]
                        if len(actions) > 90:
                            rise = None
                            if args.low_speed_course:
                                if "rise_m" not in archive:
                                    parser.error("Low-speed reference resets require per-episode rise_m")
                                heights = np.unique(archive["rise_m"][mask])
                                if len(heights) != 1:
                                    parser.error("A reference episode must have one rise")
                                rise = float(heights[0])
                            reference_resets.append((seed, actions, rise))
        print(json.dumps({"anchor_commands": anchor_commands, "benchmark_demo_seeds": replay_seeds}), flush=True)
    for path in args.flat_demos or ():
        train, _, seeds = benchmark_data(path, contract["control_dt"], evaluation.gait_clock)
        if set(seeds) & set(replay_seeds):
            parser.error("Flat and stair demonstration seeds must be disjoint")
        replay.append(sequential_replay(path, contract["control_dt"], evaluation.gait_clock)
                      if args.history else tensor_data(train))
        replay_seeds.extend(seeds)
    if args.reference_reset_probability:
        reference_resets = verified_reference_resets(
            args.results.resolve(), reference_resets, args.stair_rise,
            args.terminal_penalty, args.top_reward, args.low_speed_course)
        if not reference_resets:
            parser.error("Reference resets require at least one verified complete trajectory")
    (args.out / "run-config.json").write_text(json.dumps({
        "omni_motion_prior": not args.learn_omni, "simulator": "MuJoCo", "initial": str(args.initial),
        "training_steps": args.steps, "seed": args.seed, "envs": args.envs,
        "history": args.history,
        "stop_training": args.stop_training,
        "low_speed_course": args.low_speed_course,
        "stair_approach_curriculum": args.stair_approach_curriculum,
        "learn_omni": args.learn_omni,
        "directional_flat": args.directional_flat,
        "rear_transfer_weight": args.rear_transfer_weight,
        "flat_demo_archives": [str(path) for path in args.flat_demos or ()],
        "replay_seeds": replay_seeds, "anchor_commands": anchor_commands,
        "reference_reset_probability": args.reference_reset_probability,
        "reference_reset_seeds": [seed for seed, _, _ in reference_resets],
        "stair_speed_control": args.stair_speed_control,
        "stair_only": args.flat_fraction == 0.,
        "flat_fraction": args.flat_fraction,
        "flat_posture_weight": args.flat_posture_weight,
        "flat_hind_envelope": args.flat_hind_envelope,
        "stair_pair_balance": args.stair_pair_balance,
        "exploration_std": args.exploration_std,
        "action_alpha": args.action_alpha,
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
        "top_support_shaping": args.top_support_shaping,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "privileged_critic": args.privileged_critic,
        "simulation_teacher": args.simulation_teacher,
        "random_actor": args.random_actor,
        "checkpoint_rollouts": args.checkpoint_rollouts,
        "replay_every": args.replay_every,
        "replay_learning_rate": args.replay_learning_rate,
        "replay_epochs": args.replay_epochs,
        "bootstrap_replay_updates": args.bootstrap_replay_updates,
        "stop_demo_weight": args.stop_demo_weight,
        "stair_hold_probability": args.stair_hold_probability,
        "stair_hold_stage_max": args.stair_hold_stage_max,
    }, indent=2) + "\n")
    flat_envs = round(args.envs * args.flat_fraction)
    ranks = [1] * flat_envs + [0] * (args.envs - flat_envs)
    policy_class = "MlpPolicy"
    if args.privileged_critic or args.simulation_teacher:
        from as2w_privileged_critic import AsymmetricActorCriticPolicy, PrivilegedCriticEnv
    if args.privileged_critic:
        policy_class = AsymmetricActorCriticPolicy
    def create_env(rank):
        env = OmniEnvironment(
                             args.results.resolve(), rank, reference_resets,
                             args.reference_reset_probability, args.stair_speed_control, args.stair_rise,
                             args.terminal_penalty, args.top_reward, args.stair_rises,
                             args.flat_posture_weight, args.flat_pitch_weight,
                             args.flat_contact_penalty, args.flat_command_max,
                             args.flat_command_interval, args.flat_command_alpha,
                             args.low_speed_hind_scale, args.stop_training, args.low_speed_course,
                             args.stair_approach_curriculum, args.learn_omni, args.directional_flat,
                             args.rear_transfer_weight, args.stair_hold_probability, args.stair_hold_stage_max,
                             args.top_support_shaping, args.flat_hind_envelope, args.stair_pair_balance,
                             args.action_alpha)
        if args.history:
            env = HistoryEnv(env)
        if args.privileged_critic or args.simulation_teacher:
            return PrivilegedCriticEnv(env)
        return env

    env = SubprocVecEnv([lambda rank=rank: create_env(rank) for rank in ranks], start_method="fork")
    try:
        widths = [source[f"actor.{index}.weight"].shape[0] for index in (0, 2, 4)]
        trainer = PPO(
            policy_class, env, learning_rate=args.learning_rate, n_steps=128, batch_size=128,
            n_epochs=4, gamma=args.gamma, gae_lambda=.95, clip_range=.10, target_kl=.015,
            policy_kwargs={"activation_fn": torch.nn.ELU, "net_arch": dict(pi=widths, vf=widths),
                           "log_std_init": float(np.log(args.exploration_std))},
            device="cpu", seed=args.seed, verbose=1,
        )
        policy = trainer.policy
        if not args.random_actor:
            with torch.no_grad():
                for index, layer in enumerate((*policy.mlp_extractor.policy_net[::2], policy.action_net)):
                    layer.weight.copy_(source[f"actor.{2 * index}.weight"])
                    layer.bias.copy_(source[f"actor.{2 * index}.bias"])
        trainer.learn(total_timesteps=args.steps,
                      callback=ExportActor(source, args.out, args.checkpoint_rollouts, replay,
                                           args.replay_every, args.replay_learning_rate,
                                           args.replay_epochs, args.stop_demo_weight, args.bootstrap_replay_updates),
                      log_interval=10)
    finally:
        env.close()


if __name__ == "__main__":
    main()

