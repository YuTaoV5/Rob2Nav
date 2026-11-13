"""Asymmetric PPO inputs for the AS2-W stair simulator."""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn


ACTOR_SIZE = 60
CRITIC_EXTRA = 21


class PrivilegedCriticEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if len(env.observation_space.shape) != 1 or env.observation_space.shape[0] < ACTOR_SIZE:
            raise ValueError("Expected the canonical actor prefix")
        actor_size = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (actor_size + CRITIC_EXTRA,), np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        env = self.env.unwrapped
        self.wheels = sorted(env.wheel_index, key=env.wheel_index.get)
        if env.stairs:
            stairs = [env.model.geom(f"stair_{level}") for level in range(1, 6)]
            self.edges = np.asarray([g.pos[0] - g.size[0] for g in stairs])
            self.tops = np.asarray([g.pos[2] + g.size[2] for g in stairs])
        return self.observation(obs), info

    def observation(self, obs):
        env = self.env.unwrapped
        data = env.data
        rotation = data.xmat[env.base].reshape(3, 3)
        wheel = data.xpos[self.wheels]
        edge_distance = np.zeros(4)
        clearance = wheel[:, 2] - .086
        levels = np.zeros(4)
        if env.stairs:
            next_level = np.minimum(env.levels, 4)
            edge_distance = np.clip(self.edges[next_level] - wheel[:, 0] - .086, -.5, .5)
            clearance -= self.tops[next_level]
            levels = env.levels / 5.
        extra = np.r_[rotation.T @ data.qvel[:3], data.qpos[2],
                      data.qpos[0] - env.start_x, data.qpos[1],
                      edge_distance, clearance, levels, float(env.stairs),
                      min(getattr(env, "last_stair_impact", 0.) / 90., 2.) if env.tick and env.stairs else 0.,
                      min(env.top_streak * env.dt, 1.)]
        return np.concatenate((obs, extra.astype(np.float32)))


class AsymmetricExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        if not isinstance(self.policy_net[0], nn.Linear):
            raise ValueError("Actor must start with a linear layer")
        self.actor_size = feature_dim - CRITIC_EXTRA
        self.policy_net[0] = nn.Linear(self.actor_size, self.policy_net[0].out_features, device=device)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_size])


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device)
