"""Finite-history stair experiment. Current 60 features keep the original ordering."""
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F


class History:
    # At 50 Hz, sample offsets 0, 2, ..., 24, covering 480 ms.
    offsets = tuple(range(2, 25, 2))
    size = 60 + len(offsets) * 55

    def __init__(self):
        self.frames = deque(maxlen=self.offsets[-1] + 1)

    def push(self, observation):
        frame = np.asarray(observation, dtype=np.float32)
        self.frames.append(frame[3:58].copy())
        past = [self.frames[-min(offset + 1, len(self.frames))] for offset in self.offsets]
        return np.concatenate([frame, *past])


class HistoryEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history = History()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (History.size,), np.float32)

    def reset(self, **kwargs):
        self.history.frames.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        return self.history.push(observation)


def lift_actor(state):
    weight = state["actor.0.weight"]
    return {**state, "actor.0.weight": F.pad(weight, (0, History.size - 60))}


def sequential_replay(path, dt, clock, *, mirror=False):
    from as2w_mirror_actor import reflect_action, reflect_observation

    with np.load(path) as archive:
        observations = archive["observations"]
        actions = archive["actions"]
        seeds = archive["seed"]
    unique = np.unique(seeds)
    if not len(unique) or observations.shape != (len(seeds), 58) or actions.shape != (len(seeds), 16):
        raise ValueError("Invalid demonstration archive")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite demonstration archive")
    features, targets = [], []
    for seed in unique[:-1] if len(unique) > 1 else unique:
        mask = seeds == seed
        raw = observations[mask]
        frames = np.column_stack((raw, *clock(np.arange(len(raw)) * dt))).astype(np.float32)
        for reflected in (False, True) if mirror else (False,):
            history = History()
            features.append(np.stack([history.push(reflect_observation(frame, True) if reflected else frame)
                                      for frame in frames]))
            target = actions[mask]
            targets.append(np.stack([reflect_action(action) for action in target]) if reflected else target)
    return tuple(torch.as_tensor(np.concatenate(items), dtype=torch.float32) for items in (features, targets))


def evaluate_history():
    import eval_stair_safety

    sim = eval_stair_safety.sim
    original = sim.evaluate
    original_load = torch.load
    selected = {}

    def load(*args, **kwargs):
        checkpoint = original_load(*args, **kwargs)
        state = checkpoint["model_state_dict"]
        if state["actor.0.weight"].shape[1] != History.size:
            raise ValueError("Unexpected history input size")
        selected.update(state)
        return {**checkpoint, "model_state_dict": {
            **state, "actor.0.weight": state["actor.0.weight"][:, :60].contiguous(),
        }}

    def evaluate(model, contract, state, *args, **kwargs):
        history = History()
        defaults = np.asarray(contract["default_joint_pos"])
        limits = kwargs["limits"]

        def plan(data, obs, elapsed, advance):
            x = torch.from_numpy(history.push(obs))
            for layer in (0, 2, 4, 6):
                x = F.linear(x, selected[f"actor.{layer}.weight"], selected[f"actor.{layer}.bias"])
                if layer != 6:
                    x = F.elu(x)
            action = np.clip(x.numpy(), -6., 6.)
            return (action, np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T),
                    np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2]))

        return original(model, contract, state, *args, native_planner=plan, **kwargs)

    torch.load = load
    sim.evaluate = evaluate
    try:
        eval_stair_safety.main()
    finally:
        torch.load = original_load
        sim.evaluate = original


if __name__ == "__main__":
    evaluate_history()
