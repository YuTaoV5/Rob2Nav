"""Append a zero-command wheel-angle reference to canonical AS2-W frames."""
import numpy as np


class ZeroWheelMemory:
    size = 65

    def __init__(self):
        self.reference = None

    def push(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        zero = not np.any(obs[:3] + np.array((.4, 0., 0.), dtype=np.float32))
        if not zero:
            self.reference = None
            return np.concatenate((obs, np.zeros(5, dtype=np.float32)))
        wheel = obs[22:26]
        if self.reference is None:
            self.reference = wheel.copy()
        delta = (wheel - self.reference + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
        return np.concatenate((obs, np.clip(delta / .20, -3., 3.), np.ones(1, dtype=np.float32)))
