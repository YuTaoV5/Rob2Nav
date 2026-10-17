"""CPU adapter for the unchanged NJU three/two-mode BBC deployment networks.

No camera inputs are invented for BBC: sensing belongs to the skill supervisor.
Observations and scaling follow go2_deploy/envs/lcm_agent.py at upstream 15d16ea.
"""
import importlib.util
import pickle
import types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Upstream stores a training normalizer alongside inference tensors.
        if module == "rsl_rl.utils.utils" and name in ("Normalizer", "RunningMeanStd"):
            return getattr(load_module("utils"), name)
        return super().find_class(module, name)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "vendor/NJUAgility/module" / (name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NJUBBC:
    def __init__(self, modes=3):
        self.modes = modes
        network = load_module("net_ll").ActorCritic
        self.model = network(96+modes, 666+modes, 12, 57, 10, 4, 29, 6+modes,
            actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
            priv_encoder_dims=[64], train_with_estimated_latent=True).eval()
        self.estimator = load_module("estimator").Estimator(57, 4, [128, 64]).eval()
        compatibility_pickle = types.SimpleNamespace(Unpickler=CheckpointUnpickler, load=pickle.load, __name__="pickle")
        checkpoint = torch.load(ROOT / f"vendor/NJUAgility/models/bbc{modes}.pt", map_location="cpu", weights_only=False,
                                pickle_module=compatibility_pickle)
        self.model.load_state_dict(checkpoint["actor_critic"], strict=True)
        self.estimator.load_state_dict(checkpoint["estimator"], strict=True)
        self.default = np.tile([0., .9, -1.8], 4)
        # Deployment config uses hip_scale_reduction=0.3, unlike BBC training.
        self.scale = np.tile([.075, .25, .25], 4)
        self.reset()

    def reset(self):
        self.history = None
        self.last_action = np.zeros(12)

    def targets(self, env, velocity, jump=False, height=.50, epsilon=0.):
        r = env.rotation
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2]))
        forces = np.zeros(4)
        # Upstream contacts use normal force >14N, centered at +/-0.5.
        import mujoco
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, i, wrench)
            for leg, foot in enumerate(env.foot_ids):
                if foot in (c.geom1, c.geom2):
                    forces[leg] += max(0., wrench[0])
        prop = np.r_[roll, pitch, env.data.sensor("imu_gyro").data*.25,
                     env.data.qpos[env.qids]-self.default, env.data.qvel[env.vids]*.05,
                     self.last_action, (forces > 14).astype(float)-.5, np.zeros(12)]
        self.history = np.tile(prop, (10, 1)) if self.history is None else np.vstack((self.history[1:], prop))
        mode = self.modes-1 if jump else min(env.cfg.get("bbc_walk_mode", 0), self.modes-2)
        cmd = np.r_[velocity, height if jump else 0., 0. if jump else .30,
                    epsilon, np.eye(self.modes)[mode]]
        p = torch.tensor(prop[None], dtype=torch.float32)
        with torch.inference_mode():
            explicit = self.estimator(p)
            obs = torch.cat((p, explicit, torch.zeros(1, 29),
                             torch.tensor(self.history.reshape(1, -1), dtype=torch.float32),
                             torch.tensor(cmd[None], dtype=torch.float32)), dim=1)
            action = self.model.act_inference(obs, hist_encoding=True).numpy()[0]
        self.last_action = action.copy()
        return self.default + self.scale*action, np.full(12, 40.), np.ones(12)
