"""Export and verify r249's full flat-ground action pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
from torch import nn

from omni_motion_prior import OmniMotionPrior
from train_recovery_student import make_actor, numpy_actor
from train_stair_history_ab import OmniEnvironment


class FlatPolicy(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.prior = OmniMotionPrior()
        mask = torch.ones(16)
        mask[[2, 3, 6, 7, 10, 11]] = 0.
        self.register_buffer("hind_mask", mask)

    def forward(self, obs, elapsed):
        command = obs[:, :3] + obs.new_tensor((.4, 0., 0.))
        action = self.actor(obs).clamp(-6., 6.)
        action = self.prior(action, command, obs[:, 5:6], obs[:, 7:8], elapsed)
        low_speed = command[:, :2].square().sum(dim=1, keepdim=True) <= .20 ** 2
        return torch.where(low_speed, action * self.hind_mask, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = make_actor(torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"])
    raw_actor = numpy_actor(actor)
    policy = FlatPolicy(actor).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "policy.onnx"
    torch.onnx.export(policy, (torch.zeros(1, 60), torch.zeros(1)), str(path),
                      input_names=["obs", "elapsed"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    model = onnx.load(path)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    runner = ReferenceEvaluator(model)
    rows = []
    fixture = []
    for seed in range(41001, 41009):
        reference = OmniEnvironment(args.assets, 1, low_speed_hind_scale=0.)
        deployed = OmniEnvironment(args.assets, 1)
        deployed.apply_motion_prior = False
        ref_obs, _ = reference.reset(seed=seed)
        actual_obs, _ = deployed.reset(seed=seed)
        max_error = 0.
        ref_track, actual_track = [], []
        for tick in range(300):
            elapsed = np.asarray([tick * deployed.dt], dtype=np.float32)
            action, = runner.run(None, {"obs": actual_obs[None, :], "elapsed": elapsed})
            cmd = reference.command.copy()
            actual_cmd = deployed.command.copy()
            actual_input = ref_obs.copy()
            expected, = runner.run(None, {"obs": ref_obs[None, :], "elapsed": elapsed})
            ref_obs, _, ref_term, ref_end, _ = reference.step(raw_actor(ref_obs))
            actual_obs, _, term, end, _ = deployed.step(action[0])
            # Replay identical inputs for parity independently of closed-loop drift.
            max_error = max(max_error, float(np.max(np.abs(expected[0] - reference.previous))))
            fixture.append(np.r_[actual_input, elapsed, reference.previous])
            for env, target, track in ((reference, cmd, ref_track), (deployed, actual_cmd, actual_track)):
                rotation = env.data.xmat[env.base].reshape(3, 3)
                track.append(float(np.linalg.norm((rotation.T @ env.data.qvel[:3])[:2] - target[:2])))
            if (term or end) or (ref_term or ref_end):
                break
        rows.append({"seed": seed, "frames": tick + 1, "max_action_error": max_error,
                     "reference_tracking_m_s": float(np.mean(ref_track)),
                     "onnx_tracking_m_s": float(np.mean(actual_track)),
                     "reference_terminated": ref_term, "onnx_terminated": term})
    (args.out / "closed-loop.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savetxt(args.out / "replay.txt", fixture, fmt="%.9g")
    if max(row["max_action_error"] for row in rows) > 1e-4:
        raise RuntimeError("Full flat-policy replay exceeded action parity tolerance")


if __name__ == "__main__":
    main()
