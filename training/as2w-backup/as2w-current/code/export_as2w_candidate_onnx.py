"""Export a deterministic candidate actor and record ONNX Runtime parity."""
import argparse
import hashlib
import json
from pathlib import Path

from as2w_policy_provenance import checkpoint_record
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


def actor_from_state(state, clip):
    layers = sorted({int(key.split(".")[1]) for key in state if key.startswith("mlp.")})
    parameters = {f"mlp.{index}.{name}" for index in layers for name in ("weight", "bias")}
    extra = set(state) - parameters
    if any(not key.startswith("distribution.") for key in extra):
        raise ValueError(f"Unsupported actor state: {sorted(extra)}")
    if not layers:
        raise ValueError("Missing actor MLP")
    network = []
    for index in layers:
        weight, bias = (state[f"mlp.{index}.{name}"] for name in ("weight", "bias"))
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Non-finite actor parameter")
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict({"weight":weight, "bias":bias})
        network.append(layer)
        if index != layers[-1]:
            network.append(nn.ELU())
    network.append(nn.Hardtanh(-clip, clip))
    return nn.Sequential(*network).eval()


class ReflectedActor(nn.Module):
    def __init__(self,actor,kind):
        super().__init__()
        self.actor = actor
        if kind == "rolling":
            from as2w_rolling_fixed_exit import mirror_rolling_joints, mirror_rolling_observation
            self.mirror_obs = mirror_rolling_observation
            self.mirror_action = mirror_rolling_joints
        else:
            from as2w_recovery_control import mirror_legs, mirror_observation
            self.mirror_obs = mirror_observation
            self.mirror_action = mirror_legs

    def forward(self,obs):
        return .5*(self.actor(obs)+self.mirror_action(self.actor(self.mirror_obs(obs))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--clip", type=float, required=True)
    p.add_argument("--observations", type=Path)
    p.add_argument("--mirror",choices=("rolling","getup"))
    args = p.parse_args()
    torch.set_num_threads(1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = actor_from_state(ckpt["actor_state_dict"], args.clip)
    width = actor[0].in_features
    expected_width = {"rolling":65,"getup":42}
    if args.mirror:
        if width != expected_width[args.mirror]:
            p.error("Reflection mode does not match actor input width")
        actor = ReflectedActor(actor,args.mirror).eval()
    rng = np.random.default_rng(290)
    if args.observations:
        samples = np.load(args.observations, allow_pickle=False).astype(np.float32)
        if samples.ndim != 2 or samples.shape[1] != width or not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Invalid observation replay")
    else:
        samples = rng.normal(size=(128,width)).astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"policy.onnx"
    torch.onnx.export(actor, torch.zeros(1,width), str(path), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = np.concatenate([runtime.run(["actions"], {"obs": row[None]})[0] for row in samples])
    with torch.inference_mode():
        expected = actor(torch.from_numpy(samples)).numpy()
    error = float(np.abs(expected-actual).max())
    if error > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {error}")
    report = dict(**checkpoint_record(args.checkpoint), iteration=ckpt.get("iter"), opset=17,
                  input_shape=[1,width], output_shape=[1,actual.shape[1]], frames=len(samples),
                  onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  test_input="recorded_observations" if args.observations else "diagnostic_random_input",
                  max_action_error=error,mirror_policy=args.mirror)
    (args.output/"onnx-parity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()