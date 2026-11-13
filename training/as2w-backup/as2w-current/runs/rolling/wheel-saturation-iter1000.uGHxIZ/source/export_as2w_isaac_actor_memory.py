"""Export a deterministic RSL-RL AS2-W actor to the MuJoCo evaluator format."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_recovery_student import make_actor, numpy_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = source["actor_state_dict"]
    layers = (0, 2, 4, 6)
    expected = {f"mlp.{layer}.{part}" for layer in layers for part in ("weight", "bias")}
    if set(actor) - expected - {key for key in actor if key.startswith("distribution.")}:
        raise ValueError("Unexpected actor parameters. Check the architecture and observation normalization.")
    state = {key.replace("mlp.", "actor.", 1): actor[key] for key in sorted(expected)}
    width = state["actor.0.weight"].shape[1]
    if width not in (60, 65) or state["actor.6.weight"].shape[0] != 16:
        raise ValueError("Unexpected AS2-W actor shape")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite actor parameters")
    samples = torch.randn(128, width, generator=torch.Generator().manual_seed(281))
    prediction = samples
    for layer in layers:
        prediction = F.linear(prediction, actor[f"mlp.{layer}.weight"], actor[f"mlp.{layer}.bias"])
        if layer != 6:
            prediction = F.elu(prediction)
    evaluate = numpy_actor(make_actor(state))
    actual = np.stack([evaluate(row) for row in samples.numpy()])
    np.testing.assert_allclose(actual, prediction.clamp(-6., 6.).numpy(), atol=1e-5, rtol=1e-5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "iter": source["iter"]}, args.output)
    print(f"Verified deterministic actor at iteration {source['iter']}")


if __name__ == "__main__":
    main()
