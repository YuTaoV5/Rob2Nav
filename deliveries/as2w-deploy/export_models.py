"""Package the selected AS2-W actors and a reference trace for native C++."""

import argparse
import hashlib
import importlib.util
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODELS = {
    "recovery-r09": (0, "righting-streak-r09-model_7000.pt"),
    "recovery-r12": (0, "righting-conservative-r12-model_1999.pt"),
    "recovery-r13": (0, "righting-mixed-terrain-r13-model_1999.pt"),
    "stairs-r14": (1, "blind-stairs-r14-model_1999.pt"),
    "stairs-r17-2000": (1, "blind-stairs-imitation-r17-model_1999.pt"),
    "stairs-r17-3000": (1, "blind-stairs-imitation-r17-to5000-model_2999.pt"),
}
WIDTHS = (58, 128, 128, 128, 16)


def floats(values):
    a = np.asarray(values, dtype="<f4")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite model parameters")
    return a.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="as2w-results from the verified handoff")
    parser.add_argument("output", type=Path)
    parser.add_argument("--onnx", action="store_true", help="Also export and verify static-batch ONNX actors")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("reference", args.results / "export-as2w-policy.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    contract = json.loads(reference.CONTRACT.read_text(encoding="utf-8"))
    names = contract["joint_names"]
    expected_names = [f"{leg}_{joint}" for joint in ("hip", "thigh", "calf", "foot")
                      for leg in ("FL", "FR", "RL", "RR")]
    if names != expected_names or contract["control_dt"] != .02:
        raise ValueError("Unexpected policy joint order or cadence")
    defaults = torch.tensor(contract["default_joint_pos"], dtype=torch.float32)
    urdf = ET.parse(reference.URDF)
    by_name = {joint.attrib["name"]: joint.find("limit") for joint in urdf.findall("joint")}
    limits = torch.tensor([[float(by_name[name].get(key, "0")) for key in ("lower", "upper", "velocity")]
                           for name in names], dtype=torch.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"joint_names": names, "control_period_s": .02, "models": {}}
    rng = np.random.default_rng(413)
    raw = rng.normal(0., .7, size=(60, 58)).astype(np.float32)
    raw[:, 10:22] += defaults[:12].numpy()
    raw[:, 22:26] *= 30.
    elapsed = np.linspace(0., 9., len(raw), dtype=np.float32)
    elapsed[:4] = [0., 1.98, 2., 6.]
    for label, (mode, filename) in MODELS.items():
        checkpoint = args.results / "fall-baseline" / filename
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
        payload = b"AS2WMLP1" + struct.pack("<I", mode)
        payload += floats(defaults) + floats(limits)
        layers = []
        for i, (width_in, width_out) in enumerate(zip(WIDTHS[:-1], WIDTHS[1:])):
            weight, bias = state[f"actor.{2*i}.weight"], state[f"actor.{2*i}.bias"]
            if weight.shape != (width_out, width_in) or bias.shape != (width_out,):
                raise ValueError(f"Unexpected actor architecture in {filename}")
            payload += floats(weight) + floats(bias)
            linear = nn.Linear(width_in, width_out)
            linear.load_state_dict({"weight": weight, "bias": bias})
            layers.append(linear)
            if i < 3:
                layers.append(nn.ELU())
        module = reference.AS2WPolicy(nn.Sequential(*layers), defaults, limits, bool(mode)).eval()
        with torch.inference_mode():
            action, leg, wheel, active = module(torch.from_numpy(raw), torch.from_numpy(elapsed))
        if not torch.isfinite(action).all():
            raise ValueError(f"Nonfinite reference output in {filename}")
        (args.output / f"{label}.bin").write_bytes(payload)
        fixture = np.column_stack((elapsed, raw, action, leg, wheel, active))
        np.savetxt(args.output / f"{label}.fixture", fixture, fmt="%.9g")
        manifest["models"][label] = {
            "mode": "stairs" if mode else "recovery",
            "source_checkpoint": filename,
            "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "actor_sha256": hashlib.sha256(payload).hexdigest(),
        }
        if args.onnx:
            import onnx
            from onnx.reference import ReferenceEvaluator

            target = args.output / f"{label}.onnx"
            actor = module.actor
            example = torch.zeros((1, WIDTHS[0]))
            torch.onnx.export(actor, example, str(target), input_names=["obs"], output_names=["actions"],
                              opset_version=17, dynamo=False)
            model = onnx.load(target)
            onnx.checker.check_model(model)
            runner = ReferenceEvaluator(model)
            with torch.inference_mode():
                for row in raw[:8]:
                    obs = row.reshape(1, -1)
                    [result] = runner.run(None, {"obs": obs})
                    np.testing.assert_allclose(result, actor(torch.from_numpy(obs)).numpy(),
                                               atol=3e-6, rtol=1e-5)
            manifest["models"][label]["onnx_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
