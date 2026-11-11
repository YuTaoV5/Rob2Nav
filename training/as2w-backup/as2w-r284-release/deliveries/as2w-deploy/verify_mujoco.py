"""Replay fixed MuJoCo scenes with native C++ actions and motor targets."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import mujoco_screen as sim

PREFIX = "flat-screen-limited-no-self-seeded-friction-clip6-wheel16-angle12.5664"


def screen(binary, results, models, label, checkpoint, terrain):
    base = results / "fall-baseline"
    contract = json.loads((base / "fall-recovery-contract.json").read_text(encoding="utf-8"))
    limits = sim.load_target_limits(contract)
    state = torch.load(base / f"{checkpoint}.pt", weights_only=True, map_location="cpu")["model_state_dict"]
    suffix = "-stairs-dynamics-stress" if terrain else ""
    baseline = [json.loads(line) for line in (base / f"{PREFIX}{suffix}-{checkpoint}.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(baseline) != 16 or len({(r["pose"], r["seed"]) for r in baseline}) != 16:
        raise ValueError(f"Expected 16 distinct reference scenes for {label}")
    checked = []
    with subprocess.Popen([str(binary), "target-stream", str(models / f"{label}.bin")],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as runtime:
        def infer(raw, elapsed):
            values = np.r_[elapsed, raw].astype(np.float32)
            runtime.stdin.write(" ".join(format(x, ".9g") for x in values) + "\n")
            runtime.stdin.flush()
            line = runtime.stdout.readline()
            targets = np.fromstring(line, sep=" ")
            if targets.shape != (33,) or not np.isfinite(targets).all():
                raise RuntimeError("Invalid native target row")
            return targets[:16], targets[16:28], targets[28:32], targets[32]

        with torch.inference_mode():
            for prior in baseline:
                seed, pose = prior["seed"], prior["pose"]
                model = sim.load_flat_model(True, True, terrain, seed)
                stress = (sim.sample_dynamics_stress(seed, model.nbody - 1, 16) if terrain else None)
                row = sim.evaluate(model, contract, state, seed, sim.POSES.get(pose, 0.),
                                   limits=limits, action_clip=6., wheel_scale=16.,
                                   wheel_angle_period=4 * np.pi, dynamics_stress=stress,
                                   terrain=terrain, native_policy=infer)
                if not row["valid"]:
                    raise RuntimeError(f"Invalid simulation for {label} {pose}/{seed}")
                checked.append(dict(row, pose=pose, seed=seed, reference_passed=prior["passed"]))
        runtime.stdin.close()
        if runtime.wait():
            raise RuntimeError("Native runtime exited with an error")
    return {
        "trials": len(checked),
        "native_passed": sum(row["passed"] for row in checked),
        "reference_passed": sum(row["reference_passed"] for row in checked),
        "matching_pass_flags": sum(row["passed"] == row["reference_passed"] for row in checked),
        "rows": checked,
    }


def main():
    models = Path(__file__).resolve().parent / "models"
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("results", type=Path, help="as2w-results from the training handoff")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=manifest, action="append")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    output = {}
    for label in args.model or manifest:
        entry = manifest[label]
        checkpoint = Path(entry["source_checkpoint"]).stem
        terrain = "stairs" if entry["mode"] == "stairs" else None
        output[label] = screen(args.runtime.resolve(), sim.ROOT, models, label, checkpoint, terrain)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"} for k, v in output.items()}, indent=2))


if __name__ == "__main__":
    main()
