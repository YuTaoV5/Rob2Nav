import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

root = Path(sys.argv[1])
subprocess.run(["sha256sum", "-c", "--quiet", str(root / "inputs.sha256")], check=True)
mujoco = root / "baseline/mujoco"
isaac = root / "baseline/isaac"
onnx = root / "onnx"
checkpoint = root / "baseline/train/model_3999.pt"
artifacts = {
    "mujoco/mujoco-contract.json": mujoco / "mujoco-contract.json",
    "mujoco/mujoco-rolling-diagnostic.json": mujoco / "mujoco-rolling-diagnostic.json",
    "mujoco/mujoco-rolling-acceptance.json": mujoco / "mujoco-rolling-acceptance.json",
    "isaac/contract.json": isaac / "contract.json",
    "isaac/speed-evaluation.json": isaac / "speed-evaluation.json",
    "onnx/onnx-parity.json": onnx / "onnx-parity.json",
    "onnx/policy.onnx": onnx / "policy.onnx",
    "model_3999.pt": checkpoint,
    "run-contract.json": root / "run-contract.json",
    "inputs.sha256": root / "inputs.sha256",
}
for name, path in artifacts.items():
    if name.endswith(".json"):
        json.loads(path.read_bytes())
rows = json.loads(artifacts["mujoco/mujoco-rolling-diagnostic.json"].read_bytes())
cases = {"forward", "back", "turn_left", "turn_right", "arc_left", "arc_right"}
assert {r["scenario"] for r in rows} == cases
assert all(len({r["seed"] for r in rows if r["scenario"] == case}) == 12 for case in cases)
assert len(rows) == 72
checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
for name in ("mujoco/mujoco-contract.json", "isaac/contract.json", "onnx/onnx-parity.json"):
    assert json.loads(artifacts[name].read_bytes())["checkpoint_sha256"] == checkpoint_sha, name
archive = root / "filtered-4000-evidence.tar.gz"
partial = archive.with_suffix(".partial")
with tarfile.open(partial, "w:gz") as bundle:
    for name, path in artifacts.items():
        bundle.add(path, arcname=name, recursive=False)
partial.replace(archive)
print(hashlib.sha256(archive.read_bytes()).hexdigest(), archive)
