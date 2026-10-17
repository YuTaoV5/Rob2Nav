"""Benchmark one completed AME pilot arm without changing the old benchmark."""
import argparse
import json
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--label", choices=["control", "attention"], required=True)
p.add_argument("--isaac-python", default="/root/isaacsim/python.sh")
p.add_argument("--mujoco-python", default="/root/gpufree-data/envs/go2-mapless/bin/python")
p.add_argument("--wait-for-training", action="store_true")
args = p.parse_args()
root = Path("outputs/ame")
train = root/args.label
deadline = time.monotonic()+1800
while not (train/"completed.json").exists():
    if not args.wait_for_training or time.monotonic() > deadline:
        raise RuntimeError("Training has not completed; no partial checkpoint accepted")
    time.sleep(10)
completion = json.loads((train/"completed.json").read_text())
assert completion["requested_iterations"] == 200 and completion["last_iteration"] == 199
checkpoint = train/"model_199.pt"
out = root/(args.label+"_eval"); out.mkdir(parents=True, exist_ok=True)
records = []


def run(command, name):
    with (out/(name+".log")).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    records.append({"command": command, "returncode": result.returncode})
    (out/"commands.json").write_text(json.dumps(records, indent=2))
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


for kind in ("stairs", "platform", "stones"):
    native = out/("native_"+kind)
    run([args.isaac_python, "-m", "perceptive_rl.evaluate", "--headless", "--fused",
         "--checkpoint", str(checkpoint), "--kind", kind, "--difficulty", ".5",
         "--num-envs", "32", "--seed", "801", "--output", str(native)], "native_"+kind)
    run([args.mujoco_python, "-m", "perceptive_rl.mujoco_compare", "--backend", "learned",
         "--actor", str(native/"actor.pt"), "--io", str(native/"summary.json"),
         "--kind", kind, "--difficulty", ".5", "--seed", "801", "--episodes", "5",
         "--video", "--output", str(out/("mujoco_"+kind))], "mujoco_"+kind)
if args.label == "attention":
    run([args.isaac_python, "-m", "perceptive_rl.evaluate", "--headless", "--fused",
         "--erase-attention", "--checkpoint", str(checkpoint), "--kind", "stones",
         "--difficulty", ".5", "--num-envs", "32", "--seed", "801",
         "--output", str(out/"native_stones_without_branch")], "native_stones_without_branch")
(out/"completed.json").write_text(json.dumps({"commands": len(records), "all_passed": True}))
print("BENCHMARK_COMPLETE", args.label, flush=True)
