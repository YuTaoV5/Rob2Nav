"""Collect selected checkpoints, complete evaluations, and honest exploration logs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument("--source", default="outputs/perceptive")
p.add_argument("--output", default="artifacts/perceptive")
p.add_argument("--teacher", required=True)
p.add_argument("--student", required=True)
p.add_argument("--distillation-teacher", required=True)
args = p.parse_args()
source, dest = Path(args.source), Path(args.output)
dest.mkdir(parents=True, exist_ok=True)
for group in ["final_moe", "final_ik", "teacher_native", "student_native", "teacher_mujoco", "student_mujoco"]:
    commands = json.loads((source/group/"commands.json").read_text())
    if len(commands) != 3 or any(r["returncode"] != 0 for r in commands):
        raise RuntimeError(f"Incomplete benchmark: {group}")
groups = ["final_moe", "final_ik", "teacher_native", "student_native", "teacher_mujoco", "student_mujoco",
          "teacher_native_video", "student_native_video", "teacher_easy", "student_easy", "student_sensor_ablation", "distilled_native", "student_success_replay"]
copied = []
for name in groups:
    path = source/name
    if path.exists():
        shutil.copytree(path, dest/name, dirs_exist_ok=True)
        copied.append(name)
exploration = dest/"exploration"
exploration.mkdir(exist_ok=True)
for name in ["early_stairs", "scan200_native", "pretrained_native", "moe_suite", "ik_suite", "teacher200_platform",
             "pretrained_mujoco_flat", "pretrained_mujoco_flat_fixed"]:
    path = source/name
    if path.exists():
        # Keep raw summaries and traces; unsuccessful historical preview videos
        # are diagnostic artifacts and are not release demonstrations.
        shutil.copytree(path, exploration/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt", "*.mp4"))
if (source/"distill_save_failure").exists():
    shutil.copytree(source/"distill_save_failure", exploration/"distill_save_failure", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pt"))
for name in ["scan", "teacher", "distill", "student"]:
    path = source/name
    if path.exists():
        target = dest/"training"/name
        target.mkdir(parents=True, exist_ok=True)
        for pattern in ["*.json", "*.yaml", "events.out*", "source"]:
            for entry in path.glob(pattern):
                if entry.is_dir(): shutil.copytree(entry, target/entry.name, dirs_exist_ok=True)
                else: shutil.copy2(entry, target/entry.name)
log_dir = dest/"logs"; log_dir.mkdir(exist_ok=True)
for name in ["perceptive_boot", "teacher", "distill", "student", "distill_smoke", "distill_save_failure", "memory_smoke",
             "final_moe", "final_ik", "teacher_native", "student_native", "teacher_mujoco", "student_mujoco"]:
    log = source.parent/(name+".log")
    if log.exists(): shutil.copy2(log, log_dir/log.name)
if (source/"tests.log").exists(): shutil.copy2(source/"tests.log", log_dir/"tests.log")
for image in ["teacher_preview.jpg", "baseline_preview.jpg"]:
    if (source/image).exists(): shutil.copy2(source/image, dest/image)
if (source/"source_versions").exists():
    shutil.copytree(source/"source_versions", exploration/"source_versions", dirs_exist_ok=True)
weights = dest/"checkpoints"; weights.mkdir(exist_ok=True)
checkpoints = {"teacher.pt": Path(args.teacher), "student.pt": Path(args.student),
               "distillation_teacher.pt": Path(args.distillation_teacher), "scratch_300.pt": source/"scan/model_300.pt",
               "distilled_student.pt": source/"distill/student.pt"}
manifest = {"evaluation_groups": copied, "checkpoints": {}}
for name, path in checkpoints.items():
    shutil.copy2(path, weights/name)
    manifest["checkpoints"][name] = {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
changes = {}
for label, before, after in [
    ("teacher_PPO", source/"pretrained/checkpoint.pt", Path(args.teacher)),
    ("student_PPO", source/"distill/student.pt", Path(args.student))]:
    a = torch.load(before, map_location="cpu", weights_only=False)["model_state_dict"]
    b = torch.load(after, map_location="cpu", weights_only=False)["model_state_dict"]
    pairs = [(k, b[k]-v) for k, v in a.items() if k in b and b[k].shape == v.shape]
    changes[label] = {"changed_tensors": sum(bool(torch.any(d != 0)) for k, d in pairs),
        "compared_tensors": len(pairs),
        "parameter_delta_l2": float(torch.sqrt(sum((d.double()**2).sum() for k, d in pairs))),
        "per_tensor_l2": {k: float(torch.linalg.vector_norm(d)) for k, d in pairs}}
(dest/"parameter_updates.json").write_text(json.dumps(changes, indent=2))
manifest["external_initialization"] = {
    "url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt",
    "sha256": hashlib.sha256((source/"pretrained/checkpoint.pt").read_bytes()).hexdigest(),
    "description": "External NVIDIA initial rough policy, iteration 1499; not newly trained by Rob2Nav."}
(dest/"manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
