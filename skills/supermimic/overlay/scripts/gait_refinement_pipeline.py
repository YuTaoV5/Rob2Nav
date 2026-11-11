"""One bounded gait trial followed by independent strict and paired gait evaluations.

No automatic repeated training, promotion or production-model replacement.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--parent", type=Path, required=True)
parser.add_argument("--expected-parent-sha", required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--smoke-run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--run-name", default="velocity_gait_posture_v2_bounded_20261026")
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=33111)
parser.add_argument("--eval-seed-start", type=int, default=33120)
args = parser.parse_args()
source, parent, config, smoke, output = [p.resolve() for p in (args.source, args.parent, args.config, args.smoke_run, args.output)]
output.mkdir(parents=True, exist_ok=True)
lock = (output/"pipeline.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = output/"state.json"
if state_path.exists():
    raise SystemExit("Existing trial state: inspect it; never submit the same trial twice")
state = {"pipeline_pid": os.getpid(), "started_unix": time.time(), "capability_qualified": False}


def save(**updates):
    state.update(updates)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, log_name, phase):
    with (output/log_name).open("w") as log:
        process = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_GL="egl"))
        save(phase=phase, active_pid=process.pid, active_command=command)
        status = process.wait()
    if status:
        raise RuntimeError(f"{phase} failed with exit {status}; see {log_name}")


try:
    audit = json.loads((smoke/"gait_audit.json").read_text())
    contract = json.loads((smoke/"gait_contract.json").read_text())
    if sha(parent) != args.expected_parent_sha or Path(contract["cfg"]["resume"]).resolve() != parent:
        raise ValueError("Parent checkpoint differs from verified smoke")
    if not audit["training_completed"] or audit["shared_transition_reward_max_error"] >= 1e-5 or not audit["initialize_hold_targets"]:
        raise ValueError("Smoke audit failed")
    if contract["actor_dim"] != 350 or contract["actor_exteroception"] or contract["gait_settings"] != json.loads(config.read_text()):
        raise ValueError("Smoke task/settings/observation mismatch")
    for name, expected in contract["source_sha256"].items():
        path = source/("scripts/"+name if name == "train_gait.py" else name)
        if sha(path) != expected:
            raise ValueError(f"Source changed after smoke: {name}")
    # Inspect actual Python argv, not the text of shell commands containing a task name.
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            argv = (process_dir/"cmdline").read_bytes().split(b"\0")
            if len(argv) > 1 and b"python" in Path(os.fsdecode(argv[0])).name.encode() and Path(os.fsdecode(argv[1])).name in ("train_gait.py", "train_velocity.py"):
                raise ValueError(f"Existing training process {process_dir.name}; do not compete")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    save(phase="preflight_passed", parent_sha256=sha(parent), gait_settings=json.loads(config.read_text()),
         reset_source_sha256=sha(source/"rob2nav_reset_targets.py"), training_source_sha256=contract["source_sha256"])
    run_name = args.run_name
    command = [sys.executable, "scripts/train_gait.py", "--gait-config", str(config),
               "--task", "Go2-Velocity-Gait-V2", "--num-envs", "4096", "--iterations", str(args.iterations),
               "--buffer-len", "400", "--imag-batch", "4096", "--seed", str(args.seed),
               "--warmup-iterations", "200", "--critic-warmup-iterations", "100", "--actor-ramp-iterations", "200",
               "--lr", "0.00005", "--actor-lr-scale", "0.25", "--lr-decay-iterations", str(args.iterations), "--lr-final-frac", "0.3",
               "--resume", str(parent), "--reset-critic", "true", "--continue-run", "false",
               "--log-interval", "50", "--save-interval", "200", "--run-name", run_name]
    execute(command, "formal.log", "training")
    candidates = [p for p in (source/"logs/supermimic").glob("*"+run_name) if p.stat().st_mtime >= state["started_unix"]]
    if len(candidates) != 1:
        raise ValueError("Ambiguous or missing final run")
    run = candidates[0]
    checkpoint = run/f"model_{args.iterations}.pt"
    final_audit = json.loads((run/"gait_audit.json").read_text())
    actual = json.loads((run/"gait_contract.json").read_text())
    if not checkpoint.is_file() or not final_audit["training_completed"] or actual["gait_settings"] != state["gait_settings"] or actual["source_sha256"] != state["training_source_sha256"]:
        raise ValueError("Final checkpoint/audit/configuration mismatch")
    save(phase="training_completed", checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
    reports = []
    for seed in range(args.eval_seed_start, args.eval_seed_start+4):
        result = output/f"strict_{seed}.json"
        execute([sys.executable, "scripts/eval_velocity.py", "--checkpoint", str(checkpoint),
                 "--seed", str(seed), "--num-envs", "128", "--stage-seconds", "10",
                 "--initialize-hold-targets", "--audit-body-contact", "--output", str(result)], f"strict_{seed}.log", f"strict_evaluation_{seed}")
        report = json.loads(result.read_text())
        report["physical_gates_passed"] = (report["physics_sample_count"] == 16000 and report["falls_total"] == 0 and
             report["max_joint_velocity_ratio"] <= 1.00001 and report["max_joint_limit_excess_rad"] <= 1e-5 and
             report["max_joint_torque_ratio"] <= 1.00001 and report["max_target_rate_ratio"] <= 1.00001 and
             report["body_contact_audit"]["nonfoot_ground_contact_samples"] == 0 and not report["body_contact_audit"]["contact_buffer_overflow"])
        reports.append(report)
    execute([sys.executable, "scripts/eval_gait.py", "--checkpoint", str(checkpoint), "--seed", "33000",
             "--num-envs", "8", "--stage-seconds", "5", "--initialize-hold-targets",
             "--output", str(output/"paired_strict.json"), "--gait-output", str(output/"paired_gait.json")], "paired_gait.log", "paired_gait_evaluation")
    stem = f"velocity_gait_seed{args.seed}_model{args.iterations}"
    execute([sys.executable, "scripts/video_velocity_latest.py", "--checkpoint", str(checkpoint), "--seed", str(args.eval_seed_start+10),
             "--stage-seconds", "4", "--output-dir", str(output/"video"), "--label", f"{run_name} / model {args.iterations}",
             "--stem", stem], "video.log", "recording")
    summary = {"checkpoint": str(checkpoint), "independent_physical_gate_passes": sum(r["physical_gates_passed"] for r in reports),
               "independent_seeds": [r["seed"] for r in reports], "paired_gait": str(output/"paired_gait.json"),
               "video": str(output/"video"/(stem+".mp4")), "capability_qualified": False,
               "next": "Compare matched-seed gait and tracking against baseline; review full-speed video; diagnose first failure before another training"}
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    save(phase="completed_pending_review", active_pid=None, summary=summary)
except Exception as error:
    save(phase="failed", active_pid=None, error=str(error))
    raise
