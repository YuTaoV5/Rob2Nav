"""After a verified gait trial, allow exactly one target-rate counterfactual.

The gate requires four completed zero-fall/body-contact/joint-safe results whose
remaining physical failure is target rate. It never repeats or promotes models.
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
parser.add_argument("--prior-trial", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--intervention", choices=("targetmax", "targetweight"), default="targetmax")
parser.add_argument("--run-name", default="velocity_gait_targetmax_v3")
parser.add_argument("--seed", type=int, default=33211)
parser.add_argument("--eval-seed-start", type=int, default=33220)
args = parser.parse_args()
source, prior, config, output = [p.resolve() for p in (args.source, args.prior_trial, args.config, args.output)]
output.mkdir(parents=True, exist_ok=True)
lock = (output/"followup.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
record = output/"followup_state.json"
if record.exists():
    raise SystemExit("Followup already has state; inspect it, do not repeat")
state = {"pid": os.getpid(), "phase": "waiting_prior_trial", "created_unix": time.time()}


def save(**changes):
    state.update(changes)
    temp = record.with_suffix(".tmp");temp.write_text(json.dumps(state, indent=2));temp.replace(record)


def execute(command, log_name, phase):
    with (output/log_name).open("w") as log:
        process = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_GL="egl"))
        save(phase=phase, active_pid=process.pid, active_command=command)
        status = process.wait()
    if status:
        raise RuntimeError(f"{phase} exited {status}; no further submission")


try:
    save()
    while True:
        previous = json.loads((prior/"state.json").read_text())
        if previous["phase"] == "completed_pending_review":
            break
        if previous["phase"] == "failed" or time.time()-state["created_unix"] > 10800:
            raise RuntimeError("Prior pipeline failed or timed out")
        try:
            os.kill(previous["pipeline_pid"], 0)
        except ProcessLookupError:
            raise RuntimeError("Prior pipeline is not alive; inspect before recovery")
        time.sleep(10)
    seeds = previous["summary"]["independent_seeds"]
    if len(seeds) != 4:
        raise ValueError("Exactly four independent reports required")
    reports = [json.loads((prior/f"strict_{seed}.json").read_text()) for seed in seeds]
    if not all(r["falls_total"] == 0 and r["body_contact_audit"]["nonfoot_ground_contact_samples"] == 0 and
               r["max_joint_velocity_ratio"] <= 1.00001 and r["max_joint_limit_excess_rad"] <= 1e-5 and
               r["max_joint_torque_ratio"] <= 1.00001 and r["max_target_rate_ratio"] > 1. for r in reports):
        raise ValueError("Failure mechanism differs; do not blindly launch target-rate trial")
    if not all(max(row["rmse_xy"] for row in r["stages"]) <= .1 and max(row["rmse_yaw"] for row in r["stages"]) <= .15 for r in reports):
        raise ValueError("Tracking has degraded; diagnose rather than tighten target-rate loss")
    expected = dict(previous["gait_settings"])
    if args.intervention == "targetmax":
        expected["mapped_target_rate_reduction"] = "max"
        intervention = "target-rate squared excess mean -> max; all weights unchanged"
    else:
        if expected.get("mapped_target_rate_reduction") != "max" or expected["mapped_target_rate_weight"] != 1.:
            raise ValueError("Weight trial requires the verified max-reduction parent")
        paired = json.loads((prior/"paired_gait.json").read_text())
        forward = next(row for row in paired["stages"] if row["stage"] == "forward")
        if not (.26 <= forward["height_mean_m"] <= .33 and all(leg["complete_cycles"] >= 8 for leg in forward["legs"])):
            raise ValueError("Posture/swing-cycle quality is not preserved; do not launch weight trial")
        expected["mapped_target_rate_weight"] = 4.
        intervention = "target-rate max penalty weight 1 -> 4; all other settings unchanged"
    if json.loads(config.read_text()) != expected:
        raise ValueError("Counterfactual must change only the designated target-rate setting")
    parent = Path(previous["checkpoint"])
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    if parent_sha != previous["checkpoint_sha256"]:
        raise ValueError("Prior final model changed")
    save(phase="gate_passed", parent=str(parent), parent_sha256=parent_sha,
         evidence=[r["target_rate_peak_event"] for r in reports], intervention=intervention)
    execute([sys.executable, "-m", "pytest", "-q", "scripts/test_gait_contract.py"], "tests.log", "tests")
    smoke_name = args.run_name+"_smoke"
    start = time.time()
    command = [sys.executable, "scripts/train_gait.py", "--gait-config", str(config), "--task", "Go2-Velocity-Gait-V2",
               "--num-envs", "32", "--iterations", "24", "--warmup-iterations", "2", "--critic-warmup-iterations", "2",
               "--actor-ramp-iterations", "2", "--buffer-len", "40", "--imag-batch", "32", "--seed", str(args.seed-10),
               "--log-interval", "1", "--save-interval", "24", "--resume", str(parent), "--reset-critic", "true",
               "--continue-run", "false", "--lr", "0.00005", "--actor-lr-scale", "0.25", "--run-name", smoke_name]
    execute(command, "smoke.log", "smoke")
    runs = [p for p in (source/"logs/supermimic").glob("*"+smoke_name) if p.stat().st_mtime >= start]
    if len(runs) != 1:
        raise ValueError("Ambiguous smoke run")
    execute([sys.executable, "scripts/gait_refinement_pipeline.py", "--source", str(source), "--parent", str(parent),
             "--expected-parent-sha", parent_sha, "--config", str(config), "--smoke-run", str(runs[0]),
             "--output", str(output/"bounded_trial"), "--run-name", args.run_name+"_bounded_20261026",
             "--iterations", "800", "--seed", str(args.seed), "--eval-seed-start", str(args.eval_seed_start)], "pipeline.log", "bounded_trial")
    save(phase="completed_pending_review", active_pid=None)
except Exception as error:
    save(phase="failed", active_pid=None, error=str(error))
    raise
