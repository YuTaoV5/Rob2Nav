"""Run one reviewed AS2-W candidate and its fixed MuJoCo comparison."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .as2w_reward_modules import SOURCE, defaults, execute_trainer, run_training
else:
    from as2w_reward_modules import SOURCE, defaults, execute_trainer, run_training

STAGES = {
    "flat": (),
    "slope": ("--terrain", "slope"),
    "rough": ("--terrain", "rough"),
    "stairs": ("--terrain", "stairs"),
}
RECOVERY_TERRAINS = ("flat", "slope", "rough")
EVAL_FLAGS = ("--limit-targets", "--disable-self-collision", "--ground-friction-priority",
              "--action-clip", "6", "--wheel-scale", "16", "--wheel-angle-period", "12.566370614359172")
BASELINE = "runs/fall_recovery/righting-conservative-r12/model_1999.pt"
STAIR_BASELINE = "runs/fall_recovery/blind-stairs-imitation-r17/model_1999.pt"
STAIR_RECIPES = {
    "as2w_stairs_r14": ("train_as2w_blind_stairs.py", ()),
    "as2w_stairs_r15": ("train_as2w_stair_clearance.py", ("--stair-clearance-reward",)),
    "as2w_stairs_r16": ("train_as2w_stair_transfer.py", ("--stair-transfer-curriculum",)),
}


def digest(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
        return sha.hexdigest()


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def find_video(folder, terrain, row):
    suffix = f"-{row['pose']}-{row['seed']}.mp4"
    for path in sorted(folder.glob(f"*{suffix}")):
        simulated_terrain = next((name for name in ("slope", "rough", "stairs")
                                  if f"-{name}-dynamics-" in path.name), "flat")
        if simulated_terrain == terrain and path.stat().st_size > 1024:
            return path.name
    return None


def score_flags(target):
    params = target / "reward-profile.json"
    if not params.is_file():
        return []
    return ["--studio-parameters", str(params), "--studio-profile",
            (target / "reward-profile.txt").read_text(encoding="utf-8")]


def record_failed_replays(python, run, env, baseline, candidate):
    """Add paired clips for valid failed cases on either checkpoint."""
    pairs = tuple(zip(baseline["cases"], candidate["cases"]))
    for terrain, flags in STAGES.items():
        missing = sorted({left["seed"] for left, right in pairs if left["terrain"] == terrain
                          and any(row["valid"] and not row["success"] for row in (left, right))
                          and any(row["valid"] and not row["video"] for row in (left, right))})
        if not missing:
            continue
        video_flags = [item for seed in missing for item in ("--video-seed", str(seed))]
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            target = run / label
            subprocess.run([python, str(target / "evaluate-fall-baseline.py"),
                            "--checkpoint", report["checkpoint"], *EVAL_FLAGS, *flags, "--video", *video_flags,
                            *score_flags(target)],
                           env=env, cwd=target, check=True)
            for row in report["cases"]:
                if row["terrain"] == terrain and row["seed"] in missing:
                    row["video"] = find_video(target / "fall-baseline", terrain, row)


def evaluate(python, harness, run, label, checkpoint, env, stages):
    target = run / label
    target.mkdir()
    training = Path(env.get("ROB2NAV_AS2W_TRAINING", "/root/as2w-training"))
    for name in ("evaluate-fall-baseline.py", "official-model", "local-step-diagnostic"):
        asset = harness / name
        if name == "official-model" and not asset.exists():
            asset = training / "unitree_ros"
        (target / name).symlink_to(asset)
    # The existing evaluator resolves its location; run its contents in the
    # isolated output folder so filenames and model assets remain colocated.
    (target / "evaluate-fall-baseline.py").unlink()
    script = target / "evaluate-fall-baseline.py"
    script.write_bytes((harness / script.name).read_bytes())
    scorer = harness / "as2w_replay_score.py"
    if scorer.is_file() and "stairs" not in stages:
        (target / scorer.name).symlink_to(scorer)
        helper = harness / "reward_expression.py"
        if helper.is_file():
            (target / helper.name).symlink_to(helper)
        params = run / "parameters.json" if label == "candidate" else checkpoint.parent.parent / "parameters.json"
        profile = "候选训练配置" if label == "candidate" else "基线训练配置"
        if not params.is_file():
            profile = "R13 默认参考配置"
        write(target / "reward-profile.json", json.loads(params.read_text()) if params.is_file() else defaults())
        (target / "reward-profile.txt").write_text(profile, encoding="utf-8")
    base = target / "fall-baseline"
    base.mkdir()
    contract = harness / "fall-recovery-contract.json"
    if not contract.is_file():
        contract = training / "replay/fall-recovery-contract.json"
    (base / "fall-recovery-contract.json").symlink_to(contract)
    for terrain in stages:
        subprocess.run([python, str(script), "--checkpoint", str(checkpoint), *EVAL_FLAGS,
                        *STAGES[terrain], "--video", *score_flags(target)],
                       env=env, cwd=target, check=True)
    cases = []
    for file in sorted(base.glob("*.jsonl")):
        terrain = next((name for name in ("slope", "rough", "stairs") if f"-{name}-dynamics-" in file.name), "flat")
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cases.append(dict(row, terrain=terrain, success=row["passed"],
                              case_id=f"{terrain}:{row['pose']}:{row['seed']}",
                              video=find_video(base, terrain, row)))
    if len(cases) != 16 * len(stages) or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Incomplete or duplicate MuJoCo cases")
    return {"checkpoint": str(checkpoint), "sha256": digest(checkpoint), "cases": cases}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train-native":
        _, _, file, *args = sys.argv
        source = Path(file).resolve()
        sys.path.insert(0, str(source.parent))
        sys.argv = [file, *args]
        execute_trainer(source, ast.parse(source.read_text(encoding="utf-8")))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        _, _, source, profile, trace, *args = sys.argv
        sys.path.insert(0, str(Path(source).parent))
        sys.argv = [source, *args]
        run_training(source, profile, trace)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("harness", type=Path)
    parser.add_argument("mujoco_python")
    parser.add_argument("--sim-python", default="/isaac-sim/python.sh")
    parser.add_argument("--isaaclab", default="/root/isaaclab")
    parser.add_argument("--phase", choices=("both", "train", "evaluate"), default="both")
    args = parser.parse_args()
    run, training, harness = (path.resolve() for path in (args.run, args.training, args.harness))
    status = run / "status.json"
    phase = "preflight"
    try:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        baseline = Path(manifest.get("baseline_checkpoint", training / BASELINE))
        stair = STAIR_RECIPES.get(manifest.get("recipe"))
        source = training / (stair[0] if stair else SOURCE)
        if digest(source) != manifest["trainer_sha256"] or digest(baseline) != manifest["baseline_sha256"]:
            raise RuntimeError("训练源码或基线权重已变更，请重新创建候选")
        import fcntl
        env = dict(os.environ, ACCEPT_EULA="Y", PYTHONUNBUFFERED="1", MUJOCO_GL="egl")
        env["ROB2NAV_AS2W_TRAINING"] = str(training)
        env["ROB2NAV_AS2W_USD_DIR"] = str(run / "asset_cache")
        env.setdefault("OMP_NUM_THREADS", "4")
        updates = manifest["updates"]
        candidate_dir = run / "checkpoint"
        if args.phase != "evaluate":
          with (run.parent / ".gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            phase = "training"
            write(status, {"state": phase})
            modules = ["isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaaclab_assets"]
            env["PYTHONPATH"] = os.pathsep.join([*(f"{args.isaaclab}/source/{module}" for module in modules),
                                               str(training), *filter(None, [env.get("PYTHONPATH")])])
            entry = ([str(Path(__file__).resolve()), "train-native", str(source)] if stair else [str(Path(__file__).resolve()), "train", str(source),
                     str(run / "parameters.json"), str(run / "reward.jsonl")])
            flags = ["--blind-stairs", *stair[1]] if stair else ["--righting-resets", "--stance-height-gate", "--stance-streak-reward"]
            subprocess.run([args.sim_python, *entry,
                            "--headless", "--device", "cuda:0", "--seed", str(manifest.get("seed", 413)), "--wheel-scale", "16",
                            "--flat-domain-randomization", "--mixed-terrain", "--mirror-recovery",
                            "--conservative-ppo", *flags, "--initialize-checkpoint", str(baseline),
                            "--run-dir", str(candidate_dir), "--num-envs", str(manifest["envs"]),
                            "--iterations", str(updates)], cwd=training, env=env, check=True)
        if args.phase == "train":
            write(status, {"state": "trained"})
            return
        phase = "evaluating"
        write(status, {"state": phase})
        candidate = candidate_dir / f"model_{updates - 1}.pt"
        stages = ("stairs",) if stair else tuple(manifest.get("evaluation_terrains",
                  ("flat",) if manifest["smoke"] else RECOVERY_TERRAINS))
        base = evaluate(args.mujoco_python, harness, run, "baseline", baseline, env, stages)
        result = evaluate(args.mujoco_python, harness, run, "candidate", candidate, env, stages)
        if [row["case_id"] for row in base["cases"]] != [row["case_id"] for row in result["cases"]]:
            raise ValueError("MuJoCo case mismatch")
        record_failed_replays(args.mujoco_python, run, env, base, result)
        write(run / "comparison.json", {"baseline": base, "candidate": result, "backend": "MuJoCo",
              "protocol": {"flags": EVAL_FLAGS, "terrain": stages, "evaluator_sha256": digest(harness / "evaluate-fall-baseline.py"),
                           "contract_sha256": digest((run / "baseline/fall-baseline/fall-recovery-contract.json").resolve())}})
        write(status, {"state": "succeeded"})
    except Exception as exc:
        write(status, {"state": "failed", "phase": phase, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
