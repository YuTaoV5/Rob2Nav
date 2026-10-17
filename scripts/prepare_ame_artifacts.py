"""Package completed pilot runs, preserve raw failures, and verify source bytes."""
import hashlib
import json
import shutil
from pathlib import Path

src = Path("outputs/ame")
dest = Path("artifacts/ame")
dest.mkdir(parents=True, exist_ok=True)
manifest = {"checkpoints": {}, "training_steps_per_arm": 512*24*200,
            "seed": 44, "source_references": 0, "evaluations": {}}
manifest["initializer"] = {"published_path": "artifacts/perceptive/checkpoints/student.pt",
    "sha256": hashlib.sha256(Path("outputs/perceptive/student/model_299.pt").read_bytes()).hexdigest()}
shutil.copytree(src/"baseline_recheck", dest/"baseline_recheck", dirs_exist_ok=True)
for label in ("control", "attention"):
    train = src/label
    complete = json.loads((train/"completed.json").read_text())
    assert complete["requested_iterations"] == 200 and complete["last_iteration"] == 199
    target = dest/label
    target.mkdir(exist_ok=True)
    for f in train.iterdir():
        if f.is_file() and (not f.name.startswith("model_") or f.name == "model_199.pt"):
            shutil.copy2(f, target/f.name)
        elif f.name == "source":
            shutil.copytree(f, target/f.name, dirs_exist_ok=True)
    checkpoint = target/"model_199.pt"
    manifest["checkpoints"][label] = {"path": str(checkpoint.relative_to(dest)),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    shutil.copy2(Path("outputs")/("ame_"+label+".log"), target/"console.log")
    evaluation = src/(label+"_eval")
    assert json.loads((evaluation/"completed.json").read_text())["all_passed"]
    commands = json.loads((evaluation/"commands.json").read_text())
    assert len(commands) == (7 if label == "attention" else 6)
    assert all(c["returncode"] == 0 for c in commands)
    shutil.copytree(evaluation, dest/(label+"_eval"), dirs_exist_ok=True)
    manifest["evaluations"][label] = {}
    for path in (dest/(label+"_eval")).glob("*/summary.json"):
        record = json.loads(path.read_text())
        manifest["evaluations"][label][path.parent.name] = {"successes": record["successes"], "total": record.get("total", len(record["episodes"]))}
for path in dest.glob("**/manifest.json"):
    if path == dest/"manifest.json":
        continue
    data = json.loads(path.read_text())
    for name, expected in data.get("source_sha256", {}).items():
        assert hashlib.sha256((path.parent/"source"/name).read_bytes()).hexdigest() == expected
        manifest["source_references"] += 1
for path in dest.glob("**/summary.json"):
    data = json.loads(path.read_text())
    for name, expected in data["source_sha256"].items():
        assert hashlib.sha256((path.parent/"source"/name).read_bytes()).hexdigest() == expected
        manifest["source_references"] += 1
diagnostic = dest/"diagnostics"; diagnostic.mkdir(exist_ok=True)
for log in ("ame_smoke.log", "ame_attention_grid_diagnostic.log"):
    shutil.copy2(Path("outputs")/log, diagnostic/log)
shutil.copytree(src/"attention_grid_diagnostic"/"source", diagnostic/"grid_source", dirs_exist_ok=True)
shutil.copy2(src/"attention_grid_diagnostic"/"manifest.json", diagnostic/"grid_manifest.json")
manifest["diagnostics"] = "Early stopped grid-order attempt is not included in either arm's completed training budget."
(dest/"manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
