"""Verify every retained release trial against its archived Python sources."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root/"artifacts/active"
problems = []
count = 0
for summary in base.glob("*/summary.json"):
    trial = json.loads(summary.read_text())
    for name, digest in trial.get("source_sha256", {}).items():
        saved = base/"sources"/(digest+".py")
        if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest() != digest:
            problems.append(f"{summary.parent.name}/{name}")
        count += 1
if problems:
    raise SystemExit("Missing or modified source snapshots: "+", ".join(problems))
print("Verified", count, "trial source references")
