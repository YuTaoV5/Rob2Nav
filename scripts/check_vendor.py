"""Verify the vendored RoboGauge revision without requiring an embedded .git."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    snapshot = json.loads((ROOT / "vendor/robogauge_snapshot.json").read_text())
    problems = []
    for name, expected in snapshot["files"].items():
        path = ROOT / "vendor/RoboGauge" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append(name)
    if problems:
        raise SystemExit("Missing or changed upstream files: " + ", ".join(problems))
    print(f'RoboGauge snapshot verified: {snapshot["revision"]} ({len(snapshot["files"])} files)')


if __name__ == "__main__":
    main()
