"""Verify SHA-256 of the published code, model, configuration and evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
failures = []
for relative, expected in manifest["sha256"].items():
    path = ROOT / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(relative)
if failures:
    raise SystemExit("Missing or changed resources:\n" + "\n".join(failures))
print(f"Verified {len(manifest['sha256'])} resources; capability_qualified=false")
