"""Check the fixed upstream BBC files before loading trusted checkpoints."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]/"vendor/NJUAgility"
snapshot = json.loads((root/"snapshot.json").read_text())
changed = [name for name, digest in snapshot["files"].items()
           if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
if changed:
    raise SystemExit("NJU snapshot mismatch: "+", ".join(changed))
print("NJU snapshot verified:", snapshot["revision"], len(snapshot["files"]), "files")
