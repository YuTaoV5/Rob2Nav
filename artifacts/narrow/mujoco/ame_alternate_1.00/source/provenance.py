"""Capture source bytes at run start, before any long-running experiment."""
import hashlib
from pathlib import Path


def snapshot(out):
    dest = Path(out)/"source"
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        (dest/path.name).write_bytes(data)
    return hashes
