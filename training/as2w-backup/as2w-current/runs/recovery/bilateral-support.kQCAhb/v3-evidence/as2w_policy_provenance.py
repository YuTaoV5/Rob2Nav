"""Checkpoint identity shared by evaluation and export."""
import hashlib


def checkpoint_record(path):
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as source:
        digest = hashlib.file_digest(source,"sha256").hexdigest()
    return {"checkpoint":str(resolved),"checkpoint_sha256":digest}
