"""Expose the repository's formal AS2-W recipes through Studio."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from studio.as2w_runner import write


class FormalRequest(BaseModel):
    skill: Literal["rolling", "stepping", "recovery"]
    smoke: bool = True


def mount_formal_as2w(app, root, run_root, allow_training):
    launcher = root / "training/as2w-backup/train.sh"

    @app.post("/api/as2w/formal-runs", status_code=202)
    def submit(data: FormalRequest):
        if not allow_training:
            raise HTTPException(403, "Training is disabled")
        if not launcher.is_file():
            raise HTTPException(503, "AS2-W training snapshot is missing")
        key = uuid4().hex
        run = run_root / key
        run.mkdir(parents=True)
        write(run / "manifest.json", {"recipe": data.skill, "smoke": data.smoke,
                                      "entrypoint": "training/as2w-backup/train.sh"})
        write(run / "status.json", {"state": "queued"})
        try:
            with (run / "run.log").open("xb") as log:
                subprocess.Popen(
                    [sys.executable, "-m", "studio.as2w_formal", str(launcher),
                     str(run), data.skill, "smoke" if data.smoke else "formal"],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            write(run / "status.json", {"state": "failed", "error": str(exc)})
            raise HTTPException(503, "Could not start training worker") from exc
        return {"id": key}

    @app.get("/api/as2w/formal-runs/{key}/log")
    def log(key: str):
        if not re.fullmatch("[a-f0-9]{32}", key):
            raise HTTPException(404)
        path = run_root / key / "run.log"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="text/plain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("launcher", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("skill", choices=("rolling", "stepping", "recovery"))
    parser.add_argument("mode", choices=("smoke", "formal"))
    args = parser.parse_args()
    status = args.run / "status.json"
    try:
        import fcntl

        env = dict(os.environ, PYTHONUNBUFFERED="1")
        for name in ("AS2W_ITERATIONS", "AS2W_NUM_ENVS"):
            env.pop(name, None)
        if args.mode == "smoke":
            env.update(AS2W_ITERATIONS="3", AS2W_NUM_ENVS="8")
        with (args.run.parent / ".gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            write(status, {"state": "training"})
            subprocess.run(
                ["bash", str(args.launcher), args.skill, str(args.run / "checkpoint")],
                env=env, check=True)
        write(status, {"state": "trained"})
    except Exception as exc:
        write(status, {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
