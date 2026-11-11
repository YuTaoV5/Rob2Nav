"""AS2-W workspace routes for the existing Rob2Nav Studio."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

from .as2w_formal import mount_formal_as2w

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .as2w_reward_modules import SOURCE, catalog, defaults, resolve
from .as2w_runner import BASELINE, digest, write


class Candidate(BaseModel):
    parameters: dict = Field(default_factory=dict)
    updates: int = Field(default=2000, ge=2000, le=40000)
    envs: int = Field(default=256, ge=40, le=2048)


def mount_as2w(app, root, allow_training, *, allow_formal_training=None):
    training = Path(os.environ.get("ROB2NAV_AS2W_TRAINING", "/root/as2w-training"))
    harness = Path(os.environ.get("ROB2NAV_AS2W_EVAL", str(root / "studio/as2w_eval")))
    run_root = root / "outputs/studio/as2w"
    formal_enabled = allow_training if allow_formal_training is None else allow_formal_training
    mount_formal_as2w(app, root, run_root, formal_enabled)
    source = training / SOURCE
    baseline = training / BASELINE

    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def run_path(key):
        if not re.fullmatch("[a-f0-9]{32}", key):
            raise HTTPException(404)
        path = run_root / key
        if not path.is_dir():
            raise HTTPException(404)
        return path

    @app.get("/api/as2w")
    def config():
        try:
            modules = catalog(source.read_text(encoding="utf-8"))
        except (ValueError, OSError, StopIteration) as exc:
            raise HTTPException(503, f"AS2-W 奖励源码不可用：{exc}") from exc
        return dict(modules, training=allow_training and baseline.is_file()
                    and (harness / "evaluate-fall-baseline.py").is_file(), baseline=BASELINE)

    @app.post("/api/as2w/preview")
    def preview(data: Candidate):
        try:
            values = resolve(data.parameters)
            modules = catalog(source.read_text(encoding="utf-8"), values)
        except (ValueError, OSError, StopIteration) as exc:
            raise HTTPException(422, str(exc)) from exc
        diff = {key: {"before": default, "after": values[key]}
                for key, default in defaults().items() if values[key] != default}
        return dict(modules, parameters=values, diff=diff, updates=data.updates, envs=data.envs)

    @app.get("/api/as2w/runs")
    def runs():
        return [dict(load(path / "manifest.json") or {}, id=path.name, status=load(path / "status.json"))
                for path in sorted(run_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:40]
                for path in [path.parent]]

    @app.post("/api/as2w/runs")
    def submit(data: Candidate):
        if not allow_training:
            raise HTTPException(403, "当前服务为只读训练模式")
        if not baseline.is_file() or not (harness / "evaluate-fall-baseline.py").is_file():
            raise HTTPException(503, "缺少 AS2-W 基线权重或评估器")
        plan = preview(data)
        if not plan["diff"]:
            raise HTTPException(422, "请至少修改一个参数")
        key = uuid4().hex
        path = run_root / key
        path.mkdir(parents=True)
        write(path / "manifest.json", {**{k: plan[k] for k in ("diff", "updates", "envs")},
              "smoke": False, "trainer_sha256": digest(source), "baseline_sha256": digest(baseline)})
        write(path / "parameters.json", plan["parameters"])
        write(path / "status.json", {"state": "queued"})
        try:
            with (path / "run.log").open("xb") as log:
                subprocess.Popen([sys.executable, str(Path(__file__).with_name("as2w_runner.py")), str(path),
                                  str(training), str(harness), os.environ.get("ROB2NAV_AS2W_MUJOCO_PYTHON",
                                  str(root.parent / "envs/go2-supermimic/bin/python"))], cwd=root,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            write(path / "status.json", {"state": "failed", "error": str(exc)})
            raise HTTPException(503, str(exc)) from exc
        return {"id": key}

    @app.get("/api/as2w/runs/{key}")
    def result(key: str):
        path = run_path(key)
        trace = path / "reward.jsonl"
        samples = []
        if trace.is_file():
            for line in trace.read_text(encoding="utf-8").splitlines(keepends=True):
                if line.endswith("\n"):
                    samples.append(json.loads(line))
        return {"manifest": load(path / "manifest.json"), "status": load(path / "status.json"),
                "samples": samples[::max(1, (len(samples) + 239) // 240)], "latest": samples[-1] if samples else None,
                "comparison": load(path / "comparison.json"), "parameters": load(path / "parameters.json")}

    @app.get("/api/as2w/runs/{key}/video/{label}/{filename}")
    def video(key: str, label: str, filename: str):
        path = run_path(key)
        comparison = load(path / "comparison.json") or {}
        if label not in ("baseline", "candidate") or not re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", filename):
            raise HTTPException(404)
        if not any(row["video"] == filename for row in comparison.get(label, {}).get("cases", ())):
            raise HTTPException(404)
        return FileResponse(path / label / "fall-baseline" / filename, media_type="video/mp4")
