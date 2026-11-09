"""Read-only, per-job review and provenance for the shared experiment UI."""
import json
from pathlib import Path

from fastapi import HTTPException

from .catalog import read_json
from .experiment_native_rewards import FLIP, LABELS
from .experiment_workflow import manifest


def artifact_url(root, path):
    return "/api/files/" + path.relative_to(root).as_posix() if path.is_file() else None


def as2w_cases(root, run):
    compare = read_json(run / "comparison.json") or {}
    result = []
    for a, b in zip(compare.get("baseline", {}).get("cases", []), compare.get("candidate", {}).get("cases", [])):
        if a["case_id"] != b["case_id"]:
            return []
        sides = {}
        for label, row in (("baseline", a), ("candidate", b)):
            folder = run / label / "fall-baseline"
            video = folder / Path(row["video"]).name if row.get("video") else None
            sides[label] = {"passed": row["success"], "valid": row["valid"], "stable_s": row.get("stable_s"),
                            "video": artifact_url(root, video) if video else None,
                            "trace": video.with_suffix(".reward.json") if video else None}
        result.append(dict(id=a["case_id"], label=a["case_id"], sides=sides))
    return result


def go2_cases(root, run, parent=None):
    evaluation = read_json(run / "evaluation.json") or {}
    baseline = (read_json(parent / "evaluation.json") or {}) if parent else {}
    comparable = evaluation.get("protocol") and evaluation["protocol"] == baseline.get("protocol")
    by_seed = {r["seed"]: r for r in baseline.get("episodes", [])} if comparable else {}
    cases = []
    for row in evaluation.get("episodes", []):
        if type(row.get("seed")) is not int:
            continue
        folder = run / f"seed_{row['seed']}"
        if not (folder / "summary.json").is_file():
            continue
        def side(evidence, selected):
            return dict(passed=evidence.get("passed"), valid=True, video=artifact_url(root, selected / "episode_0.mp4"),
                        trace=selected / "trace.json")
        sides = {"candidate": side(row, folder)}
        reference = by_seed.get(row["seed"])
        if reference is not None:
            sides = {"baseline": side(reference, parent / folder.name), **sides}
        cases.append(dict(id=str(row["seed"]), label=f"Go2 · seed {row['seed']}", sides=sides))
    return cases


def go2_trace(run, source):
    rows = read_json(source) or []
    export = read_json(run / "exported/export.json") or {}
    scales = export.get("actual_reward_scales", {})
    for row in rows:
        terms = {}
        for key, raw in row.get("studio_raw", {}).items():
            if key in scales:
                terms[key] = dict(raw=raw, weighted=raw * scales[key], label=LABELS.get(key, key))
        row["terms"] = terms
        row["time"] = row.get("pts_s", row["time"])
        row["signals"] = {k: row[k] for k in ("upright", "speed", "angular_speed", "standard_pose_error") if k in row}
        if "all_feet_ground" in row:
            row["signals"]["all_feet_ground"] = int(row["all_feet_ground"])
    return dict(frames=rows, label="MuJoCo 复算 · 贡献按乘控制步长前显示")


def mount_experiment_review(app, root, jobs):
    def get_job(key):
        try:
            job = jobs.get(key)
        except KeyError as exc:
            raise HTTPException(404, "找不到任务") from exc
        return job, root / "outputs/studio/runs" / job["id"]

    def cases_for(job, run):
        recipe = job["config"].get("recipe_id")
        if job["config"].get("robot_id") == "as2w":
            return as2w_cases(root, run)
        if recipe in FLIP:
            parent = None
            if job["config"].get("parent_job_id"):
                try:
                    parent_job, reference = get_job(job["config"]["parent_job_id"])
                except HTTPException:
                    pass
                else:
                    if parent_job["config"].get("recipe_id") == recipe and parent_job["state"] == "succeeded":
                        parent = reference
            return go2_cases(root, run, parent)
        return []

    @app.get("/api/jobs/{key}/experiment-review")
    def review(key: str):
        job, run = get_job(key)
        cases = cases_for(job, run)
        trace = run / "reward.jsonl"
        samples = []
        if job["config"].get("robot_id") == "as2w" and trace.is_file():
            with trace.open() as stream:
                for line in stream:
                    if line.endswith("\n"):
                        samples.append(json.loads(line))
        return dict(cases=[dict(id=c["id"], label=c["label"], sides={
            side: {**{k:v for k,v in value.items() if k!="trace"},
                   "hasTrace": bool(value["trace"] and value["trace"].is_file())}
            for side,value in c["sides"].items()}) for c in cases],
            train=samples[::max(1,(len(samples)+249)//250)], state=job["state"], isTraining=job["config"].get("task")=="train")

    @app.get("/api/jobs/{key}/experiment-trace")
    def frame_data(key: str, case: str, side: str):
        job, run = get_job(key)
        found = next((c for c in cases_for(job, run) if c["id"] == case), None)
        trace = found["sides"].get(side, {}).get("trace") if found else None
        if not trace or not trace.is_file():
            raise HTTPException(404, "该场景没有逐步数据")
        if job["config"].get("robot_id") == "as2w":
            return read_json(trace)
        return go2_trace(trace.parent.parent, trace)

    @app.get("/api/jobs/{key}/candidate-draft")
    def candidate(key: str, case: str = ""):
        from .config import RunConfig
        from .registry import resolve_recipe
        job, run = get_job(key)
        if job["config"].get("task") != "train":
            raise HTTPException(422, "请选择训练任务")
        parent = RunConfig.model_validate(job["config"])
        cfg = parent.model_dump()
        cfg.update(name=("候选 · " + job["name"])[:80], parent_job_id=key, parent_case=case or None)
        checkpoint = job["detail"].get("checkpoint")
        if not checkpoint and (run / "exported/export.json").is_file():
            candidates = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            if candidates:
                checkpoint = candidates[-1].relative_to(root).as_posix()
        if checkpoint and job["state"] == "succeeded":
            if parent.recipe_id in FLIP and parent.recipe_options.get("fixstand_roundtrip"):
                cfg.update(initialization="resume", checkpoint=checkpoint,
                           recipe_options={**parent.recipe_options, "reset_optimizer":True})
            elif parent.robot_id == "as2w" or not parent.recipe_id or (resolve_recipe(parent) or {}).get("execution") == "native":
                cfg.update(initialization="initialize", checkpoint=checkpoint)
        try:
            baseline = {p["id"]:p["value"] for term in manifest(root,parent,resolve_recipe(parent))["terms"] for p in term["parameters"]}
        except (ValueError, OSError, StopIteration):
            baseline = {}
        return dict(config=cfg, baseline=baseline)
