"""Publish a completed research trial video without replacing a production policy."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trial", type=Path, required=True)
parser.add_argument("--rob2nav", type=Path, required=True)
parser.add_argument("--id", required=True)
parser.add_argument("--title", required=True)
parser.add_argument("--build", action="store_true")
args = parser.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_]+", args.id):
    raise SystemExit("Invalid asset id")
trial, root = args.trial.resolve(), args.rob2nav.resolve()
summary = json.loads((trial/"summary.json").read_text())
state = json.loads((trial/"state.json").read_text())
if state["phase"] not in ("completed_pending_review", "publishing"):
    raise SystemExit("Trial has not completed")
video = Path(summary["video"])
metadata = json.loads(video.with_suffix(".json").read_text())
checkpoint = Path(summary["checkpoint"])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["model_sha256"]:
    raise SystemExit("Video/model hash mismatch")
target = root/"artifacts/supermimic"
target.mkdir(parents=True, exist_ok=True)
mp4, webm, poster = [target/(args.id+ext) for ext in (".mp4", ".webm", ".jpg")]
if mp4.exists() and mp4.read_bytes() != video.read_bytes():
    raise SystemExit("Asset id belongs to a different video; choose a new id")
shutil.copyfile(video, mp4)
if not webm.exists():
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9",
                    "-threads", "2", "-crf", "32", "-b:v", "0", "-an", str(webm)], check=True)
if not poster.exists():
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-frames:v", "1", str(poster)], check=True)
reports = [json.loads(p.read_text()) for p in sorted(trial.glob("strict_*.json"))]
metadata.update(title=args.title, capability_qualified=False,
                limitations=["research candidate; target-rate and full-speed visual review remain independent gates", "not hardware qualified"],
                independent_results=[{"seed": r["seed"], "falls": r["falls_total"],
                   "nonfoot_contacts": r["body_contact_audit"]["nonfoot_ground_contact_samples"],
                   "target_rate_ratio": r["max_target_rate_ratio"]} for r in reports])
(target/(args.id+".json")).write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
target_passed = bool(reports) and all(r["max_target_rate_ratio"] <= 1.00001 for r in reports)
detail = "32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。" + ("原始目标变化率仍未全部通过；" if not target_passed else "") + "研究候选，完整原速人工复看及实机资格未完成。"
backup = root/"outputs/supermimic_gait_publish"/args.id
backup.mkdir(parents=True, exist_ok=True)
component = root/"studio/frontend/src/SuperMimicDemos.tsx"
if component.exists():
    content = component.read_text()
    if "id:'"+args.id+"'" not in content:
        marker = "const demos=["
        if marker not in content:
            raise SystemExit("Unknown demo component format; preserve it and integrate manually")
        if not (backup/component.name).exists():
            shutil.copyfile(component, backup/component.name)
        card = "\n {id:"+json.dumps(args.id)+",title:"+json.dumps(args.title, ensure_ascii=False)+",detail:"+json.dumps(detail, ensure_ascii=False)+"},"
        # Detect both quote styles on later idempotent runs.
        if json.dumps(args.id) not in content:
            component.write_text(content.replace(marker, marker+card, 1))
page = root/"studio/frontend/public/supermimic.html"
if page.exists() and args.id not in page.read_text():
    content = page.read_text()
    if not (backup/page.name).exists():
        shutil.copyfile(page, backup/page.name)
    base = "/api/files/artifacts/supermimic/"+args.id
    card = '<article><h2>'+html.escape(args.title)+'</h2><video controls playsinline preload="metadata" poster="'+base+'.jpg"><source src="'+base+'.webm" type="video/webm"><source src="'+base+'.mp4" type="video/mp4"></video><p>'+detail+'</p><a href="'+base+'.mp4">下载MP4</a></article>'
    marker = '<div class="grid">' if '<div class="grid">' in content else '<main>'
    if marker not in content:
        raise SystemExit("Unknown standalone page format; preserve it and integrate manually")
    page.write_text(content.replace(marker, marker+card, 1))
if args.build:
    subprocess.run(["npm", "run", "build"], cwd=root/"studio/frontend", check=True)
print(json.dumps({"published": args.id, "url": "/supermimic.html", "model_sha256": metadata["model_sha256"], "capability_qualified": False}))
