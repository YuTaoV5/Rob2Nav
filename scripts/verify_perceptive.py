"""Verify released weights, run-start source snapshots, and complete videos."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio

base = Path(__file__).resolve().parents[1]/"artifacts/perceptive"
manifest = json.loads((base/"manifest.json").read_text())
for name, record in manifest["checkpoints"].items():
    assert hashlib.sha256((base/"checkpoints"/name).read_bytes()).hexdigest() == record["sha256"], name
sources, videos = 0, []
for group in manifest["evaluation_groups"]:
    for path in (base/group).glob("**/summary.json"):
        summary = json.loads(path.read_text())
        for name, sha in summary["source_sha256"].items():
            assert hashlib.sha256((path.parent/"source"/name).read_bytes()).hexdigest() == sha, str(path)+name
            sources += 1
        for video in path.parent.glob("*.mp4"):
            reader = imageio.get_reader(video)
            actual = reader.count_frames()
            reader.close()
            if "video_frames" in summary:
                expected = summary["video_frames"]
            else:
                episode = int(video.stem.split("_")[-1])
                trace = json.loads((path.parent/f"trajectory_{episode}.json").read_text())
                expected = (len(trace)+1)//2
            assert actual == expected and actual > 0, (str(video), actual, expected)
            videos.append({"path": str(video.relative_to(base)), "frames": actual})
result = {"weight_files": len(manifest["checkpoints"]), "source_references": sources, "videos": videos}
(base/"verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
