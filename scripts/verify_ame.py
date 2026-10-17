"""Verify AME evidence and measure exported CPU actor latency (one thread)."""
import hashlib
import argparse
import json
import statistics
import time
from pathlib import Path
import imageio.v2 as imageio
import torch

base = Path(__file__).resolve().parents[1]/"artifacts/ame"
p = argparse.ArgumentParser()
p.add_argument("--write", action="store_true", help="Update the archived verification record, including machine-specific timings")
args = p.parse_args()
manifest = json.loads((base/"manifest.json").read_text())
initializer = manifest["initializer"]
assert hashlib.sha256((base.parents[1]/initializer["published_path"]).read_bytes()).hexdigest() == initializer["sha256"]
for item in manifest["checkpoints"].values():
    assert hashlib.sha256((base/item["path"]).read_bytes()).hexdigest() == item["sha256"]
sources = 0
diagnostic = json.loads((base/"diagnostics/grid_manifest.json").read_text())
for name, expected in diagnostic["source_sha256"].items():
    assert hashlib.sha256((base/"diagnostics/grid_source"/name).read_bytes()).hexdigest() == expected
    sources += 1
for path in list(base.glob("**/summary.json"))+[base/label/"manifest.json" for label in ("attention", "control")]:
    data = json.loads(path.read_text())
    for name, expected in data["source_sha256"].items():
        assert hashlib.sha256((path.parent/"source"/name).read_bytes()).hexdigest() == expected
        sources += 1
videos = []
for video in base.glob("**/*.mp4"):
    trace = json.loads((video.parent/"trajectory_0.json").read_text())
    reader = imageio.get_reader(video)
    frames = reader.count_frames(); reader.close()
    assert frames == (len(trace)+1)//2 and frames > 0
    videos.append({"path": str(video.relative_to(base)), "frames": frames})
latency = {}
branch = {}
torch.set_num_threads(1)
torch.manual_seed(0)
obs = torch.randn(1,419); obs[:,232:] = (torch.rand(1,187) > .5).float()
for label in ("control", "attention"):
    actor = torch.jit.load(str(base/(label+"_eval")/"native_stairs/actor.pt")).eval()
    with torch.inference_mode():
        for _ in range(100): actor(obs)
        samples = []
        for _ in range(500):
            start = time.perf_counter(); output = actor(obs)
            samples.append((time.perf_counter()-start)*1000)
        assert output.shape == (1,12) and torch.isfinite(output).all()
    latency[label] = {"median_ms": statistics.median(samples),
                      "p95_ms": sorted(samples)[474],
                      "parameters": sum(p.numel() for p in actor.parameters())}
    if label == "attention":
        frames = json.loads((base/"attention_eval/native_stones/attention.json").read_text())
        actual = torch.tensor([f["proprio"]+f["height"]+f["valid"] for f in frames])
        with torch.inference_mode():
            delta = actor(actual)-actor.base(actual)
        branch = {"sampled_frames": len(frames), "action_rms": float(delta.square().mean().sqrt()),
                  "joint_offset_rms_rad": float(delta.square().mean().sqrt()*.25)}
result = {"source_references": sources, "videos": videos,
          "cpu_latency": latency, "torch": torch.__version__,
          "attention_branch_on_native_stones": branch,
          "latency_protocol": "CPU one thread, batch 1, 100 warmup, 500 timed calls; excludes sensing and control"}
if args.write:
    (base/"verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
