from pathlib import Path
import collections
import json
import statistics

root = Path(__file__).parent
folders = [root / name for name in ("mujoco-85-32", "mujoco-85-32-support-gate")]
baseline, gate = [json.loads((p / "contract.json").read_text()) for p in folders]
assert gate.pop("support_gate_force_n") == 8
assert gate.pop("support_gate_wait_cap_s") == .30
assert baseline == gate
reports = [json.loads((p / "mujoco-stepping-diagnostic.json").read_text()) for p in folders]
out = {}
for mode in ("up", "down"):
    groups = [r[mode]["rows"] for r in reports]
    assert [r["seed"] for r in groups[0]] == [r["seed"] for r in groups[1]]
    out[mode] = {}
    for name, rows in zip(("baseline", "gate"), groups):
        out[mode][name] = {
            "trials": len(rows),
            "passes": sum(r["passed"] for r in rows),
            "failures": dict(collections.Counter(str(r["failure"]) for r in rows)),
            "median_progress_m": statistics.median(r["forward_progress_m"] for r in rows),
            "wait_frames": [r.get("support_gate_wait_frames",0) for r in rows],
        }
(root / "support-gate-comparison.json").write_text(json.dumps(out, indent=2)+"\n")
print(out)
