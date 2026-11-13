import collections
import json
import statistics
from pathlib import Path

root = Path(__file__).parent
data = {}
contracts = []
for name in ("control", "diagnostic"):
    folder = root / f"holdout-{name}"
    contract = json.loads((folder / "contract.json").read_text())
    contract.pop("checkpoint")
    contract.pop("checkpoint_sha256")
    contracts.append(contract)
    data[name] = json.loads((folder / "mujoco-stepping-diagnostic.json").read_text())
assert contracts[0] == contracts[1]
report = {}
for case in ("up", "down"):
    report[case] = {}
    assert [r["seed"] for r in data["control"][case]["rows"]] == [r["seed"] for r in data["diagnostic"][case]["rows"]]
    for name, doc in data.items():
        rows = doc[case]["rows"]
        report[case][name] = {
            "trials": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "failure_counts": dict(collections.Counter(str(r["failure"]) for r in rows)),
            "median_progress_m": statistics.median(r["forward_progress_m"] for r in rows),
            "max_abs_elevation_m": max(abs(r["signed_elevation_m"]) for r in rows),
            "min_knee_surface_m": min(r["min_knee_surface_m"] for r in rows),
        }
(root / "holdout-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(report)
