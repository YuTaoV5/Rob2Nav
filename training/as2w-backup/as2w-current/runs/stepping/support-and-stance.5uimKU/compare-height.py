import collections
import json
import statistics
from pathlib import Path
root = Path(__file__).parent
names = ("rear-predictor-control", "rear-predictor-raised-control")
contracts, reports = [], []
for name in names:
    folder = root / name
    contracts.append(json.loads((folder / "contract.json").read_text()))
    reports.append(json.loads((folder / "mujoco-stepping-diagnostic.json").read_text()))
assert contracts[0].pop("body_raise_m") == 0
assert contracts[1].pop("body_raise_m") == .02
assert contracts[0] == contracts[1]
out = {}
for case in ("up", "down"):
    pairs = [report[case]["rows"] for report in reports]
    assert [r["seed"] for r in pairs[0]] == [r["seed"] for r in pairs[1]]
    out[case] = {}
    for name, rows in zip(("baseline","raise_2cm"), pairs):
        out[case][name] = dict(passed=sum(r["passed"] for r in rows),
            count=len(rows), failures=dict(collections.Counter(str(r["failure"]) for r in rows)),
            median_progress_m=statistics.median(r["forward_progress_m"] for r in rows),
            max_directional_elevation_m=max((1 if case=="up" else -1)*r["signed_elevation_m"] for r in rows),
            median_measured_base_height_m=statistics.median(r["motion_base_height_m"] for r in rows if r["motion_base_height_m"] is not None))
(root / "body-raise-comparison.json").write_text(json.dumps(out,indent=2)+"\n")
print(out)
