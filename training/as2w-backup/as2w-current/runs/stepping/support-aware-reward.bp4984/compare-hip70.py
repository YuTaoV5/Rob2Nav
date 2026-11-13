import collections
import json
import statistics
from pathlib import Path

root = Path(__file__).parent
def load(folder):
    path = root / folder
    contract = json.loads((path / "contract.json").read_text())
    gains = (contract.pop("leg_pd_group"), contract.pop("leg_pd"))
    rows = json.loads((path / "mujoco-stepping-diagnostic.json").read_text())
    return contract, gains, rows

baseline_contract, baseline_gains, baseline = load("support-aware-matched-stairs-800")
test_contract, test_gains, test = load("support-aware-hip70-stairs-800")
assert baseline_contract == test_contract
assert baseline_gains == ("all", [55., 2.]) and test_gains == ("hip", [70., 2.])
result = {}
for mode in ("up", "down"):
    result[mode] = {}
    assert [r["seed"] for r in baseline[mode]["rows"]] == [r["seed"] for r in test[mode]["rows"]]
    for name, data in (("baseline", baseline), ("hip70", test)):
        rows = data[mode]["rows"]
        failures = [r["failure_time_s"] for r in rows if r["failure_time_s"] is not None]
        result[mode][name] = dict(
            trials=len(rows), passed=sum(r["passed"] for r in rows),
            failures=dict(collections.Counter(str(r["failure"]) for r in rows)),
            fr_hip_limit=sum(any(j["name"] == "FR_hip" for j in r["failure_joints"]) for r in rows),
            median_failure_time_s=statistics.median(failures) if failures else None,
            median_progress_m=statistics.median(r["forward_progress_m"] for r in rows),
            max_abs_elevation_m=max(abs(r["signed_elevation_m"]) for r in rows),
            max_effort_fraction_of_guard=max(r["max_effort_fraction_of_guard"] for r in rows),
        )
(root / "hip70-stairs-800-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
