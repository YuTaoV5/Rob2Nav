import json
import sys
from pathlib import Path
import numpy as np

root = Path("runs")
reference_dir = root / "r290-three-skills/go2-as2w-wheel-teacher-projected"
trace_path = root / "stepping/stairs-left-rear-trace.3ids8M/mujoco-stepping-diagnostic.json"
traces = json.loads(trace_path.read_text())
result = {}
for name, case in traces.items():
    teacher = "stairs_down" if "down" in name else "stairs_up8cm"
    with np.load(reference_dir / f"{teacher}.npz") as data:
        contact = data["go2_contact"].astype(float)
        intervals = np.diff(data["time_s"])
        support_count = contact.sum(axis=1)
    trace = case["rows"][0]["support_trace"]
    age = np.array([row["time_s"] for row in trace]) - 2
    observed = np.array([row["load_n"] for row in trace]) > 4
    frame = (age.clip(min=0) * 25).astype(int)
    use = (age >= 0) & (frame < len(contact))
    sample = frame[use]
    observed = observed[use]
    expected = sum(contact[(sample + shift).clip(0, len(contact)-1)] for shift in (-2, 0, 2)) / 3
    result[name] = {
        "trace_frames": int(use.sum()),
        "reference_duration_s": float(intervals.sum()),
        "reference_fraction_under_three_supports": float((support_count < 3).mean()),
        "reference_support_count_histogram": np.bincount(support_count.astype(int), minlength=5).tolist(),
        "reference_interval_median_s": float(np.median(intervals)),
        "reference_interval_p99_s": float(np.quantile(intervals, .99)),
        "reward_window_s": float(len(contact)/25),
        "reference_support_duty": expected.mean(axis=0).tolist(),
        "observed_support_duty": observed.mean(axis=0).tolist(),
        "always_contact_fit": float(expected.mean()),
        "observed_fit": float(1 - np.abs(observed - expected).mean()),
        "observed_leg_fit": (1 - np.abs(observed - expected).mean(axis=0)).tolist(),
        "expected_swing_fraction_observed_unloaded": float(((1 - expected) * ~observed).sum() / (1 - expected).sum()),
    }
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
