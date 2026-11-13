import json
from pathlib import Path

root = Path(__file__).parent
def load(mode):
    return json.loads((root / mode / "mujoco-recovery-diagnostic.json").read_text())["prone"]["rows"]

original = load("original")
extended = load("extended")
assert len(original) == len(extended)
comparison = []
for index, (a, b) in enumerate(zip(original, extended)):
    assert a["push_times_s"] == b["push_times_s"]
    comparison.append({
        "trial": index,
        "push_times_s": a["push_times_s"],
        "original_pass": a["passed"],
        "extended_pass": b["passed"],
        "original_final_fixed_s": a["fixed_final_window_s"],
        "extended_final_fixed_s": b["fixed_final_window_s"],
        "original_longest_fixed_s": a["longest_fixed_window_s"],
        "original_speed_gate_events": a["final_speed_gate_events"],
        "extended_first_failure": b["first_failure"],
    })
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print("original", sum(row["original_pass"] for row in comparison), "/", len(comparison))
print("extended", sum(row["extended_pass"] for row in comparison), "/", len(comparison))
for row in comparison:
    print(row["trial"], row["push_times_s"], row["original_final_fixed_s"], row["extended_final_fixed_s"], row["extended_first_failure"])
