"""Read-only policy-clock transition audit; never certifies physical success.
Run from repository root: python artifacts/backflip_rebound_training_20261024/audit_single_entry_transition.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "landing_single_entry_policy_trace_28600.json"
rows = json.loads(source.read_text(encoding="utf-8"))
names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
assert len(rows) == 450
steps = []
for previous, current in zip(rows, rows[1:]):
    assert abs(current["time_s"] - previous["time_s"] - .02) < 1e-8
    delta = [abs(max(-8., min(8., x)) - max(-8., min(8., y))) * .5
             for x, y in zip(current["mapped_action"], previous["mapped_action"])]
    joint = max(range(12), key=delta.__getitem__)
    steps.append(dict(time_s=current["time_s"], previous_clock=previous["phase_clock"],
                      clock=current["phase_clock"], joint=names[joint],
                      requested_target_delta_rad=delta[joint]))
transitions = [row for row in steps if row["clock"] < row["previous_clock"] - .1]
result = dict(source=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              samples=len(rows), policy_hz=50, action_scale=.5, action_clip=8,
              provenance="Corrected single-entry frozen-parent counterfactual, seed 28600",
              quantity="Requested joint target before 20ms latency and physical-substep slew limiting; NOT issued servo target or measured velocity",
              clock_rewinds=transitions,
              first_landing_window=[row for row in steps if 1.98 <= row["time_s"] <= 2.16],
              causal_limitation="Clock and physical observation change together; this audit does not isolate clock-only causation",
              capability_verified=False)
print(json.dumps(result, ensure_ascii=False, indent=2))
