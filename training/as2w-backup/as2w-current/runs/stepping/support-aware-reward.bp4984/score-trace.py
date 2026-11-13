import json
from pathlib import Path
import numpy as np

source = Path("runs/stepping/stairs-left-rear-trace.3ids8M/mujoco-stepping-diagnostic.json")
for case, result in json.loads(source.read_text()).items():
    trace = result["rows"][0]["support_trace"]
    time = np.array([row["time_s"] for row in trace])
    contact = np.array([row["load_n"] for row in trace]) > 4
    gap = np.array([row["wheel_gap_m"] for row in trace])
    # Match the 8-second diagnostic gait and mask out the initial fixed stand.
    phase = ((time - 2).clip(min=0) / 8) % 1
    local = (12 * phase[:, None] - 3 * np.array([0, 2, 3, 1])) % 12
    lift = np.clip((local - .85) / .65, 0, 1) * np.clip((2.48 - local) / .65, 0, 1)
    support = contact.sum(axis=1, keepdims=True) - contact
    score = np.clip(gap / np.maximum(.035 * lift, .015), 0, 1) * lift
    score *= ~contact & (support == 3) & (time[:, None] >= 2)
    print(case, "positive_frames", (score > 0).sum(axis=0).tolist(), "peak", score.max(axis=0).round(3).tolist())
