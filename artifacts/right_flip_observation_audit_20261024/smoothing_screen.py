"""Offline sensitivity screen only; does not produce deployment references."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import savgol_filter

run = Path("outputs/studio/runs/fbf788e34b284900")
reference = Path("artifacts/web_continuity_iteration_20261023/inputs/right_v1/rightflip.npz")
with np.load(reference, allow_pickle=False) as z:
    q, v = z["joint_pos"].astype(float), z["joint_vel"].astype(float)
    fps = float(z["fps"][0])
torch.set_num_threads(1)
actor = torch.jit.load(str(run / "exported/policy.pt"), map_location="cpu").eval()
results = []
for window in (5, 7, 11):
    smooth = savgol_filter(q, window, 3, axis=0)
    velocity = np.gradient(smooth, 1/fps, axis=0)
    result = dict(window=window, position_change_max=float(abs(smooth-q).max()),
                  peak_acceleration=float(abs(np.diff(velocity,axis=0)*fps).max()), seeds=[])
    for seed in range(20200, 20204):
        trace = json.loads((run / f"seed_{seed}/trace.json").read_text())
        obs = np.array([row["observation"] for row in trace], dtype=np.float32)
        idx = np.minimum(np.arange(len(trace)), len(q)-1)
        assert np.array_equal(obs[:,:12], q[idx])
        assert np.array_equal(obs[:,12:24], v[idx])
        changed = obs.copy()
        changed[:,:12], changed[:,12:24] = smooth[idx], velocity[idx]
        with torch.inference_mode():
            baseline = actor(torch.from_numpy(obs)).numpy()
            candidate = actor(torch.from_numpy(changed)).numpy()
        recorded = np.array([row["action"] for row in trace])
        error = float(abs(baseline-recorded).max())
        assert error < 1e-5, error
        result["seeds"].append(dict(seed=seed, replay_max_error=error,
            original_jump=float(.5*(baseline[139,3]-baseline[138,3])),
            candidate_jump=float(.5*(candidate[139,3]-candidate[138,3])),
            max_actor_step=float(abs(np.diff(candidate,axis=0)*.5).max())))
    results.append(result)
print(json.dumps(dict(offline_only=True, closed_loop=False, results=results), indent=2))
