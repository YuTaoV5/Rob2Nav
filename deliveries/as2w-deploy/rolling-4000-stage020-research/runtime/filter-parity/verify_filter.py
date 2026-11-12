"""Compare the standalone C++ filter against Python on captured actor outputs."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

executable, audit_script, package, baseline, candidate, audit_output, parity_output = map(Path, sys.argv[1:])
sys.argv = [str(audit_script), str(package), str(baseline), str(candidate), str(audit_output)]
audit = runpy.run_path(str(audit_script))
raw = torch.from_numpy(audit["candidate"][:, :12].copy())
vx = torch.from_numpy(audit["vx"].astype(np.float32))
wz = torch.from_numpy(audit["wz"].astype(np.float32))
resets = np.r_[True, ~audit["adjacent"]]
rng = np.random.default_rng(311)
stand = torch.from_numpy(rng.uniform(-.4, .4, size=(len(raw), 12)).astype(np.float32))
synthetic = torch.tensor([[0.] * 12, [20.] * 12, [-20.] * 12, [20.] * 12, [0.] * 12])
raw = torch.cat((raw, synthetic))
vx = torch.cat((vx, torch.tensor([.20, .20, .20, .0, .0])))
wz = torch.cat((wz, torch.tensor([0., 0., 0., -.1, .0])))
stand = torch.cat((stand, torch.zeros_like(synthetic)))
resets = np.r_[resets, True, False, False, False, True]

frames = []
expected = []
filtered = torch.zeros(12)
target = torch.zeros(12)
alpha = -np.expm1(-2. * np.pi * 5. * .020)
limit = .9 * .020 / .35
for reset, x, yaw, action, offset in zip(resets, vx, wz, raw, stand):
    if reset:
        filtered.zero_()
        target.zero_()
    filtered = torch.lerp(filtered, action.clamp(-6., 6.), alpha)
    motion = max(abs(float(x)) / .10, abs(float(yaw)) / .20)
    desired = offset + .18 * min(1., motion) * filtered.clamp(-1., 1.)
    target = target + (desired - target).clamp(-limit, limit)
    expected.append(target.numpy().copy())
    values = [int(reset), float(x), float(yaw), *action.tolist(), *offset.tolist()]
    frames.append(" ".join(map(str, values)))

run = subprocess.run([str(executable.resolve())], input="\n".join(frames) + "\n",
                     text=True, capture_output=True, check=True)
actual = np.fromstring(run.stdout, sep=" ", dtype=np.float32).reshape(-1, 12)
expected = np.asarray(expected)
assert actual.shape == expected.shape
error = float(np.max(abs(actual - expected)))
assert error < 1e-6, error
report = dict(frames=len(frames), resets=int(resets.sum()), synthetic_frames=len(synthetic),
              max_error_normalized=error, max_error_rad=error*.35,
              scope="Leg residual filter and target slew only. Stand offsets supplied as test inputs.")
parity_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
