"""Read issued-target continuity from saved traces; never modify physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
a = p.parse_args()
results = []
for folder in sorted(a.directory.glob('*independent_28*')):
    f = folder / 'control_trace.npz'
    summary = folder / 'summary.json'
    if not f.exists() or not summary.exists():
        continue
    with np.load(f) as z:
        values = z['values']
        columns = z['columns'].tolist()
    names = [n[5:] for n in columns if n.startswith('qref:')]
    assert len(names) == 12 and len(set(names)) == 12
    q = values[:, [columns.index('qref:' + n) for n in names]]
    dt = np.diff(values[:, columns.index('time')])
    # Go2 URDF limits, in the trace's explicitly named joint order.
    assert all(any(part in n for part in ('hip_joint', 'thigh_joint', 'calf_joint')) for n in names)
    limits = np.array([15.7 if 'calf' in n else 30.1 for n in names])
    assert len(dt) and np.isfinite(q).all() and np.isfinite(dt).all() and np.all(dt > 0)
    rate = abs(np.diff(q, axis=0)) / dt[:, None] / limits
    i, j = np.unravel_index(np.argmax(rate), rate.shape)
    results.append(dict(run=folder.name, trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        samples=len(values), timestamps_strictly_increasing=True,
        max_command_rate_ratio=float(rate[i, j]), time_s=float(values[i + 1, 0]),
        joint=names[j], target_delta_rad=float(q[i + 1, j] - q[i, j]),
        passed=bool(rate.max() <= 1),
        scope='issued servo qref sampled every 2ms; includes transitions; not raw policy action continuity or visual certification',
        physics_report_passed=json.loads(summary.read_text())['passed']))
print(json.dumps(results, indent=2))