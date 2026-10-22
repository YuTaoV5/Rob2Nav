"""Paired frozen-policy counterfactual; diagnostic only, never model promotion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path.cwd()
source = root / 'outputs/studio/runs/3e98ef1213334615/exported'
out = root / 'outputs/studio_diagnostics/left_v12_final_terminal_compensation'
out.mkdir(parents=True, exist_ok=False)
records = []
for taper in (50, 0):
    variant = out / f'taper_{taper}'
    variant.mkdir()
    shutil.copy2(source / 'policy.pt', variant / 'policy.pt')
    meta = json.loads((source / 'policy.json').read_text())
    original = meta['fixstand_roundtrip']['terminal_hip_taper_frames']
    assert original == 50
    meta['fixstand_roundtrip']['terminal_hip_taper_frames'] = taper
    meta.update(diagnostic_only=True, capability_verified=False,
                diagnostic_scope='Frozen actor; only terminal hip residual taper changed')
    (variant / 'policy.json').write_text(json.dumps(meta, indent=2))
    cmd = [sys.executable, '-m', 'rob2nav.mujoco_skill', '--kind', 'sideflip',
           '--policy', str(variant / 'policy.pt'), '--backflip-source',
           str(root.parent / 'Rob2Nav-upstreams/GO2_backflip'), '--motion',
           str(root / 'artifacts/web_continuity_iteration_20261023/inputs/side_v12/sideflip.npz'),
           '--seconds', '8', '--physics-hz', '1000', '--fixstand-roundtrip',
           '--diagnostics', '--seed', '27100', '--output', str(variant / 'case')]
    env = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (variant / 'execution.log').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    d = json.loads((variant / 'case/summary.json').read_text())
    records.append(dict(taper=taper, policy_sha256=hashlib.sha256((variant / 'policy.pt').read_bytes()).hexdigest(),
                        metrics=d['metrics'], successes=d['successes'], constraints=d['hardware_constraints_pass']))
    (out / 'comparison.json').write_text(json.dumps(dict(diagnostic_only=True, seed=27100,
        independent_qualification=False, records=records), indent=2))
