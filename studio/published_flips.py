"""Pinned, inference-only FIXSTAND candidates for the Studio live viewer."""
import hashlib
import json
from pathlib import Path


def live_flip(root, skill, source):
    root = Path(root).resolve()
    manifest = json.loads((root / 'studio/published_flips.json').read_text(encoding='utf-8'))
    row = manifest[skill]
    for relative, expected in row['sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Published flip resource is missing: ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Published flip resource hash mismatch: ' + relative)
    policy = root / row['policy']
    common = ['--physics-hz', '1000', '--seconds', str(row['seconds']),
              '--fixstand-roundtrip', '--seed', str(row['seed'])]
    if skill == 'backflip':
        module = 'skills.backflip.simulate'
        args = ['--source', str(source), '--onnx', str(policy), *common]
    else:
        module = 'rob2nav.mujoco_skill'
        args = ['--kind', 'sideflip', '--policy', str(policy), '--backflip-source', str(source),
                '--motion', str(root / row['reference']), *common]
    return module, args, policy, row['note']
