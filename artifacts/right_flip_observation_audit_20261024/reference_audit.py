"""Read-only reference derivative audit; run from repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
args = parser.parse_args()
with np.load(args.reference, allow_pickle=False) as motion:
    q = motion["joint_pos"].astype(float)
    velocity = motion["joint_vel"].astype(float)
    fps = float(motion["fps"][0])
assert q.shape == velocity.shape and fps > 0
assert np.isfinite(q).all() and np.isfinite(velocity).all()
gradient = np.gradient(q, 1 / fps, axis=0)
acceleration = np.diff(velocity, axis=0) * fps
frame, joint = np.unravel_index(np.abs(acceleration).argmax(), acceleration.shape)
print(json.dumps({
    "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
    "frames": len(q), "fps": fps,
    "velocity_gradient_max_error": float(np.abs(velocity-gradient).max()),
    "peak_acceleration_rad_s2": float(np.abs(acceleration).max()),
    "peak_acceleration_interval_frames": [int(frame), int(frame+1)],
    "peak_acceleration_joint_index": int(joint),
    "velocity_step_max_rad_s": float(np.abs(np.diff(velocity,axis=0)).max()),
    "physics_validation": False,
}, indent=2))
