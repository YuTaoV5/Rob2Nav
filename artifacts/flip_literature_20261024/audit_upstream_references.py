"""Read-only kinematic audit of pinned public Go2 trajectories; never a physics acceptance."""
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import numpy as np

REPO = 'yusongmin1/go2_flip_TO'
COMMIT = 'd12fb093dc391d39815b252eb209a6e4c0692726'
BASE = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/'
NAMES = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
         for joint in ('hip', 'thigh', 'calf')]
FILES = [f'datasets/go2/mocap_motions_go2/quad_{kind}_50hz.txt'
         for kind in ('backflip', 'frontflip', 'sideflip', 'sideflip_right')]
URDF = 'src/robots/go2/go2/urdf/go2.urdf'


def fetch(path):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(BASE + path, timeout=40) as response:
                return path, response.read()
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise


def audit(data, limits):
    x = json.loads(data)
    a = np.asarray(x['Frames'], dtype=float)
    dt = float(x['FrameDuration'])
    assert a.ndim == 2 and a.shape[1] == 49 and np.isfinite(a).all()
    q = a[:, 7:19]
    declared = a[:, 37:49]
    velocity = np.diff(q, axis=0) / dt
    speed = np.array([limits[n]['velocity'] for n in NAMES])
    lo = np.array([limits[n]['lower'] for n in NAMES])
    hi = np.array([limits[n]['upper'] for n in NAMES])
    quat = a[:, 3:7]
    norms = np.linalg.norm(quat, axis=1)
    quat = quat / norms[:, None]
    # Relative world rotations q[k+1] * conjugate(q[k]); xyzw convention.
    v1, w1 = quat[1:, :3], quat[1:, 3]
    v0, w0 = -quat[:-1, :3], quat[:-1, 3]
    v = w1[:, None]*v0 + w0[:, None]*v1 + np.cross(v1, v0)
    w = w1*w0 - np.sum(v1*v0, axis=1)
    sign = np.where(w < 0, -1., 1.)
    v *= sign[:, None]
    w *= sign
    n = np.linalg.norm(v, axis=1)
    angle = 2*np.arctan2(n, w)
    rotvec = v * (angle / np.maximum(n, 1e-12))[:, None]
    return {
        'samples': len(a), 'dt': dt, 'duration_seconds': (len(a)-1)*dt,
        'quaternion_max_norm_error': float(np.max(np.abs(norms-1))),
        'root_displacement_m': (a[-1, :3]-a[0, :3]).tolist(),
        'initial_to_terminal_joint_max_difference_rad': float(np.max(np.abs(q[-1]-q[0]))),
        'joint_position_limit_excess_rad': float(max(0, np.max(lo-q), np.max(q-hi))),
        'declared_joint_velocity_max_urdf_ratio': float(np.max(np.abs(declared)/speed)),
        'finite_difference_joint_velocity_max_urdf_ratio': float(np.max(np.abs(velocity)/speed)),
        'finite_difference_minus_declared_left_sample_max_rad_s': float(np.max(np.abs(velocity-declared[:-1]))),
        'finite_difference_minus_declared_right_sample_max_rad_s': float(np.max(np.abs(velocity-declared[1:]))),
        'velocity_alignment_note': 'Semi-implicit integration uses the NEXT node velocity for the interval. Left-sample disagreement alone is not an export defect.',
        'integrated_world_rotation_increments_rad': np.sum(rotvec, axis=0).tolist(),
        'angular_path_length_rad': float(np.sum(angle)),
        'terminal_orientation_distance_rad': float(2*np.arccos(np.clip(abs(quat[0]@quat[-1]), 0, 1))),
        'physics_verified': False,
    }


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        data = dict(pool.map(fetch, [URDF]+FILES))
    limits = {}
    for j in ET.fromstring(data[URDF]).findall('joint'):
        lim = j.find('limit')
        if j.attrib['name'] in NAMES:
            limits[j.attrib['name']] = {k: float(lim.attrib[k]) for k in ('lower', 'upper', 'velocity')}
    assert set(NAMES) <= set(limits)
    result = {
        'repository': REPO, 'commit': COMMIT,
        'scope': 'Stored 50Hz trajectory kinematics against the optimization URDF only. No solver run, torque/impact check, closed-loop rollout, or hardware qualification.',
        'joint_order': NAMES, 'limits': limits,
        'sources': {p: {'url': BASE+p, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
                    for p, b in data.items()},
        'results': {p: audit(data[p], limits) for p in FILES},
    }
    target = Path(__file__).with_name('upstream_reference_audit.json')
    target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    for p, x in result['results'].items():
        print(Path(p).name, json.dumps(x))


if __name__ == '__main__':
    main()
