"""CSV -> Go2 joint/body reference arrays. FK preparation, not a dynamics demonstration."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--backflip-source', required=True)
p.add_argument('--csv', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--fps', type=float, default=50.)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.kit_args = (args.kit_args+' --enable isaacsim.asset.importer.urdf').strip()
app = AppLauncher(args).app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from .common import setup_sources, robot_config, CSV_JOINTS, sha, write_json, BACKFLIP_COMMIT


def main():
    if args.output.exists():
        raise FileExistsError(args.output)
    source = setup_sources(args.backflip_source)
    robot_cfg = robot_config(source)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/args.fps, device=args.device or 'cuda:0'))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.))
    sim.reset()
    robot = scene['robot']
    raw = np.loadtxt(args.csv, delimiter=',', dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 19 or not np.isfinite(raw).all():
        raise ValueError('Expected finite XYZ + XYZW + 12 SDK-order joint positions')
    joint_cols = [7+CSV_JOINTS.index(name) for name in robot.joint_names]
    q = raw[:, joint_cols]
    dq = np.gradient(q, 1/args.fps, axis=0)
    xyz = raw[:, :3]
    quat = raw[:, [6,3,4,5]]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    lin = np.gradient(xyz, 1/args.fps, axis=0)
    rotations = Rotation.from_quat(raw[:, 3:7])
    angular = (rotations[1:]*rotations[:-1].inv()).as_rotvec()*args.fps
    angular = np.concatenate([angular, angular[-1:]], axis=0).astype(np.float32)
    log = {k:[] for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')}
    for i in range(len(raw)):
        root = robot.data.default_root_state.clone()
        root[0] = torch.as_tensor(np.r_[xyz[i], quat[i], lin[i], angular[i]], device=sim.device)
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(torch.as_tensor(q[i:i+1], device=sim.device),
            torch.as_tensor(dq[i:i+1], device=sim.device))
        sim.render()  # Like the upstream converter: update FK, deliberately no physics integration.
        scene.update(sim.get_physics_dt())
        for key in log:
            log[key].append(getattr(robot.data, key)[0].cpu().numpy().copy())
    arrays = {key:np.stack(value) for key,value in log.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Reference contains nonfinite values')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, fps=np.array([args.fps]), **arrays)
    write_json(args.output.with_suffix('.json'), dict(role='FK reference preprocessing; not physical skill validation',
        csv_sha256=sha(args.csv), npz_sha256=sha(args.output), frames=len(raw), fps=args.fps,
        csv_joint_names=CSV_JOINTS, joint_names=robot.joint_names, body_names=robot.body_names,
        asset_commit=BACKFLIP_COMMIT, asset_pd=dict(kp=40,kd=1), root_quaternion_order='wxyz'))
    print(f'FK_REFERENCE_SAVED {args.output} frames={len(raw)} joints={robot.joint_names} bodies={robot.body_names}', flush=True)


try:
    main()
except Exception:
    import traceback, os, sys
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)  # Kit shutdown must not replace a failed preprocessing status with zero.
finally:
    # All output arrays/files are already synchronized and closed. Kit graceful
    # teardown can block in headless FK-only sessions; use its documented immediate exit.
    app.close(wait_for_replicator=False, skip_cleanup=True)
