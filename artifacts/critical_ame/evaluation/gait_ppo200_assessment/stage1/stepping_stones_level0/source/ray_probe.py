"""Reconstruct a reported invalid ray and test actual mesh intersections."""
import argparse,json,sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);p.add_argument('--verify-repair',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();root=Path(a.upstream)
for folder in ('rsl_rl','source/ame_locomotion','scripts/go2'):sys.path.insert(0,str(root/folder))
app=AppLauncher(a).app
import torch,gymnasium as gym
import ame_locomotion.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.math import quat_apply_yaw
from isaaclab.utils.warp import raycast_mesh
from isaaclab.sensors import RayCaster
cfg=load_cfg_from_registry('AME-Go2-Stage1-v0','env_cfg_entry_point');cfg.scene.num_envs=1;cfg.seed=42;cfg.sim.device=a.device or 'cuda:0'
env=gym.make('AME-Go2-Stage1-v0',cfg=cfg);e=env.unwrapped;e.reset()
sensor=e.scene['height_scanner'];_ = sensor.data
case=json.loads(Path(a.diagnostic).read_text())['records'][0]
device=e.device
q=torch.tensor(case['root_quaternion'],device=device).reshape(1,4)
pos=torch.tensor(case['root_position'],device=device).reshape(1,3)
start=quat_apply_yaw(q,sensor.ray_starts[0,case['ray']].reshape(1,3))+pos
direction=sensor.ray_directions[0,case['ray']].reshape(1,3)
mesh=RayCaster.meshes[sensor.cfg.mesh_prim_paths[0]]
records=[]
for dz in (0.,5.):
    for dx,dy in ((0.,0.),(5e-5,0.),(-5e-5,0.),(0.,5e-5),(0.,-5e-5)):
        shifted=start+torch.tensor([[dx,dy,dz]],device=device)
        hit=raycast_mesh(shifted,direction,mesh=mesh,max_dist=sensor.cfg.max_distance)[0]
        records.append({'offset':[dx,dy,dz],'finite':bool(torch.isfinite(hit).all()),'hit':hit[0].tolist()})
result={'original_case':case,'ray_start':start[0].tolist(),'direction':direction[0].tolist(),
        'terrain_origin_reconstructed':e.scene.terrain.terrain_origins[case['level'],case['type']].tolist(),'queries':records}
if a.verify_repair:
    # Exercise the actual patched RayCaster method on every ray at the failed pose.
    sensor._ray_starts_w[0]=quat_apply_yaw(q.expand(sensor.num_rays,-1),sensor.ray_starts[0])+pos
    sensor._ray_directions_w[0]=sensor.ray_directions[0]
    sensor._update_ray_infos=lambda env_ids: None
    ids=torch.tensor([0],device=device)
    RayCaster._update_buffers_impl(sensor,ids)
    before=sensor._data.ray_hits_w.clone();valid=torch.isfinite(before).all(-1)
    from .repair import install
    stats=install(Path(a.output).parent)
    RayCaster._update_buffers_impl(sensor,ids)
    after=sensor._data.ray_hits_w
    assert torch.equal(before[valid],after[valid]),'A valid original return was altered'
    assert torch.isfinite(after).all(),'A mesh re-query still missed'
    result['repair_test']={'valid_original_rays_unchanged':True,'all_requeries_finite':True,
        'original_missing_rays':int((~valid).sum()),'statistics':stats}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
env.close();app.close()
