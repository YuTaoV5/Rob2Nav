"""Bounded re-query of actual Warp mesh intersections at numerical seams.

Valid original returns are untouched. No zeros, synthetic heights or geometry
changes. Failed re-queries remain nonfinite and upstream's guard still raises.
"""
import json
from pathlib import Path
import torch


def install(out):
    Path(out).mkdir(parents=True,exist_ok=True)
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh
    original=RayCaster._update_buffers_impl
    stats={'missing_rays':0,'recovered_rays':0,'unrecovered_rays':0,'max_xy_offset_m':5e-5,
           'method':'First finite actual mesh intersection at +x,-x,+y,-y offset, 50 micrometres.', 'examples':[]}
    path=Path(out)/'ray_requeries.json'

    def update(sensor,env_ids):
        original(sensor,env_ids)
        if not sensor.cfg.prim_path.endswith('/base'):return
        local_hits=sensor._data.ray_hits_w[env_ids]
        bad=(~torch.isfinite(local_hits).all(-1)).nonzero()
        if not len(bad):return
        ids=torch.arange(sensor._data.ray_hits_w.shape[0],device=local_hits.device)[env_ids]
        rows=ids[bad[:,0]];rays=bad[:,1]
        starts=sensor._ray_starts_w[rows,rays].clone()
        directions=sensor._ray_directions_w[rows,rays]
        pending=torch.ones(len(bad),device=starts.device,dtype=torch.bool)
        mesh=RayCaster.meshes[sensor.cfg.mesh_prim_paths[0]]
        stats['missing_rays']+=len(bad)
        for dx,dy in ((5e-5,0.),(-5e-5,0.),(0.,5e-5),(0.,-5e-5)):
            indices=pending.nonzero().flatten()
            if not len(indices):break
            shifted=starts[indices]+torch.tensor([dx,dy,0.],device=starts.device)
            hits=raycast_mesh(shifted,directions[indices],mesh=mesh,max_dist=sensor.cfg.max_distance)[0]
            good=torch.isfinite(hits).all(-1);chosen=indices[good]
            sensor._data.ray_hits_w[rows[chosen],rays[chosen]]=hits[good]
            if len(stats['examples'])<32:
                for j,k in enumerate(chosen[:32-len(stats['examples'])].tolist()):
                    stats['examples'].append({'env':int(rows[k]),'ray':int(rays[k]),'start':starts[k].tolist(),
                                              'offset':[dx,dy,0.],'hit':hits[good][j].tolist()})
            stats['recovered_rays']+=len(chosen)
            pending[chosen]=False
        stats['unrecovered_rays']+=int(pending.sum())
        path.write_text(json.dumps(stats,indent=2))

    RayCaster._update_buffers_impl=update
    return stats
