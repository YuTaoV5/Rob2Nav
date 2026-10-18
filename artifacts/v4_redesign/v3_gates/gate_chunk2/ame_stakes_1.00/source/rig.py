"""Explicit candidate mounts and body-occlusion filter for student observations.

The torso box is conservative geometry, not a calibrated Go2/bracket/leg model.
Depth uses a downsampled 848x480-like FoV with a 0.195 m optical min-Z assumption.
"""
import math
import torch

RIGS={'original':{'camera':(.25,0.,.06),'pitch':20.,'lidar':(0.,0.,.16),'roll':0.},
      'near_down60':{'camera':(.30,0.,.06),'pitch':60.,'lidar':(0.,0.,.16),'roll':0.},
      'near_down60_front_lidar':{'camera':(.30,0.,.06),'pitch':60.,'lidar':(.30,0.,.12),'roll':180.}}


def configure(cfg,name):
    spec=RIGS[name];cfg.moe_sensor_rig=name
    cfg.scene.depth_rays.offset.pos=spec['camera']
    cfg.scene.depth_rays.pattern_cfg.pitch_deg=spec['pitch']
    cfg.scene.lidar_rays.offset.pos=spec['lidar']
    cfg.scene.lidar_rays.offset.rot=(1.,0.,0.,0.) if spec['roll']==0 else (0.,1.,0.,0.)


def filter_hits(env,hits):
    from isaaclab.utils.math import quat_apply_inverse
    name=getattr(env.cfg,'moe_sensor_rig','original')
    if name=='original':return hits  # Preserve historical experiment contract.
    spec=RIGS[name];r=env.scene['robot'];n=hits.shape[1]
    rel=hits-r.data.root_pos_w[:,None]
    body=quat_apply_inverse(r.data.root_quat_w[:,None].expand(-1,n,-1),rel)
    cam_n=env.scene['depth_rays'].data.ray_hits_w.shape[1]
    origin=torch.tensor([spec['camera']]*cam_n+[spec['lidar']]*(n-cam_n),device=hits.device)[None]
    segment=body-origin;length=segment.norm(dim=-1)
    valid=torch.isfinite(body).all(-1)
    p=math.radians(spec['pitch'])
    optical=segment[:,:cam_n,0]*math.cos(p)-segment[:,:cam_n,2]*math.sin(p)
    valid[:,:cam_n]&=optical>=.195
    valid[:,cam_n:]&=length[:,cam_n:]>=.2
    # Segment-AABB intersections before the observed terrain surface.
    safe=torch.where(segment.abs()<1e-8,torch.full_like(segment,1e-8),segment)
    a=(torch.tensor([-.24,-.10,-.06],device=hits.device)-origin)/safe
    b=(torch.tensor([.24,.10,.07],device=hits.device)-origin)/safe
    near=torch.minimum(a,b).amax(-1);far=torch.maximum(a,b).amin(-1)
    blocked=(far>=near.clamp(min=0))&(far>0)&(near<1.)
    valid&=~blocked
    return torch.where(valid[...,None],hits,torch.inf)
