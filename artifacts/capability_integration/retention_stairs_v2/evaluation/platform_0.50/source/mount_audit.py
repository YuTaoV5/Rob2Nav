"""Geometric mount sweep with exact box terrain and conservative torso occlusion.

The straight, level probe trajectory is a visibility experiment, not locomotion.
No terrain truth enters ElevationMemory; truth is used only to score its output.
Legs, brackets, optical artifacts and odometry errors are not modeled here.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from perceptive_rl.ame_terrain import boxes
from .memory import ElevationMemory


def rotation(pitch=0.,roll=0.,device='cpu'):
    p,r=math.radians(pitch),math.radians(roll)
    ry=torch.tensor([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]],device=device)
    rx=torch.tensor([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]],device=device)
    return ry@rx


def directions(camera,device):
    if camera:
        u=(torch.arange(64,device=device)+.5-32)/32*math.tan(math.radians(43.5))
        v=(torch.arange(32,device=device)+.5-16)/16*math.tan(math.radians(29.))
        vv,uu=torch.meshgrid(v,u,indexing='ij')
        d=torch.stack((torch.ones_like(uu),-uu,-vv),-1).reshape(-1,3)
    else:
        aa,ee=torch.meshgrid(torch.arange(96,device=device)*2*math.pi/96,
                            torch.linspace(math.radians(-7),math.radians(52),12,device=device),indexing='ij')
        d=torch.stack((ee.cos()*aa.cos(),ee.cos()*aa.sin(),ee.sin()),-1).reshape(-1,3)
    return torch.nn.functional.normalize(d,dim=-1)


def cast(origins,dirs,bounds):
    # [profiles,rays,boxes,axes], return nearest positive surface intersection.
    safe=torch.where(dirs.abs()<1e-8,torch.full_like(dirs,1e-8),dirs)
    t1=(bounds[None,None,:,0]-origins[:,:,None])/safe[:,:,None]
    t2=(bounds[None,None,:,1]-origins[:,:,None])/safe[:,:,None]
    near=torch.minimum(t1,t2).amax(-1);far=torch.maximum(t1,t2).amin(-1)
    distance=torch.where((far>=near.clamp(min=0))&(far>0),torch.where(near>0,near,far),torch.inf)
    return distance.amin(-1)


def terrain_bounds(kind,difficulty,device):
    data=[(-10,10,-10,10,0.,-.2)] if kind=='flat' else boxes(difficulty,kind)
    out=torch.tensor([[[x0,y0,bottom],[x1,y1,top]] for x0,x1,y0,y1,top,bottom in data],device=device)
    if kind!='flat':out[:,:,:2]-=4.
    return out


def truth_height(xy,bounds):
    inside=((xy[...,None,:]>=bounds[:,0,:2])&(xy[...,None,:]<=bounds[:,1,:2])).all(-1)
    return torch.where(inside,bounds[:,1,2],-torch.inf).amax(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=a.device
    profiles=[{'camera_x':.25,'camera_z':.06,'camera_pitch':20,'lidar_x':0.,'lidar_z':.16,'lidar_roll':0,'name':'v1_original'}]
    for pitch in (30,45,60):
        for cx in (.22,.30):
            for roll in (0,180):
                profiles.append({'camera_x':cx,'camera_z':.06,'camera_pitch':pitch,'lidar_x':0.,'lidar_z':.16,'lidar_roll':roll,
                                 'name':f'cam{pitch}_x{cx:.2f}_lidar{roll}'})
    for lx,lz in ((.30,.12),(0.,.30)):
        for roll in (0,180):
            profiles.append({'camera_x':.30,'camera_z':.06,'camera_pitch':45,'lidar_x':lx,'lidar_z':lz,'lidar_roll':roll,
                             'name':f'cam45_lidar_x{lx:.2f}_z{lz:.2f}_roll{roll}'})
    n=len(profiles);cam=directions(True,device);lidar=directions(False,device)
    rays=torch.stack([torch.cat((cam@rotation(p['camera_pitch'],device=device).T,
                                lidar@rotation(roll=p['lidar_roll'],device=device).T)) for p in profiles])
    offsets=torch.tensor([[[p['camera_x'],0.,p['camera_z']]]*len(cam)+[[p['lidar_x'],0.,p['lidar_z']]]*len(lidar) for p in profiles],device=device)
    body=torch.tensor([[[-.24,-.10,-.06],[.24,.10,.07]]],device=device)
    body_distance=cast(offsets,rays,body)
    y,x=torch.meshgrid(torch.linspace(-.5,.5,21,device=device),torch.linspace(-.55,1.05,33,device=device),indexing='ij')
    grid=torch.stack((x.flatten(),y.flatten()),-1)
    region=(grid[:,0]>=-.4)&(grid[:,0]<=.75)&(grid[:,1].abs()<=.3)
    near_region=(grid[:,0]>=-.4)&(grid[:,0]<=.3)&(grid[:,1].abs()<=.3)
    max_range=torch.tensor([4.]*len(cam)+[8.]*len(lidar),device=device)[None]
    min_range=torch.tensor([.195]*len(cam)+[.2]*len(lidar),device=device)[None]
    rows=[]
    for kind,difficulty in [('flat',0.),('ame_stakes',0.),('ame_stakes',1.),('ame_alternate',1.)]:
        bounds=terrain_bounds(kind,difficulty,device)
        for lifetime in (1.,4.):
            memory=ElevationMemory(n,device,lifetime);stats=[]
            for step in range(71):
                pos=torch.tensor([step*.05,0.,.34],device=device)[None].expand(n,-1)
                origins=offsets+pos[:,None];distance=cast(origins,rays,bounds)
                # D435 minimum is optical forward Z, not Euclidean ray range.
                optical=distance.clone()
                optical[:,:len(cam)]*=cam[:,0][None]
                valid=(optical>=min_range)&(distance<=max_range)&(distance<body_distance)
                hits=torch.where(valid[...,None],origins+rays*distance[...,None],torch.inf)
                observed=memory.update(hits,pos,torch.zeros(n,device=device),step*.1,torch.ones(n,device=device,dtype=torch.bool)).reshape(n,3,693)
                gt=truth_height(grid+pos[0,:2],bounds);support=gt>-.3
                predicted=.34-.5-observed[:,0];correct=(observed[:,1]>.5)&((predicted-gt).abs()<.04)
                metric=lambda reg:(correct[:,support&reg].float().mean(-1) if (support&reg).any() else torch.zeros(n,device=device))
                false=(observed[:,1]>.5)&(predicted>-.3)&(~support)[None]
                stats.append(torch.stack((metric(region),metric(near_region),false.float().mean(-1)),1))
            values=torch.stack(stats).cpu().numpy()
            for i,profile in enumerate(profiles):
                rows.append({**profile,'kind':kind,'difficulty':difficulty,'memory_lifetime':lifetime,
                    'initial_near_support_recall':float(values[0,i,1]),
                    'mean_support_recall':float(values[:,i,0].mean()),'mean_near_support_recall':float(values[:,i,1].mean()),
                    'false_support_cell_fraction':float(values[:,i,2].mean())})
    result={'assumptions':{'base_height':.34,'torso_box':body.cpu().tolist(),'camera_rays':len(cam),'lidar_rays':len(lidar),
        'camera_optical_min_z':.195,'lidar_reliable_min_range':.2,'height_tolerance':.04,
        'trajectory':'level x=0..3.5 m at 0.5 m/s, 10 Hz; visibility only, not dynamically feasible traversal',
        'limitations':'torso box only; no legs/bracket optics/motion/odometry errors; approximate deterministic lidar pattern'},'rows':rows}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    scored=[]
    for profile in profiles:
        cases=[r for r in rows if r['name']==profile['name'] and r['memory_lifetime']==4.]
        scored.append({'name':profile['name'],'near_recall':float(np.mean([r['mean_near_support_recall'] for r in cases])),
                       'preview_recall':float(np.mean([r['mean_support_recall'] for r in cases]))})
    print(json.dumps(sorted(scored,key=lambda r:r['near_recall'],reverse=True),indent=2))


if __name__=='__main__':main()
