"""Wide-to-narrow support curriculum with a real -2 m pit at every level.

Intermediate layouts are explicitly training precursors. At difficulty 1 the
geometry delegates to the original AME generator, including grid truncation.
"""
import numpy as np
from perceptive_rl import ame_terrain as original


def parameters(difficulty,kind):
    p=original.parameters(difficulty,kind)
    p['side']=1.-.8*difficulty
    # Floating point must be exactly the original values at the final level.
    p['gap']=.05+difficulty*(.15-.05 if kind=='ame_alternate' else .3-.05)
    if difficulty==1.:p=original.parameters(1.,kind)
    return p


def heightfield(difficulty,kind):
    if difficulty==1.:return original.heightfield(1.,kind)
    p=parameters(difficulty,kind);scale=p['grid'];n=round(8/scale)
    side=max(1,int(p['side']/scale));gap=max(1,int(p['gap']/scale));sep=max(0,int(p['column_gap']/scale))
    low,high=side//2,side-side//2;rng=np.random.default_rng(p['seed']);h=np.full((n,n),-2.)
    for axis in (0,1):
        for index,pos in enumerate(range(0,n,side+gap)):
            offsets=((-1 if index%2 else 1)*(sep//2),) if kind=='ame_alternate' else (-max((side+sep)//2,low),max((side+sep)//2,low))
            for offset in offsets:
                transverse=int(np.clip(n//2+offset,low,n-high));x,y=(pos,transverse) if axis==0 else (transverse,pos)
                h[max(0,x-low):min(n,x+high),max(0,y-low):min(n,y+high)]=rng.integers(-6,7)*.005
    half=round(p['platform']/(2*scale));h[n//2-half:n//2+half,n//2-half:n//2+half]=0.
    return h


def boxes(difficulty,kind):
    if difficulty==1.:return original.boxes(1.,kind)
    h=heightfield(difficulty,kind);scale=.05;result=[(0.,8.,0.,8.,-2.,-2.2)];active={}
    for ix in range(h.shape[0]+1):
        spans=set()
        if ix<h.shape[0]:
            row=h[ix];changes=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
            spans={(int(a),int(b),float(row[a])) for a,b in zip(changes[:-1],changes[1:]) if row[a]>-2.}
        for span in list(active):
            if span not in spans:
                start=active.pop(span);a,b,top=span;result.append((start*scale,ix*scale,a*scale,b*scale,top,-2.))
        for span in sorted(spans):active.setdefault(span,ix)
    return result


def lane(difficulty,cfg):
    import trimesh
    meshes=[]
    for x0,x1,y0,y1,top,bottom in boxes(difficulty,cfg.kind):
        mesh=trimesh.creation.box((x1-x0,y1-y0,top-bottom));mesh.apply_translation(((x0+x1)/2,(y0+y1)/2,(top+bottom)/2));meshes.append(mesh)
    return meshes,np.array([4.,4.,0.])


def heading_cost(env):
    import torch
    q=env.scene['robot'].data.root_quat_w
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    return 1-yaw.cos()


def curriculum(env,env_ids):
    import torch
    t=env.scene.terrain
    if not hasattr(env,'_deep_ema'):
        env._deep_ema=torch.zeros(env.num_envs,device=env.device)
        env._deep_attempts=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    success=env.termination_manager.get_term('route_complete')[env_ids]
    env._deep_ema[env_ids]=.8*env._deep_ema[env_ids]+.2*success.float()
    env._deep_attempts[env_ids]+=1
    up=success&(env._deep_attempts[env_ids]>=5)&(env._deep_ema[env_ids]>.75)
    down=(env._deep_attempts[env_ids]>=8)&(env._deep_ema[env_ids]<.1)&(t.terrain_levels[env_ids]>0)
    old=t.terrain_levels[env_ids].clone()
    t.terrain_levels[env_ids]=(old+up.long()-down.long()).clamp(0,t.max_terrain_level-1)
    changed=env_ids[old!=t.terrain_levels[env_ids]]
    env._deep_ema[changed]=0;env._deep_attempts[changed]=0
    t.env_origins[env_ids]=t.terrain_origins[t.terrain_levels[env_ids],t.terrain_types[env_ids]]
    return t.terrain_levels.float().mean()


def configure(cfg,course,training=False):
    cfg.moe_training_course=course
    if course=='wide_deep':
        from isaaclab.managers import RewardTermCfg,CurriculumTermCfg
        cfg.rewards.heading_cost=RewardTermCfg(func=heading_cost,weight=-2.)
        cfg.rewards.lane_center.weight=-2.
        if training:cfg.curriculum.terrain_levels=CurriculumTermCfg(func=curriculum)
        for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.gentle_pit=False
            if terrain.kind.startswith('ame_'):terrain.function=lane
