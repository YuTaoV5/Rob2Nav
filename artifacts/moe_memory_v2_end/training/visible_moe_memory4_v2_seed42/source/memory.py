"""5 cm visible-return elevation memory; no mesh or hidden-surface queries."""
import torch


class ElevationMemory:
    def __init__(self,count,device,lifetime=1.):
        self.world=torch.zeros(count,693,3,device=device)
        self.time=torch.full((count,693),-1e6,device=device);self.lifetime=lifetime

    def reset(self,ids):self.time[ids]=-1e6

    def update(self,returns,position,yaw,now,fresh):
        hits=torch.cat((self.world,returns),1)
        times=torch.cat((self.time,torch.full_like(returns[...,2],now)),1)
        rel=hits-position[:,None]
        x=rel[...,0]*yaw.cos()[:,None]+rel[...,1]*yaw.sin()[:,None]
        y=-rel[...,0]*yaw.sin()[:,None]+rel[...,1]*yaw.cos()[:,None]
        ix=((x+.55)/.05).round().long();iy=((y+.5)/.05).round().long()
        valid=torch.isfinite(hits).all(-1)&(ix>=0)&(ix<33)&(iy>=0)&(iy<21)
        valid[:,:693]&=now-times[:,:693]<self.lifetime
        valid[:,693:]&=fresh[:,None]
        index=(iy*33+ix).clamp(0,692)
        heights=torch.where(valid,hits[...,2],-torch.inf)
        top=torch.full_like(self.time,-torch.inf)
        top.scatter_reduce_(1,index,heights,reduce='amax',include_self=True)
        mask=torch.isfinite(top);winners=valid&(heights==top.gather(1,index))
        newest=torch.full_like(self.time,-torch.inf)
        newest.scatter_reduce_(1,index,torch.where(winners,times,-torch.inf),reduce='amax',include_self=True)
        winners&=times==newest.gather(1,index)
        n=hits.shape[1];ids=torch.arange(n,device=hits.device)[None].expand(hits.shape[0],-1)
        selected=torch.full_like(index[:,:693],n)
        selected.scatter_reduce_(1,index,torch.where(winners,ids,n),reduce='amin',include_self=True)
        selected=selected.clamp(max=n-1)
        self.world=hits.gather(1,selected[...,None].expand(-1,-1,3))
        self.time=torch.where(mask,times.gather(1,selected),-1e6)
        relative=torch.where(mask,(position[:,2,None]-top-.5).clamp(-1.5,2.),0.)
        age=torch.where(mask,((now-self.time)/self.lifetime).clamp(0,1),1.)
        return torch.cat((relative,mask.float(),age),-1)
