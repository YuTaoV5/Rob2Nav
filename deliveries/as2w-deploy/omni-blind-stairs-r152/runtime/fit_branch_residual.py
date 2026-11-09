"""Distill verified branch actions into a bounded blind residual."""
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_student import sensor_frame

LAGS=(0,2,5)
WIDTH=len(LAGS)*41+17


def network():
    return nn.Sequential(nn.Linear(WIDTH,96),nn.ELU(),nn.Linear(96,64),nn.ELU(),nn.Linear(64,17))


class Residual:
    def __init__(self,path,defaults):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        self.net=network().eval()
        self.net.load_state_dict(ck["model"])
        self.mean,self.scale=ck["mean"],ck["scale"]
        self.defaults=defaults
        self.history=deque(maxlen=6)

    @torch.inference_mode()
    def apply(self,obs,action,age,gain):
        frame=sensor_frame(obs,self.defaults)
        if not self.history:
            self.history.extend([frame]*6)
        else:
            self.history.append(frame)
        data=np.r_[*(self.history[-1-i] for i in LAGS),action,min(age/150.,1.)].astype(np.float32)
        y=self.net(((torch.from_numpy(data)-self.mean)/self.scale).clamp(-8.,8.))
        probability=float(y[16].sigmoid())
        trim=y[:16].numpy().clip(-1.,1.)
        return action+gain*max(0.,(probability-.24)/.76)*trim


def load(folder,defaults,device):
    rows=[[],[]]
    for path in sorted(folder.glob("*.npz")):
        seed=int(path.stem.split("-")[1])
        if seed<387000:
            continue
        with np.load(path,allow_pickle=False) as d:
            obs=sensor_frame(d["obs"],defaults)
            act=d["nominal"]
            active=d["active"]
            diff=d["action"]-act
        inds=np.flatnonzero(active)
        first=inds[0]
        positions=np.maximum(inds[:,None]-np.asarray(LAGS),first)
        feat=np.c_[obs[positions].reshape(len(inds),-1),act[inds],np.minimum((inds-first)/150.,1.)]
        rows[int(seed>=387006)].append((feat,diff[inds]))
    return [tuple(torch.as_tensor(np.concatenate(x),dtype=torch.float32,device=device)
                  for x in zip(*r)) for r in rows]


if __name__=="__main__":
    folder,contract,out=map(Path,sys.argv[1:])
    out.mkdir(exist_ok=True,parents=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(164003)
    torch.set_num_threads(1)
    defaults=np.asarray(json.loads(contract.read_text())["default_joint_pos"])
    (x,y),(tx,ty)=load(folder,defaults,device)
    center=x.mean(0)
    scale=x.std(0).clamp_min(.05)
    x=((x-center)/scale).clamp(-8.,8.)
    tx=((tx-center)/scale).clamp(-8.,8.)
    changed=y.abs().amax(1)>.025
    tested=ty.abs().amax(1)>.025
    yes=changed.nonzero().flatten()
    no=(~changed).nonzero().flatten()
    print("frames",len(y),len(ty),"changed",int(changed.sum()),int(tested.sum()),flush=True)
    model=network().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.00055,weight_decay=.02)
    best=float("inf")
    for i in range(1,801):
        idx=torch.cat((yes[torch.randint(len(yes),(48,),device=device)],no[torch.randint(len(no),(144,),device=device)]))
        z=model(x[idx]+.035*torch.randn_like(x[idx]))
        loss=F.smooth_l1_loss(z[:,:16],y[idx],beta=.10)+.35*F.binary_cross_entropy_with_logits(z[:,16],changed[idx].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%160==0:
            with torch.no_grad():
                val=model(tx)
                prediction=val[:,:16]*((val[:,16:].sigmoid()-.24)/.76).clamp_min(0.)
                error=float((prediction[tested]-ty[tested]).abs().mean())
                baseline=float(ty[tested].abs().mean())
                drift=float(prediction[~tested].abs().mean())
                print(i,"heldout correction error",error,"zero",baseline,"unchanged drift",drift,flush=True)
                if error<baseline and error+2.*drift<best:
                    best=error+2.*drift
                    torch.save(dict(model=model.state_dict(),mean=center,scale=scale,error=error,zero=baseline,
                                    drift=drift),out/"residual.pt")
