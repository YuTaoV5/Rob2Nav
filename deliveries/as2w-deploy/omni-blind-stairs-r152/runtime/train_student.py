"""Bootstrap one proprioceptive recurrent actor from qualified AS2-W demonstrations."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sensor_frame(obs, defaults):
    return np.concatenate((obs[..., :3]+np.array((.4,0.,0.)),obs[..., 3:9],
                           obs[..., 10:26]+defaults,obs[..., 26:42]),axis=-1).astype(np.float32)


class BlindStudent(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("mean",mean)
        self.register_buffer("scale",scale)
        self.encoder=nn.Sequential(nn.Linear(41,96),nn.ELU())
        self.memory=nn.GRU(96,96,batch_first=True)
        self.actor=nn.Sequential(nn.Linear(96,96),nn.ELU(),nn.Linear(96,16))

    def forward(self, x, state=None):
        x=self.encoder(((x-self.mean)/self.scale).clamp(-12.,12.))
        x,state=self.memory(x,state)
        return self.actor(x).clamp(-6.,6.),state


def prepare(paths, defaults, device, stair):
    rows = []
    for p in paths:
        with np.load(p,allow_pickle=False) as archive:
            obs=sensor_frame(archive["obs"],defaults)
            act=archive["action"].astype(np.float32).clip(-6.,6.)
        seed=int(p.stem.split("-as2w")[1].split("-")[0] if stair else p.stem.split("-")[1])
        if not (len(obs)==len(act) and obs.shape[1]==41 and act.shape[1]==16):
            raise ValueError(p)
        rows.append((seed,obs,act))
    if len({seed for seed,_,_ in rows})<2:
        raise RuntimeError("Need distinct demonstrations for training and validation")
    holdout=max(seed for seed,_,_ in rows)
    def tensors(group):
        return [torch.as_tensor(np.stack(arr),device=device) for arr in zip(*group)]
    return (tensors([(x,y) for seed,x,y in rows if seed!=holdout]),
            tensors([(x,y) for seed,x,y in rows if seed==holdout]),holdout)


def same_side(action,defaults):
    pos=defaults[:12]+.35*action[..., :12]
    h=pos[..., :2]
    t=pos[..., 4:6]
    c=pos[..., 8:10]
    z=-.212*(torch.cos(t)+torch.cos(t+c))
    sign=torch.as_tensor((1.,-1.),device=action.device)
    y=sign*.060+sign*.1054*torch.cos(h)-z*torch.sin(h)
    return F.relu(.055-y*sign).square().mean()


def model_loss(model, x, target, default, noisy=True):
    if noisy:
        noise=torch.as_tensor((0.,)*3+(.003,)*3+(.001,)*3+(.001,)*16+(.02,)*16,device=x.device)
        x=x+noise*torch.randn_like(x)
    action,_=model(x)
    return (F.smooth_l1_loss(action,target,beta=.10)
            +5.*same_side(action,default)
            +.05*F.smooth_l1_loss(action[:,1:]-action[:,:-1],target[:,1:]-target[:,:-1],beta=.07))


if __name__=="__main__":
    contract,flat_dir,stair_dir,out=map(Path,sys.argv[1:5])
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(154039)
    torch.set_num_threads(1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defaults=np.asarray(json.loads(contract.read_text())["default_joint_pos"],np.float32)
    flat,flat_test,flat_seed=prepare(sorted(flat_dir.glob("flat-*.npz")),defaults,device,False)
    stairs,stairs_test,stair_seed=prepare(sorted(stair_dir.glob("h*-*.npz")),defaults,device,True)
    observations=torch.cat((flat[0].reshape(-1,41),stairs[0].reshape(-1,41)),dim=0)
    actor=BlindStudent(observations.mean(0),observations.std(0).clamp_min(.045)).to(device)
    optim=torch.optim.AdamW(actor.parameters(),lr=.00065,weight_decay=.009)
    q=torch.as_tensor(defaults,device=device)
    print("training",len(flat[0]),len(stairs[0]),"heldout",len(flat_test[0]),len(stairs_test[0]),
          "heldout seeds",flat_seed,stair_seed,"device",device,flush=True)
    for update in range(1,1201):
        i=torch.randint(len(flat[0]),(8,),device=device)
        j=torch.randint(len(stairs[0]),(3,),device=device)
        loss=model_loss(actor,flat[0][i],flat[1][i],q)+model_loss(actor,stairs[0][j],stairs[1][j],q)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optim.step()
        if update%300==0:
            with torch.inference_mode():
                valid=[float(model_loss(actor,*pair,q,False)) for pair in (flat_test,stairs_test)]
            print("update",update,"heldout flat and stairs loss",valid,flush=True)
            torch.save(dict(actor=actor.state_dict(),default=defaults.tolist(),heldout=valid,
                            input_size=41,memory_size=96),out/f"student-{update}.pt")
