"""Small bounded proprioceptive actuator trims for the routed stair actor."""
import numpy as np
import torch
from torch import nn

WIDTH=58
OUT=8
HIDDEN=(96,64)
AXLE=np.array((0,0,1,1))
SIDE=np.array((-1.,1.,-1.,1.))


def network():
    return nn.Sequential(nn.Linear(WIDTH,HIDDEN[0]),nn.ELU(),nn.Linear(HIDDEN[0],HIDDEN[1]),nn.ELU(),
                         nn.Linear(HIDDEN[1],OUT))


def features(obs,action,age):
    x=np.r_[obs[:9],obs[10:26],obs[26:42]*np.r_[np.full(12,.15),np.full(4,.05)],
            action,np.clip(age/175.,0.,1.)]
    return x.astype(np.float32)


def apply(action,trim,inverse):
    trim=np.clip(trim,-1.,1.)
    xyz=np.zeros((4,3))
    xyz[:,2]=.057*trim[:2][AXLE]+.027*SIDE*np.array((0.,0.,1.,1.))*trim[4]
    xyz[:,0]=.048*trim[2:4][AXLE]
    q=np.einsum("lij,lj->il",inverse,xyz).reshape(12)/.35
    wheel=.32*trim[5:7][AXLE]+.18*SIDE*trim[7]
    return (action+np.r_[q.clip(-.85,.85),wheel]).clip(-6.,6.)


class LearnedTrim:
    def __init__(self,path,inverse,gain=1.):
        ck=torch.load(path,map_location="cpu",weights_only=True)
        self.layers=network().eval()
        self.layers.load_state_dict(ck["model"])
        self.inverse=inverse
        self.gain=gain

    @torch.inference_mode()
    def __call__(self,obs,action,age):
        trim=self.layers(torch.from_numpy(features(obs,action,age))).numpy()
        return apply(action,self.gain*trim,self.inverse)
