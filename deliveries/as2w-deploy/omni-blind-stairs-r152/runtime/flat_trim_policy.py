"""Bounded residual for command-conditioned AS2-W flat locomotion."""
import numpy as np
import torch
from torch import nn

WIDTH=59
OUT=16
HIDDEN=(96,64)
SCALE=np.repeat((.22,.20,.22,.28),4)


def network():
    return nn.Sequential(nn.Linear(WIDTH,HIDDEN[0]),nn.ELU(),nn.Linear(HIDDEN[0],HIDDEN[1]),
                         nn.ELU(),nn.Linear(HIDDEN[1],OUT))


def features(obs,nominal):
    return np.r_[obs[:9],obs[10:26],obs[26:42]*np.repeat((.15,.15,.15,.05),4),
                 nominal,obs[58:60]].astype(np.float32)


def apply(obs,nominal,residual):
    strength=np.clip((abs(obs[1])+abs(obs[2])-.025)/.10,0.,1.)
    return np.clip(nominal+strength*SCALE*np.clip(residual,-1.,1.),-6.,6.)
