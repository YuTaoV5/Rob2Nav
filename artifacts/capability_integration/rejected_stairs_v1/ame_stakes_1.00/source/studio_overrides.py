"""Strict JSON overrides shared with Studio; no executable user expressions."""
import hashlib
import json
import math
from pathlib import Path

RANGES={'learning_rate':(1e-6,.01),'actor_learning_rate':(1e-7,.001),'encoder_learning_rate':(1e-7,.001),
        'entropy_coef':(0.,.1),'gamma':(.8,.9999),'lam':(.8,1.),'clip_param':(.00001,.5),
        'num_learning_epochs':(1,20),'num_mini_batches':(1,64),'max_grad_norm':(.00001,10.)}
REWARDS={'forward_progress','goal_motion','goal_position','route_bonus','termination','lane_center',
         'anchor_cost','supported_contact','heading_cost','descending_foot_risk'}


def load(path):
    if not path:return {'ppo':{},'rewards':{}},None
    raw=Path(path).read_bytes();d=json.loads(raw.decode('utf-8-sig'))
    if not isinstance(d,dict) or set(d)-{'ppo','rewards'}:raise ValueError('Unknown Studio configuration section')
    for group in ('ppo','rewards'):
        if not isinstance(d.get(group,{}),dict):raise ValueError('Studio section must be an object')
    for key,value in d.get('ppo',{}).items():
        if key not in RANGES:raise ValueError('Unknown PPO override '+key)
        lo,hi=RANGES[key]
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or not lo<=value<=hi:
            raise ValueError('Invalid PPO override '+key)
        if key in ('num_learning_epochs','num_mini_batches') and (not isinstance(value,int) or isinstance(value,bool)):
            raise ValueError('PPO count must be an integer')
    for key,value in d.get('rewards',{}).items():
        if key not in REWARDS or not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or abs(value)>1000:
            raise ValueError('Invalid reward override '+key)
    d.setdefault('ppo',{});d.setdefault('rewards',{})
    canonical=json.dumps(d,sort_keys=True,separators=(',',':')).encode()
    return d,hashlib.sha256(canonical).hexdigest()


def apply_rewards(cfg,overrides):
    for name,weight in overrides['rewards'].items():
        term=getattr(cfg.rewards,name,None)
        if term is None or not hasattr(term,'weight'):raise ValueError('Reward not available in this mode: '+name)
        term.weight=float(weight)
