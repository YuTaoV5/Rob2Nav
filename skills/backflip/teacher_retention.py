"""Optional alternating actor-only first-flip retention update (not a runtime controller).

After a normal PPO update, take one stateless SGD step on pinned teacher samples.
No critic, action-noise, PPO optimizer state, simulator state or policy history is changed.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from skills.fixstand import backflip_entry_weight


def load_reference(directory, manifest_sha256):
    directory=Path(directory)
    payload=(directory/'manifest.json').read_bytes()
    if len(manifest_sha256)!=64 or hashlib.sha256(payload).hexdigest()!=manifest_sha256:
        raise ValueError('Teacher manifest hash mismatch')
    meta=json.loads(payload)
    if meta.get('version')!='first_flip_retention_data_v1' or meta.get('action_space')!='raw actor before entry mapping and clipping':
        raise ValueError('Unknown teacher observation/action semantics')
    data_path=directory/'samples.npz'
    if hashlib.sha256(data_path.read_bytes()).hexdigest()!=meta.get('samples_sha256'):
        raise ValueError('Teacher data hash mismatch')
    with np.load(data_path,allow_pickle=False) as saved:
        data={k:saved[k].copy() for k in ('observations','raw_actions','policy_time','seed')}
    n=meta.get('samples')
    if type(n) is not int or n<1 or data['observations'].shape!=(n,60) or data['raw_actions'].shape!=(n,12) or data['policy_time'].shape!=(n,) or data['seed'].shape!=(n,):
        raise ValueError('Teacher shape mismatch')
    if not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite teacher data')
    if (data['policy_time']<0).any() or (data['policy_time']>2.1).any():raise ValueError('Teacher includes unsupported/terminal time')
    absorption=meta.get('absorption_seconds')
    if type(absorption) not in (int,float) or not math.isfinite(absorption) or not 0<=absorption<=.06:
        raise ValueError('Invalid teacher absorption window')
    if not meta.get('sources'):raise ValueError('Missing teacher sources')
    if len(set(s['seed'] for s in meta['sources']))!=len(meta['sources']):raise ValueError('Duplicate teacher seeds')
    if set(data['seed'].tolist())!={s['seed'] for s in meta['sources']}:raise ValueError('Teacher source mismatch')
    for s in meta['sources']:
        contact=s.get('first_contact_trace_s');cutoff=s.get('cutoff_s')
        if type(contact) not in (int,float) or type(cutoff) not in (int,float) or not math.isfinite(contact) or not math.isfinite(cutoff) or contact<1 or abs(cutoff-contact-absorption)>1e-9:
            raise ValueError('Teacher contact cutoff mismatch')
        t=data['policy_time'][data['seed']==s['seed']]
        if len(t)!=s['retained'] or not np.allclose(t,np.arange(len(t))*.02,rtol=0,atol=1e-9) or (t>s['cutoff_s']+1e-9).any():
            raise ValueError('Teacher time coverage mismatch')
    return data,meta


def mapped_targets(raw,time,hold_scale,action_scale,action_clip):
    weight=backflip_entry_weight(time).to(device=raw.device,dtype=raw.dtype).unsqueeze(-1)
    return torch.clamp((1-weight)*raw+weight*hold_scale*torch.tanh(raw),-action_clip,action_clip)*action_scale


class RetentionUpdater:
    def __init__(self,actor,data,*,weight,learning_rate=1e-5,batch_size=128,max_grad_norm=1.,seed=0,
                 hold_scale=.15,action_scale=.5,action_clip=8.):
        values=(weight,learning_rate,max_grad_norm,hold_scale,action_scale,action_clip)
        if not all(math.isfinite(v) for v in values) or not 0<weight<=100 or not 0<learning_rate<=.001 or not 0<max_grad_norm<=10 or not 0<=hold_scale<=.4 or action_scale<=0 or action_clip<=0:
            raise ValueError('Invalid retention settings')
        if type(batch_size) is not int or not 1<=batch_size<=4096:raise ValueError('Invalid retention batch size')
        self.actor=actor;self.params=tuple(actor.parameters())
        if not self.params or not all(p.requires_grad for p in self.params):raise ValueError('Trainable actor required')
        self.device=self.params[0].device;self.weight=weight;self.lr=learning_rate;self.batch_size=batch_size;self.max_norm=max_grad_norm
        self.hold=hold_scale;self.scale=action_scale;self.clip=action_clip
        self.obs=torch.as_tensor(data['observations'],dtype=torch.float32,device=self.device)
        self.actions=torch.as_tensor(data['raw_actions'],dtype=torch.float32,device=self.device)
        self.times=torch.as_tensor(data['policy_time'],dtype=torch.float32,device=self.device)
        self.generator=torch.Generator(device='cpu').manual_seed(seed)
        self.steps=0

    def step(self):
        ids=torch.randint(len(self.obs),(self.batch_size,),generator=self.generator).to(self.device)
        with torch.enable_grad():
            predicted=self.actor(self.obs[ids])
            if predicted.shape!=self.actions[ids].shape:raise ValueError('Actor output mismatch')
            target=mapped_targets(self.actions[ids],self.times[ids],self.hold,self.scale,self.clip)
            actual=mapped_targets(predicted,self.times[ids],self.hold,self.scale,self.clip)
            mse=(actual-target).square().mean()
            grads=torch.autograd.grad(mse*self.weight,self.params)
            norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            if not torch.isfinite(mse) or not torch.isfinite(norm):raise ValueError('Nonfinite retention gradient')
            factor=(self.max_norm/(norm+1e-12)).clamp(max=1.)
        with torch.no_grad():
            for p,g in zip(self.params,grads):p.add_(g*factor,alpha=-self.lr)
        self.steps+=1
        return {'teacher_retention_mse':float(mse.detach()),'teacher_retention_grad_norm':float(norm.detach())}


def install(runner,updater):
    if getattr(runner,'_teacher_retention_installed',False):raise ValueError('Retention already installed')
    if runner.alg.policy.actor is not updater.actor:raise ValueError('Wrong actor reference')
    original=runner.alg.update
    def update():
        losses=original()
        if not isinstance(losses,dict):raise ValueError('Unsupported PPO loss result')
        return {**losses,**updater.step()}
    runner.alg.update=update;runner._teacher_retention_installed=True
