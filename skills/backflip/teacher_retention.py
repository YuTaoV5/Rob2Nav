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


from .teacher_data import load_reference


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


def configure_runner(runner, log_dir, *, directory, manifest_sha256, seed, **settings):
    """New run sampler resets explicitly; does not promise exact resume."""
    import yaml
    from rob2nav.export_actor import ConfigLoader
    from .landing_contract import validate_landing_contract
    log_dir=Path(log_dir)
    env=yaml.load((log_dir/'params/env.yaml').read_text(),Loader=ConfigLoader)
    contract=json.loads((log_dir/'fixstand.json').read_text())
    validate_landing_contract(env,contract)
    if env['control']['action_scale']!=.5 or env['control']['action_clip']!=8. or contract.get('entry_hold_scale')!=settings['hold_scale']:
        raise ValueError('Teacher mapping differs from actual training contract')
    data, meta=load_reference(directory,manifest_sha256)
    updater=RetentionUpdater(runner.alg.policy.actor,data,seed=seed,**settings)
    install(runner,updater)
    log_dir=Path(log_dir)
    record=dict(version='actor_retention_sgd_v1',algorithm='one actor-only SGD step after each PPO update',
        manifest_sha256=manifest_sha256,samples_sha256=meta['samples_sha256'],
        teacher_policy_sha256=meta['teacher_policy_sha256'],samples=meta['samples'],
        sampler_seed=seed,sampler_resume='reset on each new training invocation; not bitwise resume',
        max_grad_norm=1.,action_scale=.5,action_clip=8.,**settings)
    (log_dir/'teacher_retention.json').write_text(json.dumps(record,indent=2)+'\n')
    contract['teacher_retention_sha256']=hashlib.sha256((log_dir/'teacher_retention.json').read_bytes()).hexdigest()
    (log_dir/'fixstand.json').write_text(json.dumps(contract,indent=2)+'\n')
    (log_dir/'teacher_manifest.json').write_bytes((Path(directory)/'manifest.json').read_bytes())
    print('[Rob2Nav] Actor retention configured: '+json.dumps(record),flush=True)
    return updater
