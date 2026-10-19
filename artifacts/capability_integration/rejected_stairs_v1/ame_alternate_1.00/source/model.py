"""Trainable reconstruction of the shipped 137k MoE plus terrain fusion.

The deployed checkpoint, rather than the newer upstream architecture, defines
the exact base network. Validate with check.py before using a different weight.
Architecture/weights: Wu Tianyang, RoboGauge/go2_rl_gym; see THIRD_PARTY_NOTICES.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

HISTORY=5
CELLS=33*21
ACTOR_DIM=45*HISTORY+3*CELLS


class DynamicsMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts_backbone=nn.Sequential(nn.Linear(225,512),nn.ELU(),nn.Linear(512,256),nn.ELU())
        self.experts_hidden=nn.Sequential(nn.Linear(256,2048),nn.ELU())
        self.experts_out=nn.Conv1d(2048,256,1,groups=8)
        self.gating_network=nn.Sequential(nn.Linear(225,512),nn.ELU(),nn.Linear(512,256),
                                          nn.ELU(),nn.Linear(256,8),nn.Softmax(dim=-1))

    def forward(self,history):
        weights=self.gating_network(history)
        features=self.experts_hidden(self.experts_backbone(history))
        experts=self.experts_out(features.unsqueeze(-1)).reshape(-1,8,32)
        return F.normalize((weights.unsqueeze(-1)*experts).sum(1),p=2.,dim=-1)


class BaseMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_moe_encoder=DynamicsMoE()
        self.actor=nn.Sequential(nn.Linear(77,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
                                 nn.Linear(256,128),nn.ELU(),nn.Linear(128,12))

    def forward(self,history):
        latent=self.student_moe_encoder(history)
        return self.actor(torch.cat((latent,history[:,-45:]),-1))


def load_base(path):
    original=torch.jit.load(str(path),map_location='cpu')
    if list(original.history.shape)!=[1,5,45]:raise ValueError('Unsupported MoE history contract')
    if list(original.normalizer.state_dict()):raise ValueError('Expected identity observation normalizer')
    base=BaseMoE()
    base.load_state_dict(original.state_dict(),strict=True)
    return base


class TerrainAttention(nn.Module):
    def __init__(self,spatial_prior=False):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GroupNorm(4,32),nn.ELU(),
                               nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.ELU())
        y,x=torch.meshgrid(torch.linspace(-.5,.5,21),torch.linspace(-.55,1.05,33),indexing='ij')
        self.register_buffer('xy',torch.stack((x.flatten(),y.flatten()),-1))
        self.spatial_prior=spatial_prior
        # FL, FR, RL, RR nominal stance neighborhoods, not predicted feet.
        centres=torch.tensor([[.25,.15],[.25,-.15],[-.2,.15],[-.2,-.15]])
        bias=-((self.xy[None]-centres[:,None]).square().sum(-1))/(2*.22**2)
        self.register_buffer('spatial_bias',torch.cat((bias,torch.zeros(4,1)),1),persistent=False)
        self.key=nn.Linear(34,64);self.value=nn.Linear(34,64)
        self.query=nn.Sequential(nn.Linear(77,64),nn.ELU())
        self.null=nn.Parameter(torch.zeros(1,1,34))

    def forward(self,map_values,query_input):
        maps=map_values.reshape(-1,3,21,33)
        valid=maps[:,1].flatten(1)>.5
        # Missing heights are zero plus an explicit mask, never inferred floor.
        masked=torch.stack((maps[:,0]*maps[:,1],maps[:,1],maps[:,2]*maps[:,1]),1)
        features=self.cnn(masked).flatten(2).transpose(1,2)
        tokens=torch.cat((features,self.xy.unsqueeze(0).expand(features.shape[0],-1,-1)),-1)
        tokens=torch.cat((tokens,self.null.expand(features.shape[0],-1,-1)),1)
        keys=self.key(tokens).reshape(-1,694,4,16).transpose(1,2)
        values=self.value(tokens).reshape(-1,694,4,16).transpose(1,2)
        query=self.query(query_input).reshape(-1,4,1,16)
        known=torch.cat((valid,torch.ones_like(valid[:,:1])),1)
        logits=(query*keys).sum(-1)/4.
        if self.spatial_prior:logits=logits+self.spatial_bias[None]
        weights=logits.masked_fill(~known[:,None,:],-1e4).softmax(-1)
        return (weights.unsqueeze(-1)*values).sum(2).reshape(-1,64)


class PerceptiveActor(nn.Module):
    def __init__(self,checkpoint,goal_conditioned=False,spatial_prior=False):
        super().__init__()
        self.base=load_base(checkpoint)
        self.anchor=copy.deepcopy(self.base)
        self.terrain=TerrainAttention(spatial_prior)
        self.fusion=nn.Sequential(nn.Linear(141,128),nn.ELU(),nn.Linear(128,12))
        self.goal_conditioned=goal_conditioned
        self.goal_projection=nn.Linear(4,77) if goal_conditioned else nn.Identity()
        if goal_conditioned:
            nn.init.zeros_(self.goal_projection.weight);nn.init.zeros_(self.goal_projection.bias)
        nn.init.zeros_(self.fusion[-1].weight);nn.init.zeros_(self.fusion[-1].bias)
        self.residual_bound=.4  # normalized action: .4 * .25 = .10 rad initially
        self.set_phase('fusion')

    def set_phase(self,phase):
        if phase not in ('value','fusion','actor','joint'):raise ValueError(phase)
        for p in self.parameters():p.requires_grad_(phase!='value')
        for p in self.anchor.parameters():p.requires_grad_(False)
        for p in self.base.parameters():p.requires_grad_(phase in ('actor','joint'))
        if phase=='actor':
            for p in self.base.student_moe_encoder.parameters():p.requires_grad_(False)
        self.residual_bound={'value':.4,'fusion':.4,'actor':.8,'joint':1.2}[phase]

    def forward(self,obs):
        history=obs[:,:225];current=history[:,-45:]
        latent=self.base.student_moe_encoder(history)
        proprio=torch.cat((latent,current),-1)
        base=self.base.actor(proprio)
        query=proprio+self.goal_projection(obs[:,2304:2308]) if self.goal_conditioned else proprio
        terrain=self.terrain(obs[:,225:2304],query)
        return base+self.residual_bound*torch.tanh(self.fusion(torch.cat((query,terrain),-1)))


def register_policy():
    from rsl_rl.modules import ActorCritic
    import rsl_rl.runners.on_policy_runner as runner
    class MoEPerceptiveActorCritic(ActorCritic):
        def __init__(self,*args,moe_checkpoint,goal_conditioned=False,spatial_prior=False,**kwargs):
            super().__init__(*args,**kwargs)
            self.actor=PerceptiveActor(moe_checkpoint,goal_conditioned,spatial_prior)
    runner.MoEPerceptiveActorCritic=MoEPerceptiveActorCritic
