"""One 50 Hz AS2-W tick, including the stair GRU and explicit control memory."""
import math
import torch
from torch import nn

from r152_onnx_model import OBS_SIZE, STATE_SIZE as FLAT_STATE, R152Step, sine, cosine
from train_student import BlindStudent
import fit_branch_residual
import flat_trim_policy
import stair_trim

PARTS=(FLAT_STATE,4,96,6*41,3)
STATE_SIZE=sum(PARTS)


def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)


class R177Step(nn.Module):
    def __init__(self,weights,contract,geometry):
        super().__init__()
        self.flat=R152Step(load(weights/"flat-policy.pt"),contract,geometry)
        checkpoint=load(weights/"stairs-gru.pt")["actor"]
        self.stairs=BlindStudent(checkpoint["mean"],checkpoint["scale"])
        self.stairs.load_state_dict(checkpoint)
        ck=load(weights/"stairs-residual.pt")
        self.branch=fit_branch_residual.network()
        self.branch.load_state_dict(ck["model"])
        self.register_buffer("branch_mean",ck["mean"])
        self.register_buffer("branch_scale",ck["scale"])
        self.ppo=stair_trim.network()
        self.ppo.load_state_dict(load(weights/"stairs-ppo.pt")["model"])
        self.flat_trim=flat_trim_policy.network()
        self.flat_trim.load_state_dict(load(weights/"flat-ppo.pt")["model"])
        self.register_buffer("flat_scale",torch.from_numpy(flat_trim_policy.SCALE))
        self.register_buffer("vel_scale",torch.tensor([.15]*12+[.05]*4,dtype=torch.float64))
        self.register_buffer("hip_x",torch.tensor([.2119,.2119,-.2119,-.2119],dtype=torch.float64))
        self.register_buffer("side",torch.tensor([-1.,1.,-1.,1.],dtype=torch.float64))
        self.register_buffer("neutral",self.displacement(torch.full((4,),-.018,dtype=torch.float64)))

    def displacement(self,z,x=None):
        zero=torch.zeros_like(z)
        delta=torch.stack((zero if x is None else x,zero,z),dim=1)
        return torch.einsum("lij,lj->il",self.flat.inverse,delta).reshape(12)/.35

    def forward(self,obs,state):
        flat_state,gate,hidden,history,summit=state.reshape(STATE_SIZE).split(PARTS)
        command,angular,gravity=obs[0,:3].double(),obs[0,3:6].double(),obs[0,6:9].double()
        q,dq=obs[0,9:25].double(),obs[0,25:41].double()
        q=torch.cat((q[:12],torch.remainder(q[12:]+2.*math.pi,4.*math.pi)-2.*math.pi))
        pos=q-self.flat.defaults
        sensor=torch.cat((command,angular,gravity,q,dq)).float()
        flat,next_flat=self.flat(sensor[None],flat_state[None])
        flat=flat[0].double()
        clock=flat_state[0].double()*(.02*(2.*math.pi/.62))
        cmd_offset=command-self.flat.command_offset
        trim_in=torch.cat((cmd_offset,angular,gravity,pos,dq*self.vel_scale,
                           flat,torch.stack((sine(clock),cosine(clock))))).float()
        strength=((command[1:3].abs().sum()-.025)/.10).clamp(0.,1.)
        flat_trim=(flat+strength*self.flat_scale*self.flat_trim(trim_in).double().clamp(-1.,1.)).clamp(-6.,6.)
        flat=torch.where(strength>0.,flat_trim,flat)

        forward_age,repeats,pitch_repeats,age_state=gate.double().unbind(0)
        straight=(command[0]>=.13)&(command[1:3].abs().sum()<.04)
        front=.086*dq[12:14].min()
        rear=.086*dq[14:16].mean()
        ready=straight&(forward_age>=25.)&(age_state==0.)
        stall=ready&(command[0]-front>=.14)&(rear-front>=.042)&(rear>=.8*command[0])
        pitch=ready&(-gravity[0]>=.10)&(-angular[1]>=.50)&(rear>=.45*command[0])
        repeats=torch.where(stall,repeats+1.,0.)
        pitch_repeats=torch.where(pitch,pitch_repeats+1.,0.)
        active=straight&((age_state>0.)|(repeats>=3.)|(pitch_repeats>=2.))
        age=age_state
        blend=((age+1.)/8.).clamp(0.,1.)
        blend=blend.square()*(3.-2.*blend)
        action,next_hidden=self.stairs(sensor.reshape(1,1,OBS_SIZE),hidden.reshape(1,1,96))
        action=flat+blend*(action[0,0].double()-flat)

        history=history.reshape(6,OBS_SIZE)
        history=torch.where(age_state==0.,sensor.expand_as(history),torch.cat((history[1:],sensor[None])))
        branch_in=torch.cat((history[5],history[3],history[0],action.float(),(age/150.).clamp_max(1.).float().reshape(1)))
        trim=self.branch(((branch_in-self.branch_mean)/self.branch_scale).clamp(-8.,8.))
        action=action+.8*((trim[16].sigmoid().double()-.24)/.76).clamp_min(0.)*trim[:16].double().clamp(-1.,1.)

        distance,level_ticks,hold=summit.double().unbind(0)
        speeds=torch.sort(dq[12:]).values
        distance=distance+.02*(.086*speeds[1:3].mean()).clamp(0.,.85)
        hip,thigh,calf=q[:12].reshape(3,4).unbind(0)
        long=-.212*(cosine(thigh)+cosine(thigh+calf))
        points=torch.stack((self.hip_x-.212*(sine(thigh)+sine(thigh+calf)),
                            -.06*self.side-.1054*self.side*cosine(hip)-long*sine(hip),
                            -.1054*self.side*sine(hip)+long*cosine(hip)),dim=1)
        heights=points@(-gravity)
        level=(distance>=1.95)&(heights.max()-heights.min()<.034)&(-gravity[2]>.985)&(dq[12:].abs().mean()<7.)
        level_ticks=torch.where(level,level_ticks+1.,0.)
        hold=torch.where((level_ticks>=6.)|(hold>0.),hold+1.,0.)
        fade=(hold/10.).clamp_max(1.)
        fade=fade.square()*(3.-2.*fade)
        neutral=torch.cat((self.neutral,-.16*angular[2]*self.side))
        held=action+fade*(neutral-action)

        ppo_in=torch.cat((cmd_offset,angular,gravity,pos,dq*self.vel_scale,action,(age/175.).clamp_max(1.).reshape(1))).float()
        t=self.ppo(ppo_in).double().clamp(-1.,1.)
        z=.057*t[:2].repeat_interleave(2)+.027*self.side*torch.tensor([0.,0.,1.,1.],device=t.device)*t[4]
        x=.048*t[2:4].repeat_interleave(2)
        wheel=.32*t[5:7].repeat_interleave(2)+.18*self.side*t[7]
        trimmed=(action+torch.cat((self.displacement(z,x).clamp(-.85,.85),wheel))).clamp(-6.,6.)
        final=torch.where(active,torch.where(hold>0.,held,trimmed),flat).float()
        next_flat=torch.cat((next_flat[0,:1],final,next_flat[0,17:]))
        gate=torch.stack((torch.where(straight,forward_age+1.,0.),repeats,pitch_repeats,
                          torch.where(active,age_state+1.,0.)))
        next_summit=torch.stack((distance,level_ticks,hold))
        memory=torch.cat((next_flat,gate.float(),torch.where(active,next_hidden.reshape(96),0.),
                          torch.where(active,history.reshape(-1),0.),torch.where(active,next_summit,0.).float()))
        return final.reshape(1,16),memory.reshape(1,STATE_SIZE)
