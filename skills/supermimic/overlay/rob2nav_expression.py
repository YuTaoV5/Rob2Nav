"""Stage-1 expressive choreography: existing locomotion teacher + shared clock.

Style names describe intended choreography, not verified perceived emotions.
No body pose is overwritten; phase/style are desired-command inputs, not sensors.
"""
from dataclasses import dataclass
import math
import torch
import rob2nav_velocity as v
from rob2nav_reset_targets import install
from go2_supermimic.supermimic import state as st
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg
TASK='Go2-SuperMimic-Expression-Rhythm-V1'
STYLES=('joyful_steps','firm_march','slow_sway','playful_dance')

def velocity_command(t,style,period):
 phi=2*math.pi*t/period
 sn=phi.sin();cs=phi.cos()
 choices=torch.stack([torch.stack([.12+.08*sn,.10*sn,.25*cs],-1),
                      torch.stack([.22+.04*sn,torch.zeros_like(sn),.12*sn],-1),
                      torch.stack([torch.zeros_like(sn),.05*sn,.12*sn],-1),
                      torch.stack([.08*cs,.18*sn,.45*sn],-1)],1)
 return choices[torch.arange(len(t),device=t.device),style.long()]

def observation(command,t,style,period):
 phi=2*math.pi*t/period
 return torch.cat([command,torch.nn.functional.one_hot(style.long(),4).to(command),phi.sin()[:,None],phi.cos()[:,None],torch.zeros(len(t),19,device=t.device)],-1)

def advance(g,dt):
 # Reuse otherwise disabled generator fields, all already packed into replay:
 # wait=time, style=choreography index, speed_cap=period (seconds).
 out=dict(g);out['wait']=g['wait']+dt
 out['chase_vel']=velocity_command(out['wait'],out['style'],out['speed_cap'])
 return out

class ExpressionCommand(v.VelocityCommand):
 def _resample_command(self,ids):
  self.gen.wait[ids]=0
  self.gen.style[ids]=torch.randint(0,4,(len(ids),),device=self.device).float()
  self.gen.speed_cap[ids]=3.+torch.rand(len(ids),device=self.device)*2.
  self.gen.chase_vel[ids]=velocity_command(self.gen.wait[ids],self.gen.style[ids],self.gen.speed_cap[ids])
  self.gen.has_chase[ids]=False;self.gen.has_dodge[ids]=False
 def _advance(self,dt):
  g=advance(self.gen.state(),dt)
  self.gen.wait=g['wait'];self.gen.chase_vel=g['chase_vel'];self.hit.zero_()
  self.gen.has_chase[:]=False;self.gen.has_dodge[:]=False
 def _update_command(self,ids):
  self._obs=observation(self.gen.chase_vel,self.gen.wait,self.gen.style,self.gen.speed_cap)

@dataclass(kw_only=True)
class ExpressionCommandCfg(v.VelocityCommandCfg):
 def build(self,env):return ExpressionCommand(self,env)

def real_reward(env):
 c=st.robot_consts(env);g=env.command_manager.get_term('target').gen
 return v.reward(st.robot_state(env),g.chase_vel,env.action_manager.action,env.action_manager.prev_action,c.soft_limits,c.default_q)

def cfg(play=False):
 c=install(v.flat_cfg(play));c.episode_length_s=12.
 c.commands['target']=ExpressionCommandCfg(resampling_time_range=(1000.,1000.),debug_vis=False,mode_probs=(1.,0.,0.),kind_curriculum=False)
 c.rewards['velocity']=RewardTermCfg(func=real_reward,weight=1.)
 return c

class ExpressionImagination(v.VelocityImagination):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.critic_dim=st.STATE_DIM+9+st.CONTEXT_DIM
 def critic_input(self,s,pose,g,xi):
  return torch.cat([s,observation(g['chase_vel'],g['wait'],g['style'],g['speed_cap'])[:,:9],xi],-1)
 def _step(self,s,pose,a_prev,hist,g,xi,enc_bias,k,dt,actor,wm):
  x=self.critic_input(s,pose,g,xi);a=actor(hist)
  s1,dp=wm(s,a,a_prev,xi,self.relief(pose,g,self.cfg.sensor_grad));pose1=st.apply_pose_delta(pose,dp)
  r,_=self._rewards(s,s1,pose1,g,a,a_prev,k);d=self.fall_prob(s1)
  r=r+self.cfg.weights.terminated*d*self.cfg.dt
  g=advance(g,dt)
  frame=st.obs_frame(s1,a,observation(g['chase_vel'],g['wait'],g['style'],g['speed_cap']),enc_bias,noise=True)
  return s1,pose1,a,st.push_history(hist,frame,self.h,self.terms),g,r,d,x

register_mjlab_task(task_id=TASK,env_cfg=cfg(),play_env_cfg=cfg(True),rl_cfg=RslRlOnPolicyRunnerCfg())
