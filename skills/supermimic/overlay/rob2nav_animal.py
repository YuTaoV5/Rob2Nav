"""Blind crawl student task: scheduled reference commands and shared rewards.

The actor gets only proprioception, intended-action history and desired reference
joint pose/height/gravity/phase. Actual global root pose is teacher-only.
"""
from dataclasses import dataclass
import math
import torch
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from rob2nav_kine_teacher import diagnostic_cfg,reset_reference,profile,ROOT
from rob2nav_kine_reference import KineReference,rotate
from go2_supermimic.supermimic import state as st
import rob2nav_velocity as v
TASK='Go2-SuperMimic-Animal-Crawl-V1';CLIP='solo8_crawl_slow';CACHE={}

def parameters(device):
 key=str(device)
 if key not in CACHE:
  import json
  _,_,motion,reference_names=profile(CLIP,device)
  teacher=json.loads((ROOT/'mujoco_teacher_crawl_v1/report.json').read_text())
  assert teacher['functional_passed']>0
  names=teacher['teacher']['mjlab_action_names'];index=[reference_names.index(n) for n in names]
  limits=torch.tensor([20.07 if '_calf_' in n else 30.1 for n in names],device=device)
  CACHE[key]=(KineReference(motion),index,limits,names)
 return CACHE[key]

def desired(phase):
 ref,index,_,_=parameters(phase.device);heading=torch.zeros((len(phase),4),device=phase.device);heading[:,0]=1
 row=ref.at(phase.long(),heading);q=row[:,7:][:,index];inv=row[:,3:7].clone();inv[:,1:]*=-1
 gravity=torch.zeros((len(phase),3),device=phase.device);gravity[:,2]=-1;gravity=rotate(inv,gravity)
 angle=2*math.pi*phase/ref.length
 return q,row[:,2:3],gravity,torch.stack((angle.sin(),angle.cos()),-1)

def command_observation(phase):
 q,z,g,cyclic=desired(phase)
 return torch.cat((q,z,g,cyclic,torch.zeros((len(phase),10),device=phase.device)),-1)

def advance(g,dt):
 assert bool(((dt-1/60).abs()<1e-7).all()) if isinstance(dt,torch.Tensor) else abs(dt-1/60)<1e-7
 out=dict(g);out['wait']=g['wait']+1.;out['chase_vel']=torch.zeros_like(g['chase_vel']);return out

def reward(s,phase,action,previous,default_q,scale):
 goal,z,gravity,_=desired(phase);_,_,limits,_=parameters(s.device)
 upright=((1-s[:,st.GRAV][:,2])/2).clamp(0,1)
 pose=torch.exp(-8*(s[:,st.Q]+default_q-goal).square().mean(-1))
 attitude=torch.exp(-2*(s[:,st.GRAV]-gravity).square().sum(-1))
 height=torch.exp(-((s[:,st.Z]-z)/.1).square().squeeze(-1))
 rate=((action-previous)*scale*60/limits).abs()
 return 4*upright*pose+4*upright*attitude+2*upright*height-.3*(s[:,st.QD]/limits).square().mean(-1)-2*(rate-.8).clamp(min=0).square().amax(-1)-.05*s[:,st.W_B].square().sum(-1)

class AnimalCommand(v.VelocityCommand):
 def _resample_command(self,ids):
  self.gen.wait[ids]=0.;self.gen.chase_vel[ids]=0.;self.gen.has_chase[ids]=False;self.gen.has_dodge[ids]=False
 def _advance(self,dt):
  g=advance(self.gen.state(),dt);self.gen.wait=g['wait'];self.gen.chase_vel=g['chase_vel'];self.hit.zero_()
 def _update_command(self,ids):self._obs=command_observation(self.gen.wait)
@dataclass(kw_only=True)
class AnimalCommandCfg(v.VelocityCommandCfg):
 def build(self,env):return AnimalCommand(self,env)

def intended_action(env):
 return getattr(env,'_kine_intended_action',torch.zeros((env.num_envs,12),device=env.device))
def reset_animal(env,env_ids):
 reset_reference(env,env_ids,CLIP)
 for name in ('_kine_intended_action','_kine_previous_intended'):
  if hasattr(env,name):getattr(env,name)[env_ids]=0
 if hasattr(env,'_kine_reward_phase'):env._kine_reward_phase[env_ids]=0
def real_reward(env):
 c=st.robot_consts(env);phase=getattr(env,'_kine_reward_phase',env.command_manager.get_term('target').gen.wait)
 return reward(st.robot_state(env),phase,intended_action(env),getattr(env,'_kine_previous_intended',torch.zeros_like(intended_action(env))),c.default_q,c.action_scale)
def cfg(play=False):
 c=diagnostic_cfg(CLIP);c.episode_length_s=12.
 # Trainer and imagined observation construction both read this calibration.
 bias=v.flat_cfg(True).events['encoder_bias'];bias.params['bias_range']=(0.,0.);c.events['encoder_bias']=bias
 c.commands['target']=AnimalCommandCfg(resampling_time_range=(1000.,1000.),debug_vis=False,mode_probs=(1.,0.,0.),kind_curriculum=False)
 c.events['reset_reference']=EventTermCfg(mode='reset',func=reset_animal,params={})
 c.observations['actor'].terms['actions'].func=intended_action
 c.observations['actor'].terms['actions'].params={}
 c.rewards={'animal':RewardTermCfg(func=real_reward,weight=1.)}
 return c

class AnimalImagination(v.VelocityImagination):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.cfg.dt=1/60;self.critic_dim=st.STATE_DIM+18+st.CONTEXT_DIM
 def critic_input(self,s,pose,g,xi):return torch.cat((s,command_observation(g['wait'])[:,:18],xi),-1)
 def fall_prob(self,s):return torch.zeros(len(s),device=s.device)
 def _step(self,s,pose,a_prev,hist,g,xi,enc_bias,k,dt,actor,wm):
  x=self.critic_input(s,pose,g,xi);a=actor(hist);s1,dp=wm(s,a,a_prev,xi,self.relief(pose,g,self.cfg.sensor_grad));pose1=st.apply_pose_delta(pose,dp)
  g1=advance(g,dt);r=reward(s1,g1['wait'],a,a_prev,k['default_q'],k['action_scale'])*self.cfg.dt
  frame=st.obs_frame(s1,a,command_observation(g1['wait']),enc_bias,noise=True)
  return s1,pose1,a,st.push_history(hist,frame,self.h,self.terms),g1,r,self.fall_prob(s1),x

register_mjlab_task(task_id=TASK,env_cfg=cfg(),play_env_cfg=cfg(True),rl_cfg=RslRlOnPolicyRunnerCfg())
