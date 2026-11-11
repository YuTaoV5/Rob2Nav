"""Task-specific SuperMimic recovery: raw proprioception, shared real/imagined reward."""
from pathlib import Path
from dataclasses import dataclass
import math,hashlib,xml.etree.ElementTree as ET
import torch
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.managers.reward_manager import RewardTermCfg
import rob2nav_velocity as v
from rob2nav_reset_targets import install
from go2_supermimic.supermimic import state as st
TASK='Go2-SuperMimic-Recovery-V1'
class RecoveryCommand(v.VelocityCommand):
 def _resample_command(self,env_ids):
  self.gen.chase_vel[env_ids]=0;self.gen.has_chase[env_ids]=False;self.gen.has_dodge[env_ids]=False
@dataclass(kw_only=True)
class RecoveryCommandCfg(v.VelocityCommandCfg):
 def build(self,env):return RecoveryCommand(self,env)
def reward(s,action,previous,default_q,scale,limits):
 upright=((1-s[:,st.GRAV][:,2])/2).clamp(0,1)
 height=torch.exp(-((s[:,st.Z].squeeze(-1)-.29)/.12).square())
 q=s[:,st.Q]+default_q
 goal=torch.zeros_like(q);goal[:,1::3]=.9;goal[:,2::3]=-1.8
 posture=torch.exp(-((q-goal).square().mean(-1))/.3)
 rate=((action-previous)*scale/.02/limits).abs()
 return 6*upright+2*upright*height+6*upright*height*posture-.3*(s[:,st.QD]/limits).square().mean(-1)-2*(rate-.8).clamp(min=0).square().amax(-1)-.03*s[:,st.W_B].square().sum(-1)-upright*s[:,st.V_YAW].square().sum(-1)
def real_reward(env):
 c=st.robot_consts(env)
 return reward(st.robot_state(env),env.action_manager.action,env.action_manager.prev_action,c.default_q,c.action_scale,velocity_limits(env))
def velocity_limits(env):
 import re
 robot=env.scene['robot'];names=env.action_manager.get_term('joint_pos')._target_names
 return torch.tensor([next(a.velocity_limit for a in robot.cfg.articulation.actuators if any(re.fullmatch(p,n) for p in a.target_names_expr)) for n in names],device=env.device)
def cfg(play=False):
 c=install(v.flat_cfg(play));c.episode_length_s=10.
 c.commands['target']=RecoveryCommandCfg(resampling_time_range=(10000.,10000.),debug_vis=False,mode_probs=(1.,0.,0.),kind_curriculum=False)
 c.terminations={'time_out':c.terminations['time_out']}
 c.events['reset_base'].params['pose_range']={'x':(-.1,.1),'y':(-.1,.1),'z':(0.,0.),'roll':(-math.pi,math.pi),'pitch':(-math.pi/2,math.pi/2),'yaw':(-math.pi,math.pi)}
 robot=c.scene.entities['robot'];robot.init_state.pos=(0.,0.,.35);robot.init_state.joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':1.5,'.*_calf_joint':-2.4}
 for a in robot.articulation.actuators:a.stiffness=30.;a.damping=.8;a.delay_min_lag=0;a.delay_max_lag=0
 c.rewards={'recovery':RewardTermCfg(func=real_reward,weight=1.)}
 return c
class RecoveryImagine(v.VelocityImagination):
 def fall_prob(self,s):return torch.zeros(s.shape[0],device=s.device)
 def _rewards(self,s0,s1,pose1,gst,a,a_prev,k):
  return reward(s1,a,a_prev,k['default_q'],k['action_scale'],self.recovery_limits)*self.cfg.dt,{}
register_mjlab_task(task_id=TASK,env_cfg=cfg(),play_env_cfg=cfg(True),rl_cfg=RslRlOnPolicyRunnerCfg())
class FRTeacher:
 def __init__(self,env,path):
  self.env=env;self.path=Path(path);self.source=self.path.parents[3];d=torch.load(path,map_location=env.device,weights_only=True)
  self.net=torch.nn.Sequential(torch.nn.Linear(45,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12)).to(env.device)
  self.net.load_state_dict({k.removeprefix('actor.'):x for k,x in d['model_state_dict'].items() if k.startswith('actor.')});self.net.eval()
  for p in self.net.parameters():p.requires_grad_(False)
  joints=[j for j in ET.parse(self.source/'resources/robots/go2/urdf/go2.urdf').getroot().findall('joint') if j.get('type') in ('revolute','continuous')];self.names=[j.get('name') for j in joints]
  current=list(env.action_manager.get_term('joint_pos')._target_names);self.to_native=torch.tensor([current.index(n) for n in self.names],device=env.device);self.to_current=torch.tensor([self.names.index(n) for n in current],device=env.device)
  self.default=torch.tensor([0 if '_hip_' in n else 1.5 if '_thigh_' in n else -2.4 for n in self.names],device=env.device)
  self.previous=torch.zeros((env.num_envs,12),device=env.device);self.consts=st.robot_consts(env);self.frames=0
 def act(self,command):
  s=st.robot_state(self.env);q=s[:,st.Q]+self.consts.default_q;obs=torch.cat([s[:,st.W_B]*.25,s[:,st.GRAV],torch.zeros_like(command),q[:,self.to_native]-self.default,s[:,st.QD][:,self.to_native]*.05,self.previous],-1).clamp(-100,100)
  with torch.no_grad():raw=self.net(obs).clamp(-10,10)
  self.previous.copy_(raw);target=(self.default+.25*raw)[:,self.to_current];self.frames+=self.env.num_envs
  return (target-self.consts.default_q)/self.consts.action_scale
 def reset(self,done):self.previous[done]=0
