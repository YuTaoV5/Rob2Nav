"""Shared real/imagined velocity reward primitives. Not yet a registered training task."""
import torch
from go2_supermimic.supermimic import state as st
CONTRACT={"version":"velocity_v1_draft","command_units":["m/s","m/s","rad/s"],
 "actor_target_layout":"vx,vy,wz followed by 25 fixed zeros; 5-frame history",
 "actor_sources":["IMU angular velocity","projected gravity","joint position","joint velocity","last action","velocity command"],
 "command_ranges":[[-0.5,1.0],[-0.3,0.3],[-0.8,0.8]],"standing_probability":0.2,
 "training_ready":False}
def command_observation(command):
 if command.ndim!=2 or command.shape[1]!=3:raise ValueError("Expected [B,3] velocity command")
 return torch.cat([command,torch.zeros((len(command),25),device=command.device,dtype=command.dtype)],dim=-1)
def reward_terms(s,command,action,previous_action,soft_limits,default_q):
 velocity=s[:,st.V_YAW][:,:2]
 yaw_rate=s[:,st.W_B][:,2]
 gravity=s[:,st.GRAV]
 q=s[:,st.Q]+default_q
 excess=((soft_limits[...,0]-q).clamp(min=0)+(q-soft_limits[...,1]).clamp(min=0)).sum(-1)
 stopped=(command.abs().sum(-1)<0.05).float()
 return {
 "tracking_xy":torch.exp(-((velocity-command[:,:2])**2).sum(-1)/0.25),
 "tracking_yaw":torch.exp(-(yaw_rate-command[:,2])**2/0.25),
 "upright":torch.exp(-gravity[:,:2].square().sum(-1)/0.16),
 "vertical_velocity":s[:,st.V_YAW][:,2].square(),
 "roll_pitch_rate":s[:,st.W_B][:,:2].square().sum(-1),
 "action_rate":(action-previous_action).square().sum(-1),
 "joint_limit":excess,
 "standing_motion":stopped*(velocity.square().sum(-1)+yaw_rate.square())}
WEIGHTS={"tracking_xy":2.0,"tracking_yaw":1.0,"upright":1.0,
"vertical_velocity":-0.5,"roll_pitch_rate":-0.05,"action_rate":-0.05,
"joint_limit":-5.0,"standing_motion":-2.0}
def reward(s,command,action,previous_action,soft_limits,default_q):
 terms=reward_terms(s,command,action,previous_action,soft_limits,default_q)
 return sum(WEIGHTS[k]*v for k,v in terms.items())


# The generator state is used only as a replay container for velocity commands.
# No ball positions or terrain truth are included in the actor's command observation.
from dataclasses import dataclass
from go2_supermimic.env.command import TargetCommand,TargetCommandCfg
from go2_supermimic.env.env_cfg import go2_chase_env_cfg
from mjlab.managers.reward_manager import RewardTermCfg
from go2_supermimic.supermimic.imagine import Imagination
from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import RslRlOnPolicyRunnerCfg

class VelocityCommand(TargetCommand):
 def _resample_command(self,env_ids):
  # Each command is in the yaw-aligned frame, held for 4–6 seconds.
  low=torch.tensor([r[0] for r in CONTRACT["command_ranges"]],device=self.device)
  high=torch.tensor([r[1] for r in CONTRACT["command_ranges"]],device=self.device)
  command=low+(high-low)*torch.rand((len(env_ids),3),device=self.device)
  stop=torch.rand(len(env_ids),device=self.device)<CONTRACT["standing_probability"]
  command[stop]=0
  self.gen.chase_vel[env_ids]=command
  self.gen.has_chase[env_ids]=False
  self.gen.has_dodge[env_ids]=False
 def _advance(self,dt):
  self.gen.has_chase[:]=False
  self.gen.has_dodge[:]=False
  self.hit.zero_()
 def _update_command(self,env_ids):
  self._obs=command_observation(self.gen.chase_vel)
 def _update_metrics(self):
  pass
@dataclass(kw_only=True)
class VelocityCommandCfg(TargetCommandCfg):
 def build(self,env):return VelocityCommand(self,env)

def real_reward(env):
 if not hasattr(env,"_velocity_reward_consts"):
  env._velocity_reward_consts=st.robot_consts(env)
 c=env._velocity_reward_consts
 command=env.command_manager.get_term("target").gen.chase_vel
 return reward(st.robot_state(env),command,env.action_manager.action,
               env.action_manager.prev_action,c.soft_limits,c.default_q)

class VelocityImagination(Imagination):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.critic_dim=st.STATE_DIM+3+st.CONTEXT_DIM
  if self.use_obst or self.use_scan:raise ValueError("Blind velocity actor cannot include exteroception")
 def critic_input(self,s,pose,gst,xi):
  return torch.cat([s,gst["chase_vel"],xi],dim=-1)
 def _rewards(self,s0,s1,pose1,gst,a,a_prev,k):
  terms=reward_terms(s1,gst["chase_vel"],a,a_prev,k["soft_limits"],k["default_q"])
  return sum(WEIGHTS[n]*v for n,v in terms.items())*self.cfg.dt,terms
 def _step(self,s,pose,a_prev,hist,gst,xi,enc_bias,k,dt,actor,wm):
  x=self.critic_input(s,pose,gst,xi)
  a=actor(hist)
  # Flat world model receives a zero-relief privileged scan; actor sees no scan.
  relief=self.relief(pose,gst,self.cfg.sensor_grad)
  s1,dpose=wm(s,a,a_prev,xi,relief)
  pose1=st.apply_pose_delta(pose,dpose)
  r,_=self._rewards(s,s1,pose1,gst,a,a_prev,k)
  d=self.fall_prob(s1)
  r=r+self.cfg.weights.terminated*d*self.cfg.dt
  frame=st.obs_frame(s1,a,command_observation(gst["chase_vel"]),enc_bias,noise=True)
  hist=st.push_history(hist,frame,self.h,self.terms)
  return s1,pose1,a,hist,gst,r,d,x

def flat_cfg(play=False):
 cfg=go2_chase_env_cfg(play=play)
 cfg.commands["target"]=VelocityCommandCfg(resampling_time_range=(4.,6.),debug_vis=False,
                                          mode_probs=(1.,0.,0.),kind_curriculum=False)
 cfg.rewards={"velocity":RewardTermCfg(func=real_reward,weight=1.),
              "terminated":cfg.rewards["terminated"]}
 assert tuple(cfg.observations["actor"].terms)==tuple(n for n,_ in st.OBS_TERMS)
 return cfg
register_mjlab_task(task_id="Go2-Velocity-Blind-Flat",env_cfg=flat_cfg(),
                   play_env_cfg=flat_cfg(True),rl_cfg=RslRlOnPolicyRunnerCfg())

