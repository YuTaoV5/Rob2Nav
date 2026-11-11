"""Torch-only Kine2Go teacher inputs/reference; wxyz, native reference joint order.

Privileged root pose belongs to the teacher interface, never the blind student.
Only the audited zero-warmup, one-frame-per-60Hz-step contract is supported.
"""
import torch

def multiply(a,b):
 aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
 q=torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.linalg.cross(av,bv)), -1)
 # Genesis normalizes quaternion products, including non-unit motion data.
 return q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)

def rotate(q,v):
 q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
 xyz=q[...,1:];t=2*torch.linalg.cross(xyz,v)
 return v+q[...,:1]*t+torch.linalg.cross(xyz,t)

def yaw_quaternion(angle):
 z=torch.zeros_like(angle);return torch.stack((torch.cos(angle/2),z,z,torch.sin(angle/2)),-1)

class KineReference:
 def __init__(self,motion):
  assert motion.ndim==2 and motion.shape[1]>=61
  self.motion=motion;self.length=len(motion);self.origin=motion[0,48:51]
  self.delta=(motion[-1,48:51]-self.origin).clone();self.delta[2]=0
  inv=motion[0,51:55].clone();inv[1:]*=-1
  # Upstream transform_quat_by_quat(end, inverse_start) means inverse_start * end.
  dq=multiply(inv,motion[-1,51:55]);forward=torch.zeros(3,device=motion.device,dtype=motion.dtype);forward[0]=1
  direction=rotate(dq,forward);self.delta_yaw=torch.atan2(direction[1],direction[0])
 def at(self,steps,heading_offset):
  assert steps.ndim==1 and heading_offset.shape==(len(steps),4)
  frames=steps.remainder(self.length);cycles=steps.div(self.length,rounding_mode='floor');assert bool((steps>=0).all())
  translation=torch.zeros((len(steps),3),device=self.motion.device,dtype=self.motion.dtype)
  # Same finite cycle accumulation as upstream; no motion-state writes.
  for k in range(int(cycles.max())):
   delta=rotate(yaw_quaternion(self.delta_yaw*k),self.delta)
   translation+=(cycles>k)[:,None]*delta
  q=multiply(heading_offset,yaw_quaternion(cycles*self.delta_yaw))
  raw=self.motion[frames]
  xyz=rotate(q,raw[:,48:51]-self.origin)+rotate(heading_offset,translation+self.origin)
  return torch.cat((xyz,multiply(q,raw[:,51:55]),raw[:,6:18]),-1)
 def future_targets(self,episode_steps,start_offset,heading_offset):
  return torch.cat([self.at(episode_steps+start_offset+offset,heading_offset) for offset in (1,2,10,30)],-1)
 @staticmethod
 def advance(episode_steps,control_steps=1):
  assert isinstance(control_steps,int) and control_steps>=0
  return episode_steps+control_steps

class TeacherHistory:
 def __init__(self,num_envs,device,dtype=torch.float32):
  self.history=torch.zeros((num_envs,93),device=device,dtype=dtype)
 def reset(self,ids):self.history[ids]=0
 def observe_after_physics(self,root_pos,root_quaternion,joints_reference_order,current_action):
  frame=torch.cat((root_pos,root_quaternion,joints_reference_order,current_action),-1).clamp(-100,100)
  assert frame.shape==(len(self.history),31)
  self.history=torch.cat((self.history[:,31:],frame.detach()),-1)
 def teacher_observation(self,targets):
  assert targets.shape==(len(self.history),76)
  return torch.cat((self.history,targets),-1)
