"""Untrained candidate objective; no environment registration or actor changes.

Desired velocity is derived from the reference schedule, expressed in reference
yaw coordinates. Actual velocity remains privileged WM/critic reward state.
"""
import torch
import rob2nav_animal as base
from rob2nav_kine_reference import rotate,yaw_quaternion
from go2_supermimic.supermimic import state as st

def reference_velocity(phase,heading_offset=None):
 ref,_,_,_=base.parameters(phase.device)
 if heading_offset is None:
  heading_offset=torch.zeros((len(phase),4),device=phase.device);heading_offset[:,0]=1
 q0=ref.at(phase.long(),heading_offset);q1=ref.at(phase.long()+1,heading_offset)
 world=(q1[:,:3]-q0[:,:3])*60
 w,x,y,z=q0[:,3:7].unbind(-1);heading=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
 return rotate(yaw_quaternion(-heading),world)[:,:2]

def velocity_bonus(s,phase,weight=2.,sigma=.25):
 assert weight>=0 and sigma>0
 target=reference_velocity(phase)
 return weight*torch.exp(-(s[:,st.V_YAW][:,:2]-target).square().sum(-1)/sigma**2)

def combined_reward(s,phase,action,previous,default_q,scale,weight=2.,sigma=.25):
 return base.reward(s,phase,action,previous,default_q,scale)+velocity_bonus(s,phase,weight,sigma)

def real_bonus(env,weight=2.,sigma=.25):
 phase=getattr(env,'_kine_reward_phase',env.command_manager.get_term('target').gen.wait)
 return velocity_bonus(st.robot_state(env),phase,weight,sigma)
