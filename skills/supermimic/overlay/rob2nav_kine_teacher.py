"""Explicit native-contract Kine2Go teacher transfer to mjlab Go2.

This is a frozen-teacher diagnostic, not a trained SuperMimic student. Simulator,
motor model and randomization differ from Genesis and are reported explicitly.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from rob2nav_kine_reference import KineReference,TeacherHistory,rotate,yaw_quaternion
from mjlab.managers.event_manager import EventTermCfg
import rob2nav_velocity as v
from rob2nav_reset_targets import hold_current_joint_targets
ROOT=Path(__file__).resolve().parent/'reproduction/animal_expression_v1'

def profile(clip,device):
 p=ROOT/'teachers'/clip;c=json.loads((p/'config.json').read_text());a,b=c['policy_cfg']['motion_range']
 motion=torch.tensor(np.load(p/'motion.npy',allow_pickle=False)[a:b],device=device)
 report=json.loads((ROOT/'native_independent'/clip/'36401'/('native_screen_'+clip+'.json')).read_text())['trials'][0]
 assert c['env_cfg']['dof_names']==report['action_joint_names']
 return p,c,motion,report['qpos_joint_names']

def reset_reference(env,env_ids,clip):
 """Only a normal reset event writes initial pose/velocity; never called per step."""
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 _,c,motion,names=profile(clip,env.device);robot=env.scene['robot']
 heading=yaw_quaternion(ids.to(motion)*torch.pi/2)
 ref=KineReference(motion);zero=torch.zeros(len(ids),device=env.device,dtype=torch.long);q=ref.at(zero,heading)
 root=torch.cat((q[:,:3]+env.scene.env_origins[ids],q[:,3:7],rotate(heading,motion[0,55:58].expand(len(ids),-1)),rotate(heading,motion[0,58:61].expand(len(ids),-1))),-1)
 current=list(robot.joint_names);index=[names.index(n) for n in current]
 robot.write_root_state_to_sim(root,env_ids=ids)
 robot.write_joint_state_to_sim(q[:,7:][:,index],motion[0,24:36][index].expand(len(ids),-1),env_ids=ids)

def diagnostic_cfg(clip):
 _,c,_,_=profile(clip,'cpu');e=c['env_cfg'];assert e['control_freq']==60 and e['decimation']==4 and e.get('warmup_time_s',0)==0
 cfg=v.flat_cfg(True);cfg.sim.mujoco.timestep=1/240;cfg.decimation=4;cfg.episode_length_s=9.
 robot=cfg.scene.entities['robot'];robot.init_state.joint_pos=dict(e['default_joint_angles'])
 for actuator in robot.articulation.actuators:
  actuator.stiffness=30.;actuator.damping=1.5;actuator.delay_min_lag=0;actuator.delay_max_lag=0
 cfg.actions['joint_pos'].scale=.25
 cfg.events={'reset_reference':EventTermCfg(mode='reset',func=reset_reference,params={'clip':clip}),'initialize_hold_targets':EventTermCfg(mode='reset',func=hold_current_joint_targets,params={})}
 cfg.terminations={'time_out':cfg.terminations['time_out']}
 return cfg

class KineMuJoCoTeacher:
 def __init__(self,env,clip):
  self.env=env;self.robot=env.scene['robot'];self.path,self.cfg,motion,self.reference_names=profile(clip,env.device)
  assert abs(env.step_dt-1/60)<1e-8 and abs(env.physics_dt-1/240)<1e-8
  self.reference=KineReference(motion);self.history=TeacherHistory(env.num_envs,env.device)
  self.clock=torch.zeros(env.num_envs,device=env.device,dtype=torch.long);self.start=torch.zeros_like(self.clock)
  self.heading=yaw_quaternion(torch.arange(env.num_envs,device=env.device,dtype=motion.dtype)*torch.pi/2)
  self.previous=torch.zeros((env.num_envs,12),device=env.device);self.current=self.previous.clone()
  current=list(self.robot.joint_names);self.to_reference=[current.index(n) for n in self.reference_names]
  action_names=list(env.action_manager.get_term('joint_pos')._target_names);native=self.cfg['env_cfg']['dof_names'];self.to_action=[native.index(n) for n in action_names]
  self.net=torch.nn.Sequential(torch.nn.Linear(169,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12)).to(env.device).eval()
  ck=torch.load(self.path/'logs/model.pt',map_location=env.device,weights_only=True);self.net.load_state_dict({k.removeprefix('actor.'):x for k,x in ck['model_state_dict'].items() if k.startswith('actor.')})
  for p in self.net.parameters():p.requires_grad_(False)
  self.metadata={'clip':clip,'checkpoint_sha256':hashlib.sha256((self.path/'logs/model.pt').read_bytes()).hexdigest(),'role':'external_frozen_teacher','control_hz':60,'physics_hz':240,'PD':[30,1.5],'configured_latency':self.cfg['env_cfg']['action_latency'],'executed_latency_control_steps':1,'native_action_names':native,'mjlab_action_names':action_names,'reference_joint_names':self.reference_names,'physical_differences':['MuJoCo/mjlab Go2 asset and solver instead of Genesis','Rob2Nav DC motor torque-speed curves retained','No Genesis domain randomization reproduced in this frozen diagnostic'],'student_training_started':False}
 def act(self):
  targets=self.reference.future_targets(self.clock,self.start,self.heading)
  obs=self.history.teacher_observation(targets)
  with torch.no_grad():self.current=self.net(obs).clamp(-self.cfg['env_cfg']['clip_actions'],self.cfg['env_cfg']['clip_actions'])
  # Exactly one previous 60Hz control frame; actuator physics-step delay is zero.
  return self.previous[:,self.to_action].clone()
 def after_step(self):
  d=self.robot.data
  self.history.observe_after_physics(d.root_link_pos_w-self.env.scene.env_origins,d.root_link_quat_w,d.joint_pos[:,self.to_reference],self.current)
  self.previous=self.current.clone();self.clock=self.reference.advance(self.clock)
