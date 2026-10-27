"""Independent crawl student: real/imaginary 60Hz, 8-step WM/32-step imagination."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import torch
import rob2nav_animal as task
from rob2nav_kine_teacher import KineMuJoCoTeacher
from go2_supermimic.supermimic import train as training,state as st
from go2_supermimic.supermimic import driver as driver_module

class Driver(KineMuJoCoTeacher):
 def __init__(self,env,path):
  super().__init__(env,task.CLIP);assert Path(path).resolve()==(self.path/'logs/model.pt').resolve()
  self.to_native=[self.metadata['mjlab_action_names'].index(n) for n in self.metadata['native_action_names']];self.clean=None
 def act(self,command):
  super().act();self.clean=self.current[:,self.to_action].detach().clone();return self.clean
 def reset(self,done):
  ids=done.nonzero(as_tuple=False).flatten();self.history.reset(ids);self.clock[ids]=0;self.previous[ids]=0

driver_module.MoeCtsDriver=Driver
training.Imagination=task.AnimalImagination
BaseTrainer=training.Trainer
class Trainer(BaseTrainer):
 def __init__(self,cfg):
  assert cfg.task==task.TASK and cfg.driver_policy and not cfg.resume
  super().__init__(cfg);assert self.obs.shape[-1]==350 and cfg.wm_window==8 and cfg.horizon==32
  assert self.imag.cfg.dt==self.env.step_dt==1/60
  self.env._kine_intended_action=torch.zeros_like(self.a_prev);self.env._kine_previous_intended=torch.zeros_like(self.a_prev)
  self.env._kine_reward_phase=self.term.gen.wait.clone();self.delay=torch.zeros_like(self.a_prev);self.teacher_frames=0;self.bc_steps=0
  self.bc=torch.optim.SGD(self.actor.parameters(),lr=.0001);self.mask=torch.zeros(cfg.num_envs,device=self.dev,dtype=torch.bool);self.cache=None
  original_step=self.env.step
  def step(action):
   n=int(self.driver_scale()*self.cfg.driver_frac*self.cfg.num_envs);self.mask=self.env_index>=self.cfg.num_envs-n
   clean=self.driver.clean
   if clean is not None and n:action[self.mask]=clean[self.mask]
   action[~self.mask]=action[~self.mask].clamp(-8,8)
   self.cache=(self.obs.detach().clone(),clean) if clean is not None else None
   self.env._kine_previous_intended.copy_(self.env._kine_intended_action);self.env._kine_intended_action.copy_(action)
   self.env._kine_reward_phase=self.term.gen.wait+1.
   result=original_step(self.delay.clone());done=result[2]|result[3]
   # The state model sees intended a_t,a_(t-1); real physics executes a_(t-1).
   # Both real and imagined actor histories carry intended a_t, not delayed input.
   self.delay=action.clone();self.delay[done]=0
   d=self.driver.robot.data
   self.driver.history.observe_after_physics(d.root_link_pos_w-self.env.scene.env_origins,d.root_link_quat_w,d.joint_pos[:,self.driver.to_reference],action[:,self.driver.to_native])
   self.driver.clock=self.term.gen.wait.long().clone();self.teacher_frames+=n
   return result
  self.env.step=step
  self.contract={'cfg':asdict(cfg),'task':task.TASK,'actor_dim':350,'command_dim':28,'active_command_channels':18,'actor_actual_global_pose':False,'actor_actual_linear_velocity':False,'actor_heightmap':False,'desired_command':'12 reference joint angles, desired height, 3 desired gravity coordinates, sin/cos phase; 10 zeros','teacher':self.driver.metadata,'control_hz':60,'physics_hz':240,'world_model_window':8,'imagination_horizon':32,'action_history':'intended actions in real and imagined observations; one previous-control-frame action physically executed','student_action_clip':8,'teacher_actions':'native clip100, no envelope-based data filtering','termination':'timeout only; imagined fall probability zero; posture reward remains','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(task.__file__)]},'from_scratch':True,'capability_qualified':False}
  (self.log_dir/'animal_contract.json').write_text(json.dumps(self.contract,indent=2))
 def improve_policy(self):
  stats=super().improve_policy()
  if self.cache is not None and self.mask.any():
   obs,target=self.cache;loss=.05*self.driver_scale()*(self.actor(obs[self.mask])-target[self.mask]).square().mean()
   self.bc.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.);assert torch.isfinite(gn);self.bc.step();self.bc_steps+=1
   stats.update(teacher_bc_loss=float(loss.detach()),teacher_bc_grad=float(gn))
  return stats
 def run(self):
  super().run();s=st.robot_state(self.env);phase=self.term.gen.wait
  shared=task.reward(s,phase,self.env._kine_intended_action,self.env._kine_previous_intended,self.consts.default_q,self.consts.action_scale)
  reward_error=float((shared-task.real_reward(self.env)).abs().max());assert reward_error<1e-5
  cmd=task.command_observation(phase);assert not cmd[:,18:].any()
  target_latest=self.obs[:,210:].reshape(self.cfg.num_envs,5,28)[:,-1];target_error=float((target_latest-cmd).abs().max());assert target_error<1e-5
  action_latest=self.obs[:,150:210].reshape(self.cfg.num_envs,5,12)[:,-1];action_error=float((action_latest-self.env._kine_intended_action).abs().max());assert action_error<1e-5
  g=task.advance(self.term.gen.state(),self.env.step_dt);clock_error=float((g['wait']-phase-1).abs().max());assert clock_error==0
  audit={'training_completed':True,'reward_max_error':reward_error,'actor_target_history_error':target_error,'actor_intended_action_history_error':action_error,'clock_frame_error':clock_error,'command_padding_zero':True,'teacher_frames':self.teacher_frames,'bc_steps':self.bc_steps,'capability_qualified':False}
  (self.log_dir/'animal_audit.json').write_text(json.dumps(audit,indent=2));print('ANIMAL_AUDIT',audit,flush=True)

training.Trainer=Trainer
if __name__=='__main__':torch.set_float32_matmul_precision('high');training.main()
