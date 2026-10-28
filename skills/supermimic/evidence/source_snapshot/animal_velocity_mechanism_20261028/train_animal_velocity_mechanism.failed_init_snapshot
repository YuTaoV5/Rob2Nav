"""Bounded animal reward mechanism: warm actor/causal WM, fresh critic/optimizers."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass,asdict
import torch
import train_animal_supermimic as animal_training
import rob2nav_animal as task
from rob2nav_animal_velocity_objective import velocity_bonus
training=animal_training.training
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1'
OriginalWorldModel=training.WorldModel;OriginalTrainer=animal_training.Trainer;OriginalCfg=training.TrainCfg;ORIGINAL_REWARD=task.reward
class DelayedWorldModel(OriginalWorldModel):
 def _inputs(self,s,a,a_prev,xi,relief):
  neutral=self.in_norm.mean[training.st.STATE_DIM:training.st.STATE_DIM+12][None].expand_as(a)
  return OriginalWorldModel._inputs(s,neutral,a_prev,xi,relief)
@dataclass(kw_only=True)
class MechanismCfg(OriginalCfg):
 actor_initialization:str=''
 world_model_initialization:str=''
 reference_velocity_weight:float=2.
 reference_velocity_sigma:float=.25
 explicit_critic_calibration_iterations:int=8
 mechanism_output:str=''
class MechanismTrainer(OriginalTrainer):
 def __init__(self,cfg):
  assert cfg.actor_initialization and cfg.world_model_initialization and cfg.mechanism_output
  assert not cfg.resume and not cfg.compile and cfg.explicit_critic_calibration_iterations>0
  def reward(s,phase,action,previous,default_q,scale):
   return ORIGINAL_REWARD(s,phase,action,previous,default_q,scale)+velocity_bonus(s,phase,cfg.reference_velocity_weight,cfg.reference_velocity_sigma)
  task.reward=reward
  super().__init__(cfg)
  ac=torch.load(cfg.actor_initialization,map_location=self.dev,weights_only=False);wc=torch.load(cfg.world_model_initialization,map_location=self.dev,weights_only=False)
  assert ac['cfg']['task']==task.TASK;assert wc.get('explicit_delayed_input') is True
  self.actor.load_state_dict(ac['actor']);self.wm.load_state_dict(wc['world_model'])
  assert not self.opt_actor.state and not self.opt_critic.state and not self.opt_wm.state
  self.actor_frozen_until=self.policy_start+cfg.explicit_critic_calibration_iterations
  # Freeze observation normalization too: no hidden policy change during calibration.
  self.actor.norm.update=lambda x:None
  self.initial_actor={k:v.detach().clone() for k,v in self.actor.state_dict().items()};self.initial_critic={k:v.detach().clone() for k,v in self.critic.named_parameters()};self.initial_wm={k:v.detach().clone() for k,v in self.wm.named_parameters()}
  self.actual_actor_steps=[];self.actual_bc_steps=[];self.blocked_bc_steps=[];self.freeze_records=[];self.calibration_end=None
  actor_step=self.opt_actor.step;bc_step=self.bc.step
  def checked_actor_step(*args,**kwargs):
   assert self.it>self.actor_frozen_until;self.actual_actor_steps.append({'iteration':self.it,'lr':self.opt_actor.param_groups[0]['lr']});return actor_step(*args,**kwargs)
  def checked_bc_step(*args,**kwargs):
   if self.it<=self.actor_frozen_until:self.blocked_bc_steps.append(self.it);return None
   self.actual_bc_steps.append(self.it);return bc_step(*args,**kwargs)
  self.opt_actor.step=checked_actor_step;self.bc.step=checked_bc_step
  physical={n:getattr(self.env.sim.data,n).detach().cpu().clone() for n in ['qpos','qvel','ctrl','act','qacc','qacc_warmstart'] if hasattr(getattr(self.env.sim.data,n,None),'detach')};physical['actor_observation']=self.obs.detach().cpu().clone();physical['phase']=self.term.gen.wait.detach().cpu().clone()
  initial_path=self.log_dir/'mechanism_initial.pt';torch.save(physical,initial_path)
  def digest(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
  self.contract.update(variant='animal_reference_velocity_mechanism_v1',from_scratch=False,actor_initialization_sha256=hashlib.sha256(Path(cfg.actor_initialization).read_bytes()).hexdigest(),wm_initialization_sha256=hashlib.sha256(Path(cfg.world_model_initialization).read_bytes()).hexdigest(),critic_initialization='fresh',optimizer_initialization='all fresh',explicit_actor_frozen_until=self.actor_frozen_until,policy_start=self.policy_start,actor_norm_frozen_all_run=True,bc_guard='SGD blocked through actor_frozen_until; legacy bc_steps counts attempts',delayed_wm='current slot held at running mean in both forward and update_stats; intended action remains external memory',reference_velocity_reward={'weight':cfg.reference_velocity_weight,'sigma':cfg.reference_velocity_sigma,'actual_velocity_privileged_reward_only':True},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(task.__file__),Path(animal_training.__file__)]},initial_physical_hashes={k:digest(v) for k,v in physical.items()},initial_tensor_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest())
  (self.log_dir/'mechanism_contract.json').write_text(json.dumps(self.contract,indent=2))
 def actor_delta(self):return max(float((v-self.initial_actor[k]).abs().max()) for k,v in self.actor.state_dict().items() if v.numel())
 def parameter_delta(self,model,initial):return max(float((v-initial[k]).abs().max()) for k,v in model.named_parameters())
 def collect(self):
  result=super().collect()
  if self.it<=self.actor_frozen_until:
   delta=self.actor_delta();assert delta==0;self.freeze_records.append({'iteration':self.it,'after':'collection','actor_state_max_delta':delta})
  return result
 def improve_policy(self):
  stats=super().improve_policy()
  if self.it<=self.actor_frozen_until:
   delta=self.actor_delta();assert delta==0 and not self.opt_actor.state;self.freeze_records.append({'iteration':self.it,'after':'critic_and_guarded_BC','actor_state_max_delta':delta})
  if self.it==self.actor_frozen_until:
   self.calibration_end={'iteration':self.it,'actor_state_max_delta':self.actor_delta(),'critic_parameter_max_delta':self.parameter_delta(self.critic,self.initial_critic),'wm_parameter_max_delta':self.parameter_delta(self.wm,self.initial_wm),'adam_actor_state_entries':len(self.opt_actor.state),'actual_bc_steps_so_far':len(self.actual_bc_steps)}
   assert self.calibration_end['critic_parameter_max_delta']>0 and self.calibration_end['wm_parameter_max_delta']>0
  return stats
 def run(self):
  super().run();assert self.calibration_end is not None and self.actual_actor_steps
  s=training.st.robot_state(self.env);phase=self.env._kine_reward_phase
  shared=task.reward(s,phase,self.env._kine_intended_action,self.env._kine_previous_intended,self.consts.default_q,self.consts.action_scale);reward_error=float((shared-task.real_reward(self.env)).abs().max());assert reward_error<1e-5
  result={'phase':'completed_pending_review','checkpoint':str(self.last_ckpt.resolve()),'log_dir':str(self.log_dir.resolve()),'iterations':self.cfg.iterations,'variant':'animal_reference_velocity_mechanism_v1','weight':self.cfg.reference_velocity_weight,'critic_calibration_end':self.calibration_end,'actor_state_delta_final':self.actor_delta(),'critic_parameter_delta_final':self.parameter_delta(self.critic,self.initial_critic),'wm_parameter_delta_final':self.parameter_delta(self.wm,self.initial_wm),'actual_actor_steps':self.actual_actor_steps,'actual_bc_steps':self.actual_bc_steps,'blocked_bc_steps':self.blocked_bc_steps,'legacy_bc_attempts':self.bc_steps,'teacher_frames':self.teacher_frames,'freeze_records':self.freeze_records,'reward_max_error':reward_error,'actor_norm_frozen_all_run':True,'capability_qualified':False,'wm_response_globally_calibrated':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  p=Path(self.cfg.mechanism_output);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(result,indent=2));print('MECHANISM_COMPLETE',json.dumps(result),flush=True)
training.TrainCfg=MechanismCfg;training.WorldModel=DelayedWorldModel;training.Trainer=MechanismTrainer
if __name__=='__main__':torch.set_float32_matmul_precision('high');training.main()
