"""FR-Net functional teacher -> independently trained SuperMimic recovery student."""
import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_recovery as task
from go2_supermimic.supermimic import train as training,driver
class DirectCommand:
 def __init__(self,term):self.term=term
 def __call__(self,*a):return self.term.gen.chase_vel
 def resample(self,*a):pass
driver.MoeCtsDriver=task.FRTeacher
Original=training.Trainer
class RecoveryTrainer(Original):
 def __init__(self,cfg):
  assert cfg.task==task.TASK and cfg.driver_policy and cfg.driver_frac>0 and not cfg.resume
  super().__init__(cfg);self.steer=DirectCommand(self.term);self.imag.recovery_limits=task.velocity_limits(self.env)
  self.teacher_frames=0;self.clean_error=0.;self.bc_steps=0;self.bc_optimizer=torch.optim.SGD(self.actor.parameters(),lr=.0001);self.cache=None
  original_act=self.driver.act
  def act(command):
   clean=original_act(command).detach();self.cache=(self.obs.detach().clone(),clean);return clean
  self.driver.act=act;original_step=self.env.step
  def step(action):
   n=int(self.driver_scale()*self.cfg.driver_frac*self.cfg.num_envs);mask=self.env_index>=self.cfg.num_envs-n
   if n and self.cache is not None:action[mask]=self.cache[1][mask];self.teacher_frames+=n
   self.bc_mask=mask;return original_step(action)
  self.env.step=step
  files=[Path(__file__),Path(task.__file__)]
  self.contract={'cfg':asdict(cfg),'task':task.TASK,'teacher_sha256':hashlib.sha256(Path(cfg.driver_policy).read_bytes()).hexdigest(),'teacher_gate':'functional recovery6/6; velocity and limits failed, explicitly accepted by user as demonstrations','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'actor_dim':self.obs.shape[-1],'actor_exteroception':False,'teacher_fraction':cfg.driver_frac,'kp':30,'kd':.8,'delay_steps':0,'reset_hold':True,'episode_seconds':10,'termination':'timeout only; falling is initial task state','bc_weight':.05,'bc_lr':.0001,'bc_filter':'none, function-only teacher; student reward regularizes','capability_qualified':False}
  assert self.obs.shape[-1]==350
  (self.log_dir/'recovery_contract.json').write_text(json.dumps(self.contract,indent=2))
 def improve_policy(self):
  result=super().improve_policy()
  if self.cache is not None and self.bc_mask.any():
   obs,targets=self.cache;loss=(self.actor(obs[self.bc_mask])-targets[self.bc_mask]).square().mean()*.05*self.driver_scale();self.bc_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.);self.bc_optimizer.step();self.bc_steps+=1;result['teacher_bc_loss']=float(loss.detach())
  return result
 def run(self):
  super().run();c=self.consts;s=task.st.robot_state(self.env);r=task.reward(s,self.env.action_manager.action,self.env.action_manager.prev_action,c.default_q,c.action_scale,self.imag.recovery_limits);err=float((r-task.real_reward(self.env)).abs().max());assert err<1e-5
  audit={'training_completed':True,'reward_max_error':err,'teacher_frames':self.teacher_frames,'bc_steps':self.bc_steps,'capability_qualified':False};(self.log_dir/'recovery_audit.json').write_text(json.dumps(audit,indent=2));print('RECOVERY_AUDIT',audit,flush=True)
training.Trainer=RecoveryTrainer;training.Imagination=task.RecoveryImagine
if __name__=='__main__':torch.set_float32_matmul_precision('high');training.main()
