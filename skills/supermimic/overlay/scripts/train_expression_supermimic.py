"""Bounded SuperMimic smoke with a frozen MoE teacher and explicit choreography.

The existing policy supplies locomotion, the new command program supplies rhythm;
neither component is described as an existing emotional-control checkpoint.
"""
import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_expression as task
from rob2nav_moe_teacher import RoboGaugeTeacher
from go2_supermimic.supermimic import train as training,driver
PROFILE=Path('/root/gpufree-data/Rob2Nav/deployment/skills/walk_moe.json')
def teacher(env,path):
 t=RoboGaugeTeacher(env,PROFILE)
 assert Path(path).resolve()==t.path.resolve()
 return t
driver.MoeCtsDriver=teacher
class Direct:
 def __init__(self,term):self.term=term
 def __call__(self,*a):return self.term.gen.chase_vel
 def resample(self,*a):pass
Original=training.Trainer
class Trainer(Original):
 def __init__(self,cfg):
  assert cfg.task==task.TASK and cfg.driver_policy and not cfg.resume
  super().__init__(cfg);self.steer=Direct(self.term)
  assert self.obs.shape[-1]==350
  self.teacher_frames=0;self.bc_steps=0;self.cache=None
  self.bc=torch.optim.SGD(self.actor.parameters(),lr=.0001)
  oa=self.driver.act
  def act(command):
   clean=oa(command).detach();self.cache=(self.obs.detach().clone(),clean);return clean
  self.driver.act=act;os=self.env.step
  def step(action):
   n=int(self.driver_scale()*self.cfg.driver_frac*self.cfg.num_envs)
   self.teacher_mask=self.env_index>=self.cfg.num_envs-n
   if n and self.cache is not None:
    action[self.teacher_mask]=self.cache[1][self.teacher_mask];self.teacher_frames+=n
   return os(action)
  self.env.step=step
  self.contract={'cfg':asdict(cfg),'styles':task.STYLES,'phase_storage':'packed generator.wait seconds; style=index; speed_cap=period seconds','actor_dim':350,'command_dim':28,'active_command_channels':9,'actor_exteroception':False,'teacher':self.driver.metadata,'teacher_role':'existing locomotion weights plus new rhythmic command program; no released emotion weights claimed','shared_clock':'task.advance used in real and imagined transitions','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(task.__file__)]},'capability_qualified':False}
  (self.log_dir/'expression_contract.json').write_text(json.dumps(self.contract,indent=2))
 def improve_policy(self):
  out=super().improve_policy()
  if self.cache is not None and self.teacher_mask.any():
   obs,targets=self.cache;loss=.05*self.driver_scale()*(self.actor(obs[self.teacher_mask])-targets[self.teacher_mask]).square().mean()
   self.bc.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
   assert torch.isfinite(norm);self.bc.step();self.bc_steps+=1;out['teacher_bc_loss']=float(loss.detach());out['teacher_bc_grad']=float(norm)
  return out
 def run(self):
  super().run();c=self.consts;g=self.term.gen;s=task.st.robot_state(self.env)
  r=task.v.reward(s,g.chase_vel,self.env.action_manager.action,self.env.action_manager.prev_action,c.soft_limits,c.default_q)
  err=float((r-task.real_reward(self.env)).abs().max());assert err<1e-5
  obs=task.observation(g.chase_vel,g.wait,g.style,g.speed_cap);assert obs.shape[-1]==28 and not obs[:,9:].any()
  # Clock and desired-command progression match packed replay seeds exactly.
  a=task.advance(g.state(),self.env.step_dt)
  expected=task.velocity_command(g.wait+self.env.step_dt,g.style,g.speed_cap)
  clock_err=float((a['chase_vel']-expected).abs().max());assert clock_err==0
  audit={'training_completed':True,'reward_max_error':err,'clock_command_error':clock_err,'teacher_frames':self.teacher_frames,'bc_steps':self.bc_steps,'command_padding_zero':True,'capability_qualified':False}
  (self.log_dir/'expression_audit.json').write_text(json.dumps(audit,indent=2));print('EXPRESSION_AUDIT',audit,flush=True)
training.Trainer=Trainer;training.Imagination=task.ExpressionImagination
if __name__=='__main__':torch.set_float32_matmul_precision('high');training.main()
