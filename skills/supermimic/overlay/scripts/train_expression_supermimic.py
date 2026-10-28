"""Bounded SuperMimic smoke with a frozen MoE teacher and explicit choreography.

The existing policy supplies locomotion, the new command program supplies rhythm;
neither component is described as an existing emotional-control checkpoint.
"""
import sys,json,hashlib,argparse
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--actor-initialization')
parser.add_argument('--world-model-parent')
extra,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
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
  initialization={'actor':'from_scratch','world_model':'from_scratch','critic':'from_scratch','optimizers':'fresh'}
  if extra.actor_initialization:
   ip=Path(extra.actor_initialization);init=torch.load(ip,map_location=self.env.device,weights_only=False)
   assert init.get('initialization_only') and init.get('new_training_performed') is False
   assert init['cfg']['task']==task.TASK and init['obs_dim']==350
   assert init['parent_sha256']=='f54551a392112befb3b92985388f66f43fcd8f92c8bd5d43b1bf803b4674ba6f'
   self.actor.load_state_dict(init['actor'])
   initialization.update(actor='warm_start_with_neutral_new_channels',actor_initialization_path=str(ip),actor_initialization_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),parent_sha256=init['parent_sha256'])
  if extra.world_model_parent:
   assert extra.actor_initialization,'World-model trial requires the audited actor initialization'
   wp=Path(extra.world_model_parent);wh=hashlib.sha256(wp.read_bytes()).hexdigest()
   assert wh=='f54551a392112befb3b92985388f66f43fcd8f92c8bd5d43b1bf803b4674ba6f'
   audit_path=Path(__file__).resolve().parents[1]/'reproduction/animal_expression_v1/world_model_transfer_audit.json'
   wa=json.loads(audit_path.read_text());assert wa['parent_sha256']==wh and all(wa['physical_factory_fields_equal'].values())
   scores={r['model']:r['per_horizon_common_normalized_mse'][-1] for r in wa['rows']}
   assert scores['pretrained']<scores['normalization_only']
   parent=torch.load(wp,map_location='cpu',weights_only=False)
   assert tuple(parent['cfg']['wm_hidden'])==tuple(cfg.wm_hidden)
   self.wm.load_state_dict(parent['world_model'])
   initialization.update(world_model='warm_start_weights_and_normalizers',world_model_parent=str(wp),world_model_parent_sha256=wh,world_model_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),critic='from_scratch_new_reward_and_83D_input')
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
  self.contract['initialization']=initialization
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
