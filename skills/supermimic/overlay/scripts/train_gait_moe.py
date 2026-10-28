"""RoboGauge MoE-assisted SuperMimic: same-command demonstrations plus optional BC.

Teacher-only sampling noise is removed before execution AND replay storage.
Student exploration and differentiable world-model training remain unchanged.
BC is a separately recorded actor-only no-momentum SGD extension, not a claim
that the original paper used this auxiliary objective.
"""
import argparse
from dataclasses import asdict
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rob2nav_moe_teacher import RoboGaugeTeacher

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--teacher-profile',required=True)
parser.add_argument('--teacher-bc-weight',type=float,default=.1)
parser.add_argument('--teacher-bc-lr',type=float,default=.0001)
extra,remaining=parser.parse_known_args()
if not 0<=extra.teacher_bc_weight<=2 or not 0<extra.teacher_bc_lr<=.001:
 raise ValueError('Invalid teacher auxiliary settings')
profile=Path(extra.teacher_profile).resolve()
spec=json.loads(profile.read_text());policy=profile.parents[2]/spec['policy']
sys.argv=[sys.argv[0]]+remaining
import train_gait as base
from go2_supermimic.supermimic import driver as upstream_driver

def teacher_factory(env,path):
 if Path(path).resolve()!=policy.resolve():raise ValueError('driver-policy differs from verified Rob2Nav profile')
 return RoboGaugeTeacher(env,profile)
upstream_driver.MoeCtsDriver=teacher_factory


class DirectCommand:
 def __init__(self,term):self.term=term
 def __call__(self,*unused):return self.term.gen.chase_vel
 def resample(self,*unused):pass


class MoeGaitTrainer(base.GaitTrainer):
 allow_moe_teacher=True
 def __init__(self,cfg):
  if not cfg.driver_policy or not 0<cfg.driver_frac<=.5:
   raise ValueError('Explicit teacher model and fraction (0,.5] required')
  super().__init__(cfg)
  self.steer=DirectCommand(self.term)
  self.bc_optimizer=torch.optim.SGD(self.actor.parameters(),lr=extra.teacher_bc_lr)
  self.bc_updates=0;self.teacher_frames=0;self.bc_eligible_frames=0;self.command_error=0.;self.clean_error=0.
  self.teacher_observations=None;self.teacher_actions=None;self.teacher_mask=None
  original_act=self.driver.act
  def act(command):
   self.command_error=max(self.command_error,float((command-self.term.gen.chase_vel).abs().max()))
   self.teacher_observations=self.obs.detach().clone()
   self.teacher_actions=original_act(command).detach()
   rate=(self.teacher_actions-self.a_prev).abs()*self.consts.action_scale/self.env.step_dt/self.imag.gait_velocity_limits
   self.teacher_target_eligible=rate.amax(-1)<=.8
   return self.teacher_actions
  self.driver.act=act
  original_step=self.env.step
  def step(action):
   count=int(self.driver_scale()*self.cfg.driver_frac*self.cfg.num_envs)
   mask=self.env_index>=self.cfg.num_envs-count
   self.teacher_mask=mask
   if count:
    # In-place update keeps Trainer.collect's replay action identical to execution.
    action[mask]=self.teacher_actions[mask]
    self.clean_error=max(self.clean_error,float((action[mask]-self.teacher_actions[mask]).abs().max()))
    self.teacher_frames+=count
   result=original_step(action)
   robot=self.env.scene['robot'];q=robot.data.joint_pos;limits=robot.data.joint_pos_limits
   state_valid=((q>=limits[...,0])&(q<=limits[...,1])).all(-1)&(robot.data.joint_vel.abs()/self.imag.gait_velocity_limits<=.8).all(-1)
   # A data-quality mask for auxiliary imitation only; real transitions and all
   # evaluation samples remain intact, including the teacher's known failures.
   self.teacher_mask=mask & self.teacher_target_eligible & state_valid & ~(result[2]|result[3])
   self.bc_eligible_frames+=int(self.teacher_mask.sum())
   return result
  self.env.step=step
  files=[Path(__file__),Path(sys.modules['rob2nav_moe_teacher'].__file__),Path(base.__file__)]
  self.teacher_record={'teacher':self.driver.metadata,'same_velocity_command':True,
    'teacher_fraction':cfg.driver_frac,'teacher_fade':cfg.driver_fade,'teacher_action_noise':0.,
    'bc_weight':extra.teacher_bc_weight,'bc_lr':extra.teacher_bc_lr,'bc_optimizer':'actor-only no-momentum SGD',
    'bc_quality_filter':'teacher mapped rate<=.8, post-transition joint speed<=.8 and hard positions valid; no terminal transition',
    'student_observation_unchanged':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'capability_qualified':False}
  self.record['cfg']=asdict(cfg);self.record['teacher_training']=self.teacher_record
  (self.log_dir/'gait_contract.json').write_text(json.dumps(self.record,indent=2))
  (self.log_dir/'teacher_contract.json').write_text(json.dumps(self.teacher_record,indent=2))

 def improve_policy(self):
  metrics=super().improve_policy()
  if extra.teacher_bc_weight and self.it>self.actor_frozen_until and self.teacher_mask is not None and bool(self.teacher_mask.any()):
   mask=self.teacher_mask
   prediction=self.actor(self.teacher_observations[mask])
   # Compare physical joint-target offsets, not incompatible raw action scales.
   error=(prediction-self.teacher_actions[mask])*self.consts.action_scale[mask]/.25
   loss=extra.teacher_bc_weight*self.driver_scale()*error.square().mean()
   self.bc_optimizer.zero_grad(set_to_none=True);loss.backward()
   norm=base.training.clip_grads(self.actor.parameters(),self.cfg.grad_clip,self.cfg.grad_clip_elem)
   if bool(torch.isfinite(norm)) and bool(torch.isfinite(loss)):
    self.bc_optimizer.step();self.bc_updates+=1
   metrics['teacher_bc_loss']=float(loss.detach());metrics['teacher_bc_grad_norm']=float(norm)
   if self.it % self.cfg.log_interval==0:
    print('MOE_TEACHER',self.it,'samples',int(mask.sum()),'bc_loss',metrics['teacher_bc_loss'],'updates',self.bc_updates,flush=True)
  return metrics

 def save(self,it):
  path=super().save(it)
  checkpoint=torch.load(path,map_location=self.dev,weights_only=False)
  checkpoint['teacher_training']=self.teacher_record
  torch.save(checkpoint,path)
  return path

 def run(self):
  super().run()
  report={'teacher_frames':self.teacher_frames,'bc_eligible_frames':self.bc_eligible_frames,'teacher_calls':self.driver.calls,'teacher_mapping_error':self.driver.mapping_error,
   'teacher_command_error':self.command_error,'teacher_executed_action_error':self.clean_error,'teacher_bc_updates':self.bc_updates,
   'student_only_evaluation_required':True,'capability_qualified':False}
  assert self.teacher_frames>0 and self.command_error<1e-6 and self.clean_error<1e-6 and self.driver.mapping_error<1e-6
  if extra.teacher_bc_weight and self.cfg.iterations>self.actor_frozen_until:assert self.bc_updates>0
  (self.log_dir/'teacher_audit.json').write_text(json.dumps(report,indent=2));print('TEACHER_AUDIT',json.dumps(report),flush=True)


base.training.Trainer=MoeGaitTrainer
if __name__=='__main__':
 torch.set_float32_matmul_precision('high');base.training.main()
