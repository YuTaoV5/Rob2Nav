"""Held-out dynamics test on real expression trajectories; no training.

Compares raw-fresh, parent-normalization-only, and pretrained world models on
the SAME recorded states/actions. Reports the parent's common output scale.
"""
import sys,json,hashlib,copy,dataclasses,functools
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic');sys.path.insert(0,str(BASE))
import torch
import rob2nav_expression as task
from rob2nav_reset_targets import install
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target import terrain as tr
OUT=BASE/'reproduction/animal_expression_v1';torch.manual_seed(36300)
parent=BASE/'logs/supermimic/2026-10-26_02-47-27_velocity_gait_moe_dataonly_v1_bounded_20261026/model_1200.pt'
sha=hashlib.sha256(parent.read_bytes()).hexdigest();assert sha=='f54551a392112befb3b92985388f66f43fcd8f92c8bd5d43b1bf803b4674ba6f'
ck=torch.load(parent,map_location='cpu',weights_only=False)
def norm(x):
 if dataclasses.is_dataclass(x):return {f.name:norm(getattr(x,f.name)) for f in dataclasses.fields(x)}
 if isinstance(x,dict):return {str(k):norm(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [norm(v) for v in x]
 if isinstance(x,functools.partial):return {'function':norm(x.func),'args':norm(x.args),'kwargs':norm(x.keywords)}
 if callable(x):return getattr(x,'__module__','')+'.'+getattr(x,'__qualname__',type(x).__name__)
 if x is None or isinstance(x,(str,float,int,bool)):return x
 return str(x)
old=install(task.v.flat_cfg(True));cfg=task.cfg(True);old.scene.num_envs=cfg.scene.num_envs=8;cfg.seed=36300
physical={k:norm(getattr(old,k))==norm(getattr(cfg,k)) for k in ('sim','decimation','scene','actions','events')}
assert all(physical.values()),physical
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');term=env.command_manager.get_term('target');original=term._resample_command
def fixed(ids):
 original(ids);term.gen.style[ids]=(ids%4).float();term.gen.speed_cap[ids]=4.
 term.gen.chase_vel[ids]=task.velocity_command(term.gen.wait[ids],term.gen.style[ids],term.gen.speed_cap[ids])
term._resample_command=fixed;obs,_=env.reset();initial=torch.load(OUT/'warmstart_actor_initialization_only.pt',map_location=env.device,weights_only=False)
actor=Actor(350,tuple(initial['cfg']['actor_hidden']),12).to(env.device).eval();actor.load_state_dict(initial['actor'])
ctx=st.ContextReader(env);states=[st.robot_state(env,field=term.height_field).clone()];acts=[];previous=[];contexts=[];reliefs=[];dposes=[];valid=[]
with torch.inference_mode():
 for i in range(160):
  pose=st.planar_pose(env);a=actor(obs['actor']);assert torch.isfinite(a).all() and a.abs().max()<8
  acts.append(a.clone());previous.append(env.action_manager.action.clone());contexts.append(ctx.read().clone())
  reliefs.append(tr.relief(pose[:,:2],pose[:,2],term.height_field,term.obstacle_layout).clone())
  obs,_,done,trunc,_=env.step(a)
  states.append(st.robot_state(env,field=term.height_field).clone());dposes.append(st.pose_delta(pose,st.planar_pose(env)));valid.append((~(done|trunc)).float())
env.close()
s=torch.stack(states);a=torch.stack(acts);ap=torch.stack(previous);xi=torch.stack(contexts);rel=torch.stack(reliefs);dp=torch.stack(dposes);v=torch.stack(valid)
starts=torch.arange(0,160,8,device=s.device);idx=starts[:,None]+torch.arange(8,device=s.device)
def windows(x):return x[idx].permute(0,2,1,3).reshape(-1,8,x.shape[-1])
target=windows(s[1:]);pose_target=windows(dp);aa=windows(a);pp=windows(ap);rr=windows(rel);x0=s[starts].reshape(-1,st.STATE_DIM);xx=xi[starts].reshape(-1,st.CONTEXT_DIM)
vv=v[idx].permute(0,2,1).reshape(-1,8).cumprod(1)
learned=WorldModel(tuple(ck['cfg']['wm_hidden'])).to(s.device).eval();learned.load_state_dict(ck['world_model'])
torch.manual_seed(36301);fresh=WorldModel(tuple(ck['cfg']['wm_hidden'])).to(s.device).eval();norm_only=copy.deepcopy(fresh)
norm_only.in_norm.load_state_dict(learned.in_norm.state_dict());norm_only.out_norm.load_state_dict(learned.out_norm.state_dict())
scale=learned.out_norm.var.sqrt()+1e-6;rows=[]
with torch.inference_mode():
 for label,model in [('fresh',fresh),('normalization_only',norm_only),('pretrained',learned)]:
  pred=x0.clone();errors=[];raw=[];pose_errors=[]
  for j in range(8):
   pred,dpose=model(pred,aa[:,j],pp[:,j],xx,rr[:,j])
   e=torch.cat([pred-target[:,j],dpose-pose_target[:,j]],-1)
   errors.append(float(((e/scale).square().mean(-1)*vv[:,j]).sum()/vv[:,j].sum().clamp(min=1)))
   raw.append(float(((pred-target[:,j]).square().mean(-1)*vv[:,j]).sum()/vv[:,j].sum().clamp(min=1)))
   pose_errors.append(float(((dpose-pose_target[:,j]).square().mean(-1)*vv[:,j]).sum()/vv[:,j].sum().clamp(min=1)))
  rows.append({'model':label,'per_horizon_common_normalized_mse':errors,'state_raw_mse':raw,'pose_delta_raw_mse':pose_errors})
critic_old=ck['critic']['net.0.weight'].shape[1];critic_new=st.STATE_DIM+9+st.CONTEXT_DIM
report={'seed':36300,'num_envs':8,'control_transitions':160,'seconds':3.2,'valid_window_samples':int(vv.sum()),'reset_transitions':int((1-v).sum()),'parent_sha256':sha,'physical_factory_fields_equal':physical,'rows':rows,'critic_old_input_dim':critic_old,'critic_new_input_dim':critic_new,'critic_value_semantics_same':False,'critic_note':'new reward and changing commands; old value is not certified, recalibration required','training_started':False,'model_weights_saved':False,'capability_qualified':False,'scope':'frozen dynamics predictions on one held-out expression trajectory batch, not actor-training success'}
(OUT/'world_model_transfer_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
