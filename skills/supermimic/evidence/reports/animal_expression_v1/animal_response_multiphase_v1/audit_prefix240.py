"""Local frozen-WM response versus paired physics branches after normal resets.

The plant has a one-control-frame delay. Compare both one- and two-step action
responses, preserving all baseline-repeat noise and observed-state mismatches.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/root/gpufree-data/go2-supermimic')
import torch
import rob2nav_animal as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor,WorldModel
from go2_supermimic.supermimic import state as st
from go2_supermimic.target import terrain as tr
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_action_response_prefix240_seed36727';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'purpose':'local action-response calibration, no training'},f)
state={'phase':'running','world_model_globally_calibrated':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
try:
 torch.set_float32_matmul_precision('highest');torch.manual_seed(36727);cfg=task.cfg(True);cfg.scene.num_envs=4;cfg.seed=36727
 env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((4,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(4,device=env.device)
 final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(final['checkpoint'],map_location=env.device,weights_only=False)
 actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device).eval();actor.load_state_dict(ck['actor']);actor.requires_grad_(False)
 wm=WorldModel(tuple(ck['cfg']['wm_hidden'])).to(env.device).eval();wm.load_state_dict(ck['world_model']);wm.requires_grad_(False)
 records=[];base_action=None;prefix=240;epsilon=.05
 def step(action,delay):
  env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(action);env._kine_reward_phase=env.command_manager.get_term('target').gen.wait+1
  obs,_,done,truncated,_=env.step(delay);assert not bool((done|truncated).any());return obs,action.clone()
 def branch(joint=None,sign=0):
  nonlocal_placeholder=None
  torch.manual_seed(36727);obs,_=env.reset();delay=torch.zeros((4,12),device=env.device)
  with torch.no_grad():
   for _ in range(prefix):obs,delay=step(actor(obs['actor']).clamp(-8,8),delay)
   s0=st.robot_state(env).clone();pose=st.planar_pose(env);xi=st.ContextReader(env).read();relief=tr.relief(pose[:,:2],pose[:,2],None,None,grad=False)
   physical={n:getattr(env.sim.data,n).detach().clone() for n in ('qpos','qvel','ctrl','act','qacc','qacc_warmstart') if hasattr(getattr(env.sim.data,n,None),'detach')}
   action=actor(obs['actor']).clamp(-8,8) if base_action is None else base_action.clone()
   if joint is not None:action[:,joint]+=sign*epsilon
   assert action.abs().max()<8
   previous=delay.clone();_,delay=step(action,delay);s1=st.robot_state(env).clone()
   second=action.clone() if base_action is None else base_action.clone();_,delay=step(second,delay);s2=st.robot_state(env).clone()
  return {'joint':joint,'sign':sign,'s0':s0,'s1':s1,'s2':s2,'a0':action,'a_prev':previous,'xi':xi,'relief':relief,'physical':physical}
 baseline=branch();base_action=baseline['a0'].clone();repeat=branch();records.extend([baseline,repeat])
 actual_one=[];actual_two=[];predicted_one=[];predicted_two=[];mismatches=[]
 for joint in range(12):
  plus=branch(joint,1);minus=branch(joint,-1);records.extend([plus,minus])
  actual_one.append((plus['s1'][:,st.V_YAW][:,:2]-minus['s1'][:,st.V_YAW][:,:2])/(2*epsilon));actual_two.append((plus['s2'][:,st.V_YAW][:,:2]-minus['s2'][:,st.V_YAW][:,:2])/(2*epsilon))
  with torch.no_grad():
   def predicted(action):
    s1,_=wm(baseline['s0'],action,baseline['a_prev'],baseline['xi'],baseline['relief']);s2,_=wm(s1,base_action,action,baseline['xi'],baseline['relief']);return s1[:,st.V_YAW][:,:2],s2[:,st.V_YAW][:,:2]
   pp=predicted(base_action+torch.nn.functional.one_hot(torch.tensor(joint,device=env.device),12)*epsilon);pm=predicted(base_action-torch.nn.functional.one_hot(torch.tensor(joint,device=env.device),12)*epsilon)
   predicted_one.append((pp[0]-pm[0])/(2*epsilon));predicted_two.append((pp[1]-pm[1])/(2*epsilon))
  for row in (plus,minus):
   mismatch={n:float((row['physical'][n]-baseline['physical'][n]).abs().max()) if row['physical'][n].numel() else 0. for n in baseline['physical']};mismatches.append(mismatch)
 real1=torch.stack(actual_one);real2=torch.stack(actual_two);pred1=torch.stack(predicted_one);pred2=torch.stack(predicted_two)
 noise=float((baseline['s2'][:,st.V_YAW][:,:2]-repeat['s2'][:,st.V_YAW][:,:2]).norm())
 cosine=float(torch.nn.functional.cosine_similarity(real2.flatten(),pred2.flatten(),dim=0));relative=float((real2-pred2).norm()/real2.norm().clamp(min=1e-8))
 def cpu(x):
  if isinstance(x,torch.Tensor):return x.detach().cpu()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,list):return [cpu(v) for v in x]
  return x
 artifact=OUT/'branches.pt';torch.save(cpu({'records':records,'actual_one_step':real1,'actual_two_step':real2,'predicted_one_step':pred1,'predicted_two_step':pred2}),artifact)
 state.update(phase='completed_pending_review',prefix_control_steps=prefix,num_envs=4,epsilon_action=epsilon,branch_count=len(records),one_step_actual_response_norm=float(real1.norm()),one_step_predicted_response_norm=float(pred1.norm()),two_step_actual_response_norm=float(real2.norm()),two_step_predicted_response_norm=float(pred2.norm()),two_step_response_cosine=cosine,two_step_relative_response_error=relative,baseline_repeat_endpoint_velocity_noise_norm=noise,prefix_max_observed_physical_difference={n:max(d[n] for d in mismatches) for n in baseline['physical']},artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Only local paired responses after a 40-step prefix; not global derivative validation or functional qualification. All state writes are normal reset only.')
 env.close()
except Exception as exc:state.update(phase='failed_requires_review',error=repr(exc))
save();print(json.dumps(state,indent=2),flush=True)
