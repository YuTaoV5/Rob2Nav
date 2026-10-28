"""Parallel reset-based paired pulses at six prefixes; no physical state copying."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor,WorldModel
from go2_supermimic.supermimic import state as st
from go2_supermimic.target import terrain as tr
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_paired_excitation_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'prefixes':[30,90,150,240,360,450],'seeds':list(range(37000,37006)),'amplitudes':[.05,.15],'num_envs':104,'headings':4,'groups':'baseline,baseline-repeat,12 joints x plus/minus; each group has four matching headings','post_pulse_steps':8,'policy_updated':False,'world_model_updated':False},f)
state={'phase':'collecting','forks':[],'policy_updated':False,'world_model_updated':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save();torch.set_float32_matmul_precision('highest');cfg=task.cfg(True);cfg.scene.num_envs=104;cfg.seed=37000;env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((104,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(104,device=env.device)
final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(final['checkpoint'],map_location=env.device,weights_only=False);actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device).eval();actor.load_state_dict(ck['actor']);actor.requires_grad_(False);wm=WorldModel(tuple(ck['cfg']['wm_hidden'])).to(env.device).eval();wm.load_state_dict(ck['world_model']);wm.requires_grad_(False);reader=st.ContextReader(env)
def step(a,prev):
 env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(a);env._kine_reward_phase=env.command_manager.get_term('target').gen.wait+1
 obs,rew,done,trunc,_=env.step(prev);assert not (done|trunc).any();return obs,a.clone()
def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 return x
try:
 for index,prefix in enumerate([30,90,150,240,360,450]):
  seed=37000+index
  for epsilon in [.05,.15]:
   torch.manual_seed(seed);obs,_=env.reset();prev=torch.zeros((104,12),device=env.device);prefix_actions=[];state.update(seed=seed,prefix=prefix,epsilon=epsilon);save()
   with torch.no_grad():
    for k in range(prefix):
     # Same intended prefix for each corresponding heading, despite solver differences.
     a=actor(obs['actor'][:4]).clamp(-8,8).repeat(26,1);prefix_actions.append(a[:4].clone());obs,prev=step(a,prev)
    s0=st.robot_state(env).clone();pose0=st.planar_pose(env).clone();xi=reader.read().clone();relief=tr.relief(pose0[:,:2],pose0[:,2],None,None,grad=False);phase0=env.command_manager.get_term('target').gen.wait.clone();base=actor(obs['actor'][:4]).clamp(-8+epsilon,8-epsilon);pulse=base.repeat(26,1)
    for j in range(12):pulse[(2+2*j)*4:(3+2*j)*4,j]+=epsilon;pulse[(3+2*j)*4:(4+2*j)*4,j]-=epsilon
    physical={n:getattr(env.sim.data,n).detach().clone() for n in ['qpos','qvel','ctrl','act','qacc','qacc_warmstart'] if hasattr(getattr(env.sim.data,n,None),'detach')}
    if 'qpos' in physical:physical['qpos'][:,:3]-=env.scene.env_origins
    records=[];s=s0;pose=pose0;previous=prev.clone()
    for k in range(8):
     a=pulse if k==0 else base.repeat(26,1);p=env.command_manager.get_term('target').gen.wait.clone();rel=tr.relief(pose[:,:2],pose[:,2],None,None,grad=False);obs,prev=step(a,prev);sn=st.robot_state(env).clone();pn=st.planar_pose(env).clone();records.append({'s0':s,'s1':sn,'a_intended':a.clone(),'a_previous':previous,'a_executed':previous.clone(),'xi':xi,'relief':rel,'pose0':pose,'pose_delta':st.pose_delta(pose,pn),'phase0':p,'phase1':env.command_manager.get_term('target').gen.wait.clone()});s=sn;pose=pn;previous=prev.clone()
    s1=records[0]['s1'].reshape(26,4,-1);s2=records[1]['s1'].reshape(26,4,-1);real1=torch.stack([(s1[2+2*j,:,st.V_YAW][:,:2]-s1[3+2*j,:,st.V_YAW][:,:2])/(2*epsilon) for j in range(12)]);real2=torch.stack([(s2[2+2*j,:,st.V_YAW][:,:2]-s2[3+2*j,:,st.V_YAW][:,:2])/(2*epsilon) for j in range(12)])
    # Frozen parent response uses one common baseline input per matching heading.
    ps0=s0[:4].repeat(24,1);px=xi[:4].repeat(24,1);pr=relief[:4].repeat(24,1);pa=prev[:4] # overwritten below with actual pre-pulse previous input
    pa=records[0]['a_previous'][:4].repeat(24,1);plusminus=pulse[8:];q1,_=wm(ps0,plusminus,pa,px,pr);q2,_=wm(q1,base.repeat(24,1),plusminus,px,pr);q1=q1.reshape(12,2,4,-1);q2=q2.reshape(12,2,4,-1);pred1=(q1[:,0,:,st.V_YAW][...,:2]-q1[:,1,:,st.V_YAW][...,:2])/(2*epsilon);pred2=(q2[:,0,:,st.V_YAW][...,:2]-q2[:,1,:,st.V_YAW][...,:2])/(2*epsilon)
    start_groups=s0.reshape(26,4,-1);start_gap=float((start_groups[2:]-start_groups[:1]).abs().max());physical_gap={n:float((v.reshape(26,4,*v.shape[1:])[2:]-v.reshape(26,4,*v.shape[1:])[:1]).abs().max()) if v.numel() else 0. for n,v in physical.items()};noise=float((s2[0,:,st.V_YAW][:,:2]-s2[1,:,st.V_YAW][:,:2]).norm());tensors={k:torch.stack([v[k] for v in records]) for k in records[0]};history=float((tensors['a_previous'][1:]-tensors['a_intended'][:-1]).abs().max());assert history==0 and torch.equal(tensors['a_previous'],tensors['a_executed'])
   artifact=OUT/f'pulse_prefix{prefix}_eps{epsilon:.2f}.pt';payload=cpu({'seed':seed,'prefix':prefix,'epsilon':epsilon,'group_layout':'groups0/1 baseline repeats, groups2+2*j plus and3+2*j minus, each4 corresponding headings','initial_joint_perturbation':False,'prefix_actions':torch.stack(prefix_actions),'pre_pulse_physical_qpos_relative_to_origin':physical,'tensors':tensors,'actual_one_step':real1,'actual_two_step':real2,'parent_predicted_one_step':pred1,'parent_predicted_two_step':pred2});torch.save(payload,artifact)
   state['forks'].append({'seed':seed,'prefix':prefix,'epsilon':epsilon,'split':'train' if index<4 else 'validation' if index==4 else 'test','transitions':104*8,'baseline_repeat_velocity_gap':noise,'real_two_step_effect_norm':float(real2.norm()),'real_one_step_response_norm':float(real1.norm()),'parent_two_step_response_relative_error':float((pred2-real2).norm()/real2.norm().clamp(min=1e-8)),'parent_two_step_response_cosine':float(torch.nn.functional.cosine_similarity(pred2.flatten(),real2.flatten(),dim=0)),'start_state_max_gap':start_gap,'pre_pulse_physical_max_gap':physical_gap,'action_history_error':history,'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'bytes':artifact.stat().st_size});save()
 env.close();state.update(phase='completed_pending_review',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(Path(final['checkpoint']).read_bytes()).hexdigest(),limitations='Parallel different environment instances with corresponding heading and identical intended action prefix; physical starts NOT asserted identical. Qpos comparison subtracts environment origin on clones only. Prefix/seed jointly change; initial poses not perturbed. Pulses are excitation data, not a functional policy or physical-envelope acceptance. No failure/contact filtering, no model/actor update, no physical state writes except normal resets.')
except Exception as exc:state.update(phase='failed_requires_review',error=repr(exc))
save();print(json.dumps(state,indent=2))
