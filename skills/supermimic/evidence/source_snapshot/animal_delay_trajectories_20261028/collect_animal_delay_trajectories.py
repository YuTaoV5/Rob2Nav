"""Bounded real trajectory acquisition; frozen teacher/student, no optimization."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as task
from rob2nav_kine_teacher import KineMuJoCoTeacher
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
from go2_supermimic.target import terrain as tr
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_delay_trajectories_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'seeds':[36820,36821,36822,36823],'num_envs':16,'control_steps_per_episode':480,'split':{'train':[36820,36821],'validation':[36822],'test':[36823]},'policy_updated':False,'world_model_updated':False},f)
state={'phase':'collecting','episodes':[],'policy_updated':False,'world_model_updated':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save();torch.set_float32_matmul_precision('highest');cfg=task.cfg(True);cfg.scene.num_envs=16;cfg.seed=36820;env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((16,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(16,device=env.device)
final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(final['checkpoint'],map_location=env.device,weights_only=False);actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device).eval();actor.load_state_dict(ck['actor']);actor.requires_grad_(False);teacher=KineMuJoCoTeacher(env,task.CLIP);to_native=[teacher.metadata['mjlab_action_names'].index(n) for n in teacher.metadata['native_action_names']];mask=torch.arange(16,device=env.device)>=8
reader=st.ContextReader(env);metadata={'student_parent_sha256':hashlib.sha256(Path(final['checkpoint']).read_bytes()).hexdigest(),'teacher':teacher.metadata,'roles':['student_noise_.05']*4+['student_noise_.15']*4+['external_teacher']*8,'control_hz':60,'physics_hz':240,'window':8,'initial_perturbation_added':False,'control':'normal task reset, one previous intended-action control frame physically executed; no rollout state writes','no_envelope_or_contact_filtering':True,'actor_dimension':350,'actor_actual_root_or_linear_velocity':False}
try:
 for seed in [36820,36821,36822,36823]:
  torch.manual_seed(seed);obs,_=env.reset();teacher.history.reset(torch.arange(16,device=env.device));teacher.clock.zero_();teacher.previous.zero_();delay=torch.zeros((16,12),device=env.device);ou=torch.zeros_like(delay);rows=[];state['seed']=seed;state['control_step']=0;save()
  initial={n:getattr(env.sim.data,n).detach().cpu().clone() for n in ['qpos','qvel','ctrl','act','qacc','qacc_warmstart'] if hasattr(getattr(env.sim.data,n,None),'detach')}
  with torch.no_grad():
   for step in range(480):
    s0=st.robot_state(env).clone();pose0=st.planar_pose(env).clone();xi=reader.read().clone();relief=tr.relief(pose0[:,:2],pose0[:,2],None,None,grad=False);phase0=env.command_manager.get_term('target').gen.wait.clone();teacher.clock=phase0.long().clone();teacher.act();a=actor(obs['actor']);ou=.9*ou+(.19**.5)*torch.randn_like(ou);scale=torch.full((16,1),.05,device=env.device);scale[4:8]=.15;a=(a+scale*ou).clamp(-8,8);a[mask]=teacher.current[:,teacher.to_action][mask]
    previous=delay.clone();env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(a);env._kine_reward_phase=phase0+1
    obs,rew,done,trunc,info=env.step(previous);ended=done|trunc;s1=st.robot_state(env).clone();pose1=st.planar_pose(env).clone();phase1=env.command_manager.get_term('target').gen.wait.clone();d=teacher.robot.data
    teacher.history.observe_after_physics(d.root_link_pos_w-env.scene.env_origins,d.root_link_quat_w,d.joint_pos[:,teacher.to_reference],a[:,to_native]);teacher.previous=a[:,to_native].clone();teacher.clock=phase1.long().clone();ids=ended.nonzero(as_tuple=False).flatten()
    if len(ids):teacher.history.reset(ids);teacher.clock[ids]=0;teacher.previous[ids]=0
    delay=a.clone();delay[ended]=0
    row={'s0':s0,'s1':s1,'a_intended':a,'a_previous':previous,'a_executed':previous,'xi':xi,'relief':relief,'pose0':pose0,'pose_delta':st.pose_delta(pose0,pose1),'phase0':phase0,'phase1':phase1,'reset_mask':ended,'teacher_mask':mask,'reward':rew}
    rows.append({k:v.detach().cpu().clone() for k,v in row.items()})
    if (step+1)%120==0:state['control_step']=step+1;save()
  tensors={k:torch.stack([row[k] for row in rows]) for k in rows[0]};valid=(~tensors['reset_mask']).all(0);windows=sum(int((~tensors['reset_mask'][start:start+8]).all(0).sum()) for start in range(0,480,8));path=OUT/f'trajectory_{seed}.pt';torch.save({'metadata':metadata,'seed':seed,'initial':initial,'tensors':tensors},path);state['episodes'].append({'seed':seed,'split':'train' if seed<36822 else 'validation' if seed==36822 else 'test','environments':16,'controls':480,'transitions':480*16,'nonoverlap_valid_8step_windows':windows,'reset_events':int(tensors['reset_mask'].sum()),'finite':all(bool(torch.isfinite(v).all()) for v in tensors.values() if v.is_floating_point()),'executed_equals_previous':bool(torch.equal(tensors['a_executed'],tensors['a_previous'])),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size});save()
 env.close();state.update(phase='completed_pending_review',metadata=metadata,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Frozen mixed-policy data only; teacher trajectories can fail or contact body and are retained. Fixed normal reference reset, no added joint perturbation. Seed/episode split disjoint, environments within one episode share simulation context. No functional qualification or WM fitting.')
except Exception as exc:state.update(phase='failed_requires_review',error=repr(exc))
save();print(json.dumps(state,indent=2))
