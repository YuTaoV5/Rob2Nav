"""Cross a complete episode; verify zero-dt reset and intended-action histories."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1'
torch.manual_seed(36714);cfg=task.cfg(True);cfg.scene.num_envs=4;cfg.seed=36714
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((4,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(4,device=env.device)
obs,_=env.reset();ps=list((BASE/'logs/supermimic').glob('*_animal_crawl_kine_teacher_smoke_v1_20261028/model_24.pt'));assert len(ps)==1
ck=torch.load(ps[0],map_location=env.device,weights_only=False);actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval()
g=env.command_manager.get_term('target').gen;phase=torch.tensor([0.,1.,10.,100.],device=env.device);mixed=task.advance({'wait':phase,'chase_vel':torch.zeros((4,3),device=env.device)},torch.tensor([1/60,0,1/60,0],device=env.device))
assert torch.equal(mixed['wait'],phase+torch.tensor([1.,0,1.,0.],device=env.device))
delay=torch.zeros((4,12),device=env.device);resets=0;errors={'phase':0.,'target_history':0.,'intended_action_history':0.}
with torch.inference_mode():
 for i in range(735):
  old=g.wait.clone();action=actor(obs['actor']).clamp(-8,8);env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(action);env._kine_reward_phase=old+1
  obs,_,done,truncated,_=env.step(delay);reset=done|truncated;resets+=int(reset.sum());delay=action.clone();delay[reset]=0
  expected=old+1;expected[reset]=0
  errors['phase']=max(errors['phase'],float((g.wait-expected).abs().max()))
  errors['target_history']=max(errors['target_history'],float((obs['actor'][:,210:].reshape(4,5,28)[:,-1]-task.command_observation(g.wait)).abs().max()))
  errors['intended_action_history']=max(errors['intended_action_history'],float((obs['actor'][:,150:210].reshape(4,5,12)[:,-1]-env._kine_intended_action).abs().max()))
assert resets>=4 and all(v<1e-5 for v in errors.values())
report={'passed':True,'num_envs':4,'control_steps':735,'physics_steps':2940,'reset_environment_events':resets,'mixed_zero_and_normal_dt_passed':True,'errors':errors,'task_source_sha256':hashlib.sha256(Path(task.__file__).read_bytes()).hexdigest(),'meaning':'complete-episode control contract, no training or functional qualification'}
(ROOT/'animal_reset_contract_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));env.close()
