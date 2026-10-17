"""Experimental external-teacher gait initialization of the original AME actor.

This deliberately differs from the strict scratch replication. It preserves the
AME actor and control contract, but learns on rough terrain with MLP labels.
Only warm-start subsequent PPO from this checkpoint; its optimizer is not PPO.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--teacher-checkpoint',required=True)
p.add_argument('--teacher-io',required=True);p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=500)
p.add_argument('--seed',type=int,default=43)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
root=Path(a.upstream).resolve()
for folder in ('rsl_rl','source/ame_locomotion','scripts/go2'):sys.path.insert(0,str(root/folder))
app=AppLauncher(a).app
import torch
import gymnasium as gym
import ame_locomotion.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from runner import Go2Runner
from .teacher import Teacher,add_scanner
from .repair import install
from .records import capture
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);capture(out,a,root);install(out)
cfg=load_cfg_from_registry('AME-Go2-Stage1-v0','env_cfg_entry_point')
agent=load_cfg_from_registry('AME-Go2-Stage1-v0','rsl_rl_cfg_entry_point')
cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device or 'cuda:0'
g=cfg.scene.terrain.terrain_generator;g.sub_terrains={'rough':g.sub_terrains['rough']}
g.sub_terrains['rough'].proportion=1.
# Keep the original eight-column footprint: before the first reset Isaac Lab
# initializes 512 robot clones on a grid, which exceeds a one-column mesh.
add_scanner(cfg);agent.device=cfg.sim.device;agent.seed=a.seed
dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
env=gym.make('AME-Go2-Stage1-v0',cfg=cfg);e=env.unwrapped
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
runner=Go2Runner(wrapped,agent.to_dict(),log_dir=str(out),device=cfg.sim.device)
runner.load(a.checkpoint,load_optimizer=False)
learner=runner.alg.policy;learner.eval();teacher=Teacher(e,a.teacher_checkpoint,a.teacher_io)
optimizer=torch.optim.Adam(learner.parameters(),lr=1e-4)
obs=wrapped.get_observations();term=e.action_manager.get_term('joint_pos');start=time.monotonic()
meta={'protocol':'external MLP gait imitation; not strict AME scratch replication',
      'teacher_sha256':hashlib.sha256(Path(a.teacher_checkpoint).read_bytes()).hexdigest(),
      'ppo_resume_allowed':False,'terrain':'original stage1 rough only',
      'teacher_probability':'linear 1.0 to 0.5', 'steps_per_iteration':24,
      'supervision':'clipped joint targets in upstream normalized action coordinates'}
(out/'protocol.json').write_text(json.dumps(meta,indent=2))
try:
    for it in range(a.iterations):
        inputs=[];labels=[];prob=1.-.5*it/max(1,a.iterations-1)
        with torch.no_grad():
            for step in range(24):
                expert=teacher(obs).clamp(term.lower,term.upper)
                student=learner.act_inference(obs).clamp(term.lower,term.upper)
                inputs.append(obs['policy'].clone());labels.append(expert.clone())
                actions=torch.where(torch.rand(a.num_envs,1,device=e.device)<prob,expert,student)
                obs,_,_,_=wrapped.step(actions)
        x=torch.cat(inputs);y=torch.cat(labels);losses=[]
        for epoch in range(2):
            for ids in torch.randperm(len(x),device=e.device).chunk(8):
                prediction=learner.act_inference({'policy':x[ids]})
                loss=(prediction-y[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation loss')
                optimizer.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(learner.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
        row={'iteration':it,'mse':sum(losses)/len(losses),'teacher_probability':prob,
             'elapsed_s':time.monotonic()-start}
        with (out/'imitation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if it%10==0:print('IMITATION',row,flush=True)
        if it%100==0 or it==a.iterations-1:
            runner.current_learning_iteration=it
            runner.save(out/f'model_{it}.pt',infos=meta)
    (out/'train_summary.json').write_text(json.dumps({'last_iteration':it,
        'elapsed_s':time.monotonic()-start,'protocol':meta},indent=2))
finally:env.close();app.close()
