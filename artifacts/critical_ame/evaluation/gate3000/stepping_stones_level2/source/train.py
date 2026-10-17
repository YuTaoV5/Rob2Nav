"""Original upstream training with nonfinite-ray diagnostics, no input repair."""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--upstream',required=True)
p.add_argument('--stage',type=int,choices=[1,2],required=True)
p.add_argument('--num-envs',type=int,default=512)
p.add_argument('--iterations',type=int,required=True)
p.add_argument('--checkpoint')
p.add_argument('--resume',action='store_true')
p.add_argument('--repair-rays',action='store_true')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
root=Path(a.upstream).resolve()
for folder in ('rsl_rl','source/ame_locomotion','scripts/go2'):
    sys.path.insert(0,str(root/folder))
app=AppLauncher(a).app
import torch
import gymnasium as gym
import ame_locomotion.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from ame_locomotion.tasks.manager_based.ame_locomotion.go2 import mdp
from runner import Go2Runner

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'arguments.json').write_text(json.dumps(vars(a),indent=2))
from .records import capture
capture(out,a,root)
original_finite=mdp.finite


def diagnostic_finite(name,x):
    if name.startswith('height scanner') and not torch.isfinite(x).all():
        env=inspect.currentframe().f_back.f_locals.get('env')
        bad=(~torch.isfinite(x).all(-1)).nonzero()
        if env is not None:
            robot=env.scene['robot'];terrain=env.scene.terrain;sensor=env.scene['height_scanner']
            records=[]
            for eid,rid in bad[:64].tolist():
                records.append({'env':eid,'ray':rid,'root_position':robot.data.root_pos_w[eid].tolist(),
                    'origin':env.scene.env_origins[eid].tolist(),'root_quaternion':robot.data.root_quat_w[eid].tolist(),
                    'root_velocity':robot.data.root_lin_vel_w[eid].tolist(),
                    'level':int(terrain.terrain_levels[eid]),'type':int(terrain.terrain_types[eid]),
                    'sensor_position':sensor.data.pos_w[eid].tolist(),'episode_step':int(env.episode_length_buf[eid])})
            (out/'invalid_rays.json').write_text(json.dumps({'common_step':int(env.common_step_counter),
                'count':len(bad),'records':records},indent=2))
    return original_finite(name,x)


mdp.finite=diagnostic_finite
if a.repair_rays:
    from .repair import install
    install(out)
cfg=load_cfg_from_registry(f'AME-Go2-Stage{a.stage}-v0','env_cfg_entry_point')
agent=load_cfg_from_registry(f'AME-Go2-Stage{a.stage}-v0','rsl_rl_cfg_entry_point')
cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device or 'cuda:0'
agent.device=cfg.sim.device;agent.seed=a.seed
dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
env=gym.make(f'AME-Go2-Stage{a.stage}-v0',cfg=cfg)
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
runner=Go2Runner(wrapped,agent.to_dict(),log_dir=str(out),device=cfg.sim.device)
if a.checkpoint:
    if a.resume:
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        info=saved.get('infos')
        if isinstance(info,dict) and info.get('ppo_resume_allowed') is False:
            raise ValueError('Imitation initializer requires warm-start, not PPO resume')
    initialization=runner.load(a.checkpoint,load_optimizer=a.resume)
    if isinstance(initialization,dict) and initialization.get('protocol'):
        # Carry the external-initialization label through later PPO checkpoints.
        initialization=dict(initialization,ppo_resume_allowed=True)
        (out/'initialization.json').write_text(json.dumps(initialization,indent=2))
        original_save=runner.save
        def save_with_protocol(path,infos=None):
            return original_save(path,infos=initialization if infos is None else infos)
        runner.save=save_with_protocol
elif a.resume:raise ValueError('Resume requires checkpoint')
start=time.monotonic()
try:
    runner.learn(a.iterations,init_at_random_ep_len=False)
    (out/'train_summary.json').write_text(json.dumps({'additional_iterations':a.iterations,
        'last_iteration':runner.current_learning_iteration,'elapsed_s':time.monotonic()-start},indent=2))
finally:
    env.close();app.close()
