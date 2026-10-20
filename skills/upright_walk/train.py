"""Train Go2 rise and upright velocity tracking in real IsaacLab dynamics."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--backflip-source', required=True)
p.add_argument('--unitree-source', required=True)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--num-envs', type=int, default=4096)
p.add_argument('--iterations', type=int, default=5000)
p.add_argument('--seed', type=int, default=44)
p.add_argument('--resume', type=Path)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.kit_args = (args.kit_args+' --enable isaacsim.asset.importer.urdf').strip()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from skills.sideflip.common import setup_sources, sha, write_json, BACKFLIP_COMMIT, UNITREE_COMMIT
from .task import make_cfg


def main():
    source = setup_sources(args.backflip_source, args.unitree_source)
    from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    cfg = make_cfg(source,args.num_envs,args.seed,args.device or 'cuda:0')
    agent = BasePPORunnerCfg()
    agent.seed,agent.device = args.seed,cfg.sim.device
    agent.max_iterations,agent.save_interval = args.iterations,100
    agent.experiment_name = 'rob2nav_go2_upright_walk'
    args.output.mkdir(parents=True,exist_ok=False)
    dump_yaml(str(args.output/'env.yaml'),cfg)
    dump_yaml(str(args.output/'agent.yaml'),agent)
    torch.set_num_threads(4)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env,clip_actions=100.)
        write_json(args.output/'contract.json',dict(robot='Go2',asset_commit=BACKFLIP_COMMIT,
            runner_source_commit=UNITREE_COMMIT,role='real PhysX PPO training, no root stabilization force',
            reset_distribution='50% four feet, 50% balanced upright; skill validation must start on four feet',
            command_frame='horizontal forward is projected negative body Z when upright',
            target_projected_gravity=[-1,0,0],actor_observations=int(wrapped.get_observations()['policy'].shape[-1]),action_dim=12,
            joint_names=env.scene['robot'].joint_names,default_joint_pos=env.scene['robot'].data.default_joint_pos[0].tolist(),
            action_scale=.5,kp=40,kd=1,policy_hz=50,physics_hz=200,seed=args.seed,
            num_envs=args.num_envs,requested_iterations=args.iterations,
            source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},hardware_connected=False))
        runner_cfg = agent.to_dict()
        runner_cfg['obs_groups'] = {'policy':['policy'],'critic':['policy']}
        runner = OnPolicyRunner(wrapped,runner_cfg,log_dir=str(args.output),device=cfg.sim.device)
        if args.resume:
            runner.load(str(args.resume))
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        write_json(args.output/'completion.json',dict(execution_complete=True,requested_iterations=args.iterations,
            capability_verified=False,hardware_connected=False))
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback, os, sys
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(1)  # Preserve failure status even when Kit shutdown exits with zero.
finally:
    app.close()
