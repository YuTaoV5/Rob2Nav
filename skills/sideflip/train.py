"""Train a real Go2 motion-imitation task using the pinned Unitree mimic MDP."""
import argparse
import json
from pathlib import Path
import re
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--backflip-source', required=True)
p.add_argument('--unitree-source', required=True)
p.add_argument('--motion', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--num-envs', type=int, default=4096)
p.add_argument('--iterations', type=int, default=5000)
p.add_argument('--seed', type=int, default=43)
p.add_argument('--resume', type=Path)
p.add_argument('--transfer-motors',action='store_true',help='Experimental explicit DC motor envelope without PhysX hard speed clamp')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.kit_args = (args.kit_args+' --enable isaacsim.asset.importer.urdf').strip()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from .common import setup_sources, robot_config, sha, write_json, BACKFLIP_COMMIT, UNITREE_COMMIT


def motion_finished(env):
    cmd = env.command_manager.get_term('motion')
    return cmd.time_steps >= cmd.motion.time_step_total-2


def main():
    source = setup_sources(args.backflip_source, args.unitree_source)
    from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import RobotEnvCfg
    from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
    contract = json.loads(args.motion.with_suffix('.json').read_text())
    if contract['npz_sha256'] != sha(args.motion) or contract['asset_commit'] != BACKFLIP_COMMIT:
        raise RuntimeError('Reference/asset hash mismatch')
    from .task import make_cfg
    cfg = make_cfg(source,args.motion,contract,args.num_envs,args.seed,args.device or 'cuda:0')
    if args.transfer_motors:
        from .common import enable_transfer_motors
        enable_transfer_motors(cfg,source)
        cfg.rewards.undesired_contacts.weight=-10.
    bodies = contract['body_names']
    anchor = cfg.commands.motion.anchor_body_name
    feet = cfg.terminations.ee_body_pos.params['body_names']
    agent = BasePPORunnerCfg()
    agent.seed = args.seed
    agent.device = cfg.sim.device
    agent.max_iterations = args.iterations
    agent.save_interval = 100
    agent.experiment_name = 'rob2nav_go2_sideflip'
    args.output.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(args.output/'env.yaml'), cfg)
    dump_yaml(str(args.output/'agent.yaml'), agent)
    write_json(args.output/'contract.json', dict(robot='Go2', role='real PhysX PPO training, not FK replay',
        unitree_commit=UNITREE_COMMIT, backflip_asset_commit=BACKFLIP_COMMIT,
        reference_sha256=sha(args.motion), source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
        joint_names=contract['joint_names'], body_names=bodies, anchor=anchor, distal_bodies=feet,
        action_scale=.5, kp=40, kd=1, actor_observations=69, action_dim=12,
        policy_hz=round(1/(cfg.sim.dt*cfg.decimation)), physics_hz=round(1/cfg.sim.dt),
        motor_model='dc_linear_v1' if args.transfer_motors else 'ideal_pd', seed=args.seed, requested_iterations=args.iterations,
        num_envs=args.num_envs, hardware_connected=False))
    torch.set_num_threads(4)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=100.)
        runner_cfg = agent.to_dict()
        runner_cfg['obs_groups'] = {'policy':['policy'], 'critic':['critic']}
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(args.output), device=cfg.sim.device)
        if args.resume:
            runner.load(str(args.resume))
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        write_json(args.output/'completion.json', dict(execution_complete=True,
            requested_iterations=args.iterations, capability_verified=False, hardware_connected=False))
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
