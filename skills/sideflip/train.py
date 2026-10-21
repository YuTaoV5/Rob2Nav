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
p.add_argument('--fixstand-roundtrip',action='store_true')
p.add_argument('--boundary-residual',action='store_true')
p.add_argument('--boundary-residual-limit',type=float,default=.05)
p.add_argument('--boundary-bridge-limit',type=float)
p.add_argument('--boundary-terminal-hip-taper-frames',type=int,default=0)
p.add_argument('--boundary-blend-frames',type=int,default=0)
p.add_argument('--quiet-preparation',action='store_true')
p.add_argument('--peak-endpoint-objective',action='store_true')
p.add_argument('--finetune-learning-rate',type=float)
p.add_argument('--reset-std',type=float)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.boundary_residual and not args.fixstand_roundtrip:p.error('--boundary-residual requires --fixstand-roundtrip')
if not .01<=args.boundary_residual_limit<=.3:p.error('Boundary residual limit must be in [.01,.3] rad')
if args.boundary_bridge_limit is not None and (not args.boundary_residual or not .01<=args.boundary_bridge_limit<=args.boundary_residual_limit):
    p.error('Boundary bridge limit requires residual mode and value in [.01, hold limit]')
if args.boundary_terminal_hip_taper_frames and (not args.boundary_residual or not 2<=args.boundary_terminal_hip_taper_frames<=50):
    p.error('Terminal hip taper requires residual mode and 2..50 frames')
if args.peak_endpoint_objective and not args.fixstand_roundtrip:p.error('Peak endpoint objective requires FIXSTAND')
if args.boundary_blend_frames and (not args.boundary_residual or not 2<=args.boundary_blend_frames<=20):p.error('Boundary blending requires residual mode and 2..20 frames')
if args.quiet_preparation and not args.boundary_blend_frames:p.error('Quiet preparation requires continuous boundary mapping')
if args.finetune_learning_rate is not None and not 0.<args.finetune_learning_rate<=.001:p.error('Invalid finetune learning rate')
if args.reset_std is not None and not .01<=args.reset_std<=1.:p.error('Invalid reset std')
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
    if args.fixstand_roundtrip:
        from skills.fixstand import configure_sideflip
        configure_sideflip(cfg,contract,args.boundary_residual,args.boundary_residual_limit,args.boundary_bridge_limit,args.boundary_terminal_hip_taper_frames,args.boundary_blend_frames)
    if args.peak_endpoint_objective:
        from skills.fixstand import configure_sideflip_peak_objective
        configure_sideflip_peak_objective(cfg,source,contract['frames']/contract['fps'])
    if args.quiet_preparation:
        from skills.fixstand import configure_quiet_preparation
        configure_quiet_preparation(cfg)
    bodies = contract['body_names']
    anchor = cfg.commands.motion.anchor_body_name
    feet = cfg.terminations.ee_body_pos.params['body_names']
    agent = BasePPORunnerCfg()
    agent.seed = args.seed
    agent.device = cfg.sim.device
    agent.max_iterations = args.iterations
    agent.save_interval = 100
    agent.experiment_name = 'rob2nav_go2_sideflip'
    if args.finetune_learning_rate is not None:
        agent.algorithm.learning_rate=args.finetune_learning_rate
        agent.algorithm.schedule='fixed'
    args.output.mkdir(parents=True, exist_ok=False)
    if args.fixstand_roundtrip:
        from skills.fixstand import write_contract
        write_contract(args.output/'fixstand.json','sideflip',reference_sha256=sha(args.motion),actor_observations=69,
            action_parameterization=('reference_residual_v4' if args.boundary_blend_frames else 'reference_residual_v3' if args.boundary_terminal_hip_taper_frames else 'reference_residual_v2') if args.boundary_residual else 'joint_offset_v1',
            padded_region_frames=90,residual_limit_rad=args.boundary_residual_limit,
            bridge_residual_limit_rad=args.boundary_bridge_limit if args.boundary_bridge_limit is not None else args.boundary_residual_limit,
            terminal_hip_taper_frames=args.boundary_terminal_hip_taper_frames,
            boundary_blend_frames=args.boundary_blend_frames,
            quiet_preparation_seconds=2.4 if args.quiet_preparation else 0.,
            peak_endpoint_objective=args.peak_endpoint_objective)
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
            runner.load(str(args.resume),load_optimizer=args.finetune_learning_rate is None)
        if args.reset_std is not None:
            import math
            parameters=dict(runner.alg.policy.named_parameters());key='log_std' if 'log_std' in parameters else 'std'
            if key not in parameters:raise RuntimeError('Unsupported policy noise parameter')
            with torch.no_grad():parameters[key].fill_(math.log(args.reset_std) if key=='log_std' else args.reset_std)
            runner.alg.optimizer.state.pop(parameters[key],None)
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
