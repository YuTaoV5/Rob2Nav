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
p.add_argument('--transfer-motors',action='store_true')
p.add_argument('--stepping-gait',action='store_true')
p.add_argument('--reset-std',type=float)
p.add_argument('--fixstand-roundtrip',action='store_true')
p.add_argument('--endpoint-residual',action='store_true')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.endpoint_residual and not (args.fixstand_roundtrip and args.stepping_gait):
    p.error('--endpoint-residual requires --fixstand-roundtrip and --stepping-gait')
if args.stepping_gait and not args.transfer_motors:p.error('--stepping-gait requires --transfer-motors')
if args.reset_std is not None and not .01<=args.reset_std<=1.:p.error('--reset-std must be in [0.01,1]')
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
    if args.transfer_motors:
        from skills.sideflip.common import enable_transfer_motors
        enable_transfer_motors(cfg,source)
        cfg.commands.base_velocity.debug_vis=False
    if args.stepping_gait:
        from .task import enable_stepping_rewards
        enable_stepping_rewards(cfg)
    if args.fixstand_roundtrip:
        from skills.fixstand import configure_upright
        configure_upright(cfg,args.endpoint_residual)
    agent = BasePPORunnerCfg()
    agent.seed,agent.device = args.seed,cfg.sim.device
    agent.max_iterations,agent.save_interval = args.iterations,100
    agent.experiment_name = 'rob2nav_go2_upright_walk'
    if args.reset_std is not None:agent.policy.init_noise_std=args.reset_std
    args.output.mkdir(parents=True,exist_ok=False)
    if args.fixstand_roundtrip:
        from skills.fixstand import write_contract
        write_contract(args.output/'fixstand.json','upright',actor_observations=52,phase_seconds=[1.,5.,6.5,8.],
            action_parameterization='upright_endpoint_v2' if args.endpoint_residual else 'joint_offset_v1',
            residual_limit_rad=.20,release_seconds=[1.,1.5],recovery_seconds=[5.,6.5])
    dump_yaml(str(args.output/'env.yaml'),cfg)
    dump_yaml(str(args.output/'agent.yaml'),agent)
    torch.set_num_threads(4)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env,clip_actions=100.)
        write_json(args.output/'contract.json',dict(robot='Go2',asset_commit=BACKFLIP_COMMIT,
            runner_source_commit=UNITREE_COMMIT,role='real PhysX PPO training, no root stabilization force',
            reset_distribution='100% shared standard four-foot stance' if args.fixstand_roundtrip else '50% four feet, 50% balanced upright; skill validation must start on four feet',
            command_frame='horizontal forward is projected negative body Z when upright',
            target_projected_gravity=[-1,0,0],actor_observations=int(wrapped.get_observations()['policy'].shape[-1]),action_dim=12,
            joint_names=env.scene['robot'].joint_names,default_joint_pos=env.scene['robot'].data.default_joint_pos[0].tolist(),
            action_scale=.5,kp=40,kd=1,policy_hz=round(1/(cfg.sim.dt*cfg.decimation)),physics_hz=round(1/cfg.sim.dt),
            motor_model='dc_linear_v1' if args.transfer_motors else 'ideal_pd',seed=args.seed,
            reward_version='stepping_v1' if args.stepping_gait else 'upright_velocity_v1',reset_std=args.reset_std,
            num_envs=args.num_envs,requested_iterations=args.iterations,
            source_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
            shared_asset_source_sha256=sha(Path(__file__).parents[1]/'sideflip'/'common.py'),hardware_connected=False))
        runner_cfg = agent.to_dict()
        runner_cfg['obs_groups'] = {'policy':['policy'],'critic':['policy']}
        runner = OnPolicyRunner(wrapped,runner_cfg,log_dir=str(args.output),device=cfg.sim.device)
        if args.resume:
            if args.fixstand_roundtrip:
                prior=args.resume.parent/'fixstand.json'
                if prior.is_file():
                    import json
                    previous=json.loads(prior.read_text())
                    if previous.get('actor_observations')!=52 or previous.get('phase_seconds')!=[1.,5.,6.5,8.]:
                        raise RuntimeError('Incompatible fixstand phase contract')
                    runner.load(str(args.resume))
                else:
                    from skills.fixstand import initialize_expanded_policy
                    transfer=initialize_expanded_policy(runner,args.resume)
                    write_json(args.output/'transfer.json',transfer)
            else:runner.load(str(args.resume))
        if args.reset_std is not None:
            import math
            parameters=dict(runner.alg.policy.named_parameters())
            key='log_std' if 'log_std' in parameters else 'std'
            if key not in parameters:raise RuntimeError('Unsupported policy noise parameter')
            value=math.log(args.reset_std) if key=='log_std' else args.reset_std
            with torch.no_grad():parameters[key].fill_(value)
            runner.alg.optimizer.state.pop(parameters[key],None)
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
