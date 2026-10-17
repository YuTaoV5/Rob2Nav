"""Run with /root/isaacsim/python.sh -m perceptive_rl.train --headless."""
import argparse
import json
import hashlib
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--blind", action="store_true")
parser.add_argument("--fused", action="store_true", help="Use occlusion-aware depth/lidar observations instead of ideal height rays")
parser.add_argument("--attention", action="store_true", help="AME-inspired CNN and proprioceptive attention; requires --fused")
parser.add_argument("--kind", choices=["mixed", "flat", "stairs", "platform", "stones"], default="mixed")
parser.add_argument("--resume")
parser.add_argument("--bootstrap", help="Initialize network weights only; start a new PPO optimizer and curriculum")
parser.add_argument("--teacher-velocity", action="store_true", help="Privileged teacher also observes true body velocity (235 inputs)")
parser.add_argument("--output", default="outputs/perceptive/scan")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from .task import make_cfg
from .provenance import snapshot
from .io import actuator_gains

out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
if args.blind and args.fused:
    raise ValueError("--blind and --fused are mutually exclusive")
cfg = make_cfg(args.num_envs, args.seed, args.blind, args.kind, fused=args.fused, teacher_velocity=args.teacher_velocity)
cfg.sim.device = args.device or "cuda:0"
cfg.log_dir = str(out)
agent = UnitreeGo2RoughPPORunnerCfg()
agent.seed = args.seed
agent.device = cfg.sim.device
agent.max_iterations = args.iterations
agent.save_interval = 100
agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
agent.policy.init_noise_std = .6
if args.attention:
    if not args.fused:
        raise ValueError("--attention requires --fused")
    from .attention import register_policy
    register_policy()
    agent.policy.class_name = "AttentionActorCritic"
agent.algorithm.entropy_coef = .005
agent.algorithm.num_learning_epochs = 4
source = snapshot(out)
(out/"manifest.json").write_text(json.dumps({"args": vars(args), "source_sha256": source, "torch": torch.__version__}, indent=2))
dump_yaml(str(out/"env.yaml"), cfg)
dump_yaml(str(out/"agent.yaml"), agent)
env = ManagerBasedRLEnv(cfg=cfg)
wrapper = RslRlVecEnvWrapper(env, clip_actions=10.)
runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(out), device=cfg.sim.device)
robot = env.scene["robot"]
kp, kd = actuator_gains(robot)
(out/"io.json").write_text(json.dumps({"joint_names": robot.joint_names,
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "stiffness": kp,
    "damping": kd,
    "action_scale": cfg.actions.joint_pos.scale,
    "observation_terms": env.observation_manager.active_terms,
    "observation_shapes": {k: list(v) for k, v in env.observation_manager.group_obs_dim.items()},
    "scan": {"x": [-.55, 1.05, .1], "y": [-.5, .5, .1], "ordering": "xy"}}, indent=2))
if args.resume:
    runner.load(args.resume)
if args.bootstrap:
    if args.resume:
        raise ValueError("Choose resume or bootstrap")
    checkpoint = torch.load(args.bootstrap, map_location=cfg.sim.device, weights_only=False)
    state = checkpoint["model_state_dict"]
    if args.attention and "actor.0.weight" in state:
        # Preserve the published student exactly; only the new branch starts fresh.
        initial = runner.alg.policy.state_dict()
        for key, value in state.items():
            target = "actor.base." + key[6:] if key.startswith("actor.") else key
            if target in initial:
                initial[target] = value
        runner.alg.policy.load_state_dict(initial, strict=True)
    else:
        runner.alg.policy.load_state_dict(state, strict=True)
try:
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    (out/"completed.json").write_text(json.dumps({"requested_iterations": args.iterations,
        "last_iteration": runner.current_learning_iteration, "stage": "PPO"}, indent=2))
    print("PPO_COMPLETE", runner.current_learning_iteration, flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    wrapper.close()
    app.close()
