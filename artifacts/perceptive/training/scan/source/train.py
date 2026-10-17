"""Run with /root/isaacsim/python.sh -m perceptive_rl.train --headless."""
import argparse
import json
import hashlib
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--blind", action="store_true")
parser.add_argument("--kind", choices=["mixed", "flat", "stairs", "platform", "stones"], default="mixed")
parser.add_argument("--resume")
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

out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
cfg = make_cfg(args.num_envs, args.seed, args.blind, args.kind)
cfg.sim.device = args.device or "cuda:0"
cfg.log_dir = str(out)
agent = UnitreeGo2RoughPPORunnerCfg()
agent.seed = args.seed
agent.device = cfg.sim.device
agent.max_iterations = args.iterations
agent.save_interval = 100
agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
agent.policy.init_noise_std = .6
agent.algorithm.entropy_coef = .005
agent.algorithm.num_learning_epochs = 4
source = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
(out/"manifest.json").write_text(json.dumps({"args": vars(args), "source_sha256": source, "torch": torch.__version__}, indent=2))
dump_yaml(str(out/"env.yaml"), cfg)
dump_yaml(str(out/"agent.yaml"), agent)
env = ManagerBasedRLEnv(cfg=cfg)
wrapper = RslRlVecEnvWrapper(env, clip_actions=10.)
runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(out), device=cfg.sim.device)
if args.resume:
    runner.load(args.resume)
try:
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
finally:
    wrapper.close()
    app.close()
