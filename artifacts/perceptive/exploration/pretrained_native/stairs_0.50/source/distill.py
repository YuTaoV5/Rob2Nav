"""DAgger-style imitation: ideal-height teacher -> visible depth/lidar student.

Labels are teacher actions on student-visited states. This stage is imitation,
not reinforcement learning; follow it with train --fused --resume for PPO.
"""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seed", type=int, default=43)
p.add_argument("--output", default="outputs/perceptive/distill")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
from torch import nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from .task import make_cfg
from .provenance import snapshot

torch.set_num_threads(4)
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
sources = snapshot(out)
cfg = make_cfg(args.num_envs, args.seed, fused=True)
cfg.sim.device = args.device or "cuda:0"
# Expose the student to multiple terrain levels, not just the easiest reset row.
cfg.scene.terrain.max_init_terrain_level = 5
env = ManagerBasedRLEnv(cfg=cfg)
wrapper = RslRlVecEnvWrapper(env, clip_actions=10.)
agent = UnitreeGo2RoughPPORunnerCfg()
agent.device = cfg.sim.device
agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
agent.seed = args.seed
agent.policy.init_noise_std = .3
runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(out), device=cfg.sim.device)
state = torch.load(args.teacher, map_location=cfg.sim.device, weights_only=False)["model_state_dict"]
teacher_dim = state["actor.0.weight"].shape[1]
if teacher_dim not in (232, 235):
    raise ValueError("Expected an ideal-height teacher")
teacher = nn.Sequential(nn.Linear(teacher_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12)).to(cfg.sim.device)
teacher.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
teacher.eval()
student = runner.alg.policy
initial = student.state_dict()
for key, value in state.items():
    if key == "actor.0.weight":
        initial[key].zero_()
        initial[key][:, :232] = value[:, 3:] if teacher_dim == 235 else value
        if teacher_dim == 235:
            # Approximate initial velocity by the velocity command; subsequent
            # imitation optimizes all weights using actual teacher labels.
            initial[key][:, 6:9] += value[:, :3]
    elif key in initial and initial[key].shape == value.shape and not key.startswith("std"):
        initial[key] = value
student.load_state_dict(initial)
optimizer = torch.optim.Adam(student.actor.parameters(), lr=3e-4)
obs = wrapper.get_observations()
log = []
try:
    for step in range(args.steps):
        with torch.no_grad():
            label = teacher(obs["critic"] if teacher_dim == 235 else obs["critic"][:, 3:])
        prediction = student.actor(obs["policy"])
        loss = (prediction-label).square().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.actor.parameters(), 1.)
        optimizer.step()
        # Mixed roll-in decays from teacher to student over the first half.
        beta = max(0., 1.-step/(args.steps*.5))
        use_teacher = torch.rand((env.num_envs, 1), device=env.device) < beta
        with torch.no_grad():
            action = torch.where(use_teacher, label, prediction).clamp(-10., 10.)
            obs, reward, done, extras = wrapper.step(action)
        if step % 50 == 0:
            entry = {"step": step, "mse": float(loss), "teacher_probability": beta,
                     "terrain_level": float(env.scene.terrain.terrain_levels.float().mean())}
            log.append(entry); print(json.dumps(entry), flush=True)
        if step % 500 == 0:
            runner.save(str(out/f"student_{step}.pt"))
    runner.save(str(out/"student.pt"))
    (out/"training.json").write_text(json.dumps(log, indent=2))
    (out/"manifest.json").write_text(json.dumps({"args": vars(args),
        "teacher_sha256": hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest(),
        "source_sha256": sources}, indent=2))
    dump_yaml(str(out/"env.yaml"), cfg)
    dump_yaml(str(out/"agent.yaml"), agent)
finally:
    wrapper.close()
    app.close()
