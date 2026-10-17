"""Fixed-difficulty held-out native-physics evaluation and actor export."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--seed", type=int, default=701)
p.add_argument("--kind", choices=["flat", "stairs", "platform", "stones"], default="platform")
p.add_argument("--difficulty", type=float, default=.5)
p.add_argument("--blind", action="store_true")
p.add_argument("--fused", action="store_true")
p.add_argument("--output", required=True)
p.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from torch import nn
from isaaclab.envs import ManagerBasedRLEnv
from .task import make_cfg

torch.set_num_threads(4)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
actor = nn.Sequential(nn.Linear(state["actor.0.weight"].shape[1], 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))
actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
actor.eval().to(args.device or "cuda:0")
torch.jit.script(actor.cpu()).save(str(out/"actor.pt"))
actor.to(args.device or "cuda:0")
cfg = make_cfg(args.num_envs, args.seed, args.blind, args.kind,
               evaluation=True, difficulty=args.difficulty, fused=args.fused)
cfg.sim.device = args.device or "cuda:0"
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (4., 4., 2.5)
cfg.viewer.lookat = (2., 0., 0.)
env = ManagerBasedRLEnv(cfg=cfg, render_mode="rgb_array" if args.video else None)
env._capture_eval = True
robot = env.scene["robot"]
joint_names = robot.joint_names
body_names = env.scene["contact_forces"].body_names
calves = [i for i, name in enumerate(body_names) if "calf" in name]
feet = [i for i, name in enumerate(body_names) if "foot" in name]
obs, _ = env.reset()
done_first = torch.zeros(args.num_envs, device=env.device, dtype=torch.bool)
impulses = torch.zeros(args.num_envs, device=env.device)
max_progress = torch.zeros(args.num_envs, device=env.device)
records, trace = [], []
writer = None
if args.video:
    import imageio.v2 as imageio
    writer = imageio.get_writer(out/"episode_0.mp4", fps=25, macro_block_size=1)
try:
    for step in range(int(cfg.episode_length_s/.02)+5):
        with torch.inference_mode():
            actions = actor(obs["policy"])
            obs, reward, terminated, truncated, extras = env.step(actions.clamp(-10., 10.))
        position = env._eval_position
        forces = env._eval_forces
        impulses += torch.linalg.vector_norm(forces[:, calves], dim=-1).sum(-1)*.02*(~done_first)
        max_progress = torch.maximum(max_progress, position[:, 0]*(~done_first))
        ended = (terminated | truncated) & ~done_first
        for i in ended.nonzero().flatten().tolist():
            reached = bool(env.termination_manager.get_term("reached")[i])
            support = int((forces[i, feet, 2] > 5.).sum())
            upright = float(-env._eval_gravity[i, 2])
            records.append({"env": i, "reached": reached,
                "supported_success": reached and support >= 2 and upright > .7 and float(position[i, 2]) > .18,
                "support_feet": support, "upright": upright,
                "base_contact": bool(env.termination_manager.get_term("base_contact")[i]),
                "off_lane": bool(env.termination_manager.get_term("off_lane")[i]),
                "timeout": bool(env.termination_manager.get_term("time_out")[i]),
                "seconds": round((step+1)*.02, 3), "position": position[i].cpu().tolist(),
                "max_progress": float(max_progress[i]), "calf_impulse_proxy": float(impulses[i])})
        if not done_first[0]:
            trace.append({"t": (step+1)*.02, "position": position[0].cpu().tolist(),
                          "action": actions[0].cpu().tolist()})
            if writer and step % 2 == 0:
                # Follow the robot; reference origin remains fixed for metrics.
                pos = robot.data.root_pos_w[0].cpu().numpy()
                env.sim.set_camera_view(pos+np.array([2.5, 3., 1.8]), pos+np.array([.6, 0., 0.]))
                writer.append_data(env.render())
        done_first |= ended
        if done_first.all():
            break
    checkpoint = Path(args.checkpoint)
    result = {"args": vars(args), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "episodes": records, "successes": sum(r["supported_success"] for r in records),
        "reached": sum(r["reached"] for r in records), "total": len(records),
        "joint_names": joint_names, "body_names": body_names,
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "stiffness": robot.data.joint_stiffness[0].cpu().tolist(),
        "damping": robot.data.joint_damping[0].cpu().tolist(),
        "action_scale": cfg.actions.joint_pos.scale,
        "observation_dim": actor[0].in_features,
        "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}}
    (out/"summary.json").write_text(json.dumps(result, indent=2))
    (out/"trace.json").write_text(json.dumps(trace))
    print("EVALUATION", result["successes"], "/", result["total"], flush=True)
finally:
    if writer:
        writer.close()
    env.close()
    app.close()
