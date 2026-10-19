"""Goal-conditioned sensor-driven navigation and anticipatory locomotion trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import imageio.v2 as imageio
import numpy as np
from .active_env import ActiveTerrainEnv
from .terrain_env import terrain_config
from .terrain_planner import TerrainNavigator


def video_frame(env, info):
    from PIL import Image, ImageDraw
    frame = Image.fromarray(env.render())
    draw = ImageDraw.Draw(frame)
    mode = "NJU jump" if env.bbc_active else "joint jump" if env.skill.active else "anticipatory walk"
    if not env.bbc_active and env.bbc is not None:
        if env.runup_stage in ("retreat", "brake", "abort"):
            mode = "MoE "+env.runup_stage
        elif env.bbc_approaching or env.cfg.get("jump_backend") == "nju_all" or (env.cfg.get("jump_backend") == "nju_runup" and env.jump_count == 0):
            mode = "NJU runup"
    lines = [f"GOAL ({env.goal[0]:.1f}, {env.goal[1]:.1f})   remaining {info['distance']:.2f} m",
             f"Mode: {mode}   sensors: {env.sensor_mode}",
             f"Lift requests: {info['anticipatory_lifts']}   jumps: {info['jumps']}",
             f"Landing observed: {env.landing_confirmed}   guard: {env.jump_guard or 'none'}"]
    draw.rectangle((0, 0, 640, 70), fill=(20, 25, 30))
    for i, line in enumerate(lines):
        draw.text((8, 4+16*i), line, fill=(240, 245, 250))
    depth = np.uint8(np.clip(env.depth/env.cfg["max_range"], 0, 1)*255)
    inset = Image.fromarray(depth).convert("RGB").resize((160, 120))
    frame.paste(inset, (480, 360))
    return np.asarray(frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="slalom", choices=["empty", "slalom", "random", "blocked", "stairs", "platform", "gap", "mixed", "course"])
    p.add_argument("--goal", nargs=2, type=float)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=200)
    p.add_argument("--height", type=float, default=.08)
    p.add_argument("--gap-width", type=float, default=.30)
    p.add_argument("--jump-backend", choices=["primitive", "nju", "nju_all", "nju_runup", "disabled"], default="primitive")
    p.add_argument("--config")
    p.add_argument("--sensors", choices=["both", "depth", "lidar"], default="both")
    p.add_argument("--no-lift", action="store_true")
    p.add_argument("--straight", action="store_true", help="Fixed velocity locomotion ablation, no navigation planner")
    p.add_argument("--max-seconds", type=float, default=75.)
    p.add_argument("--video", action="store_true")
    p.add_argument("--output", default="outputs/active")
    args = p.parse_args()
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")}
    cfg = terrain_config(args.config)
    cfg.update(terrain_height=args.height, gap_width=args.gap_width,
        jump_backend=args.jump_backend, anticipatory_lift=not args.no_lift,
        max_seconds=args.max_seconds, water_recenter=True)
    if args.goal is not None:
        cfg["goal"] = args.goal
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = ActiveTerrainEnv(cfg, scene=args.scene, sensor_mode=args.sensors)
    results = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed+episode)
            import mujoco
            replay_start=float(env.data.time)
            geometry=[]
            for i in range(env.model.ngeom):
                if env.model.geom_bodyid[i]!=0:continue
                typ=int(env.model.geom_type[i]);size=env.model.geom_size[i].tolist()
                if typ not in (0,5,6):continue
                geometry.append(dict(type=typ,size=size,position=env.model.geom_pos[i].tolist(),quaternion=env.model.geom_quat[i].tolist()))
            (out/'scene.json').write_text(json.dumps(dict(geometries=geometry,backend='MuJoCo',role='历史MoE / IK / NJU技能组合',
                joint_names=[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_JOINT,int(i)) for i in env.joint_ids],quaternion_order='wxyz')))
            stream=(out/'replay.jsonl').open('w',buffering=1)
            nav = TerrainNavigator(cfg)
            trace = []
            writer = imageio.get_writer(out/f"episode_{episode}.mp4", fps=10) if args.video else None
            try:
                while True:
                    yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
                    if args.straight:
                        action = np.array([.65, 0., np.clip(-yaw, -1., 1.)])
                    else:
                        action = nav.act(env.points, env.level_water, env.data.qpos[:2], yaw, env.goal, env.command)
                    _, reward, terminated, truncated, info = env.step(action)
                    trace.append(dict(step=env.steps, reward=reward, **info))
                    stream.write(json.dumps(dict(time=float(env.data.time)-replay_start,position=env.data.qpos[:3].tolist(),
                        quaternion=env.data.qpos[3:7].tolist(),joints=env.data.qpos[env.qids].tolist()))+'\n')
                    if writer:
                        writer.append_data(video_frame(env, info))
                    if terminated or truncated:
                        break
            finally:
                stream.close()
                if writer:
                    writer.close()
            require_jump = any(env.goal[0] > right for _, right, _ in getattr(env, "terrain_meta", {}).get("gaps", []))
            info.update(reached_goal=info["success"], seed=args.seed+episode, scene=args.scene,
                        timeout=bool(truncated), goal=env.goal.tolist())
            info["success"] = bool(info["success"] and (not require_jump or info["crossed_gap_airborne"]))
            results.append(info)
            (out/f"trajectory_{episode}.json").write_text(json.dumps(trace, indent=2))
            (out/f"events_{episode}.json").write_text(json.dumps(env.lift_events, indent=2))
            print(json.dumps(info), flush=True)
    finally:
        env.close()
    summary = dict(config=cfg, arguments=vars(args), episodes=len(results),
        successes=sum(r["success"] for r in results), results=results, source_sha256=sources,
        observation_boundary="ray depth/range, ideal RGB, proprioception, simulated odometry; no scene waypoint or obstacle metadata inputs")
    (out/"summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
