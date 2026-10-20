"""Repeatable closed-loop trials; episode traces and stable-stop checks retained."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
from go2_nav.env import Go2MaplessEnv, load_config
from go2_nav.planner import ReactivePlanner
from .controller import action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--checkpoint')
    p.add_argument('--scene', choices=['empty', 'slalom', 'blocked', 'random'], default='empty')
    p.add_argument('--goal', nargs=2, type=float, default=[2., .5])
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--seed', type=int, default=300)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    cfg['goal'] = args.goal
    policy = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        policy = PPO.load(args.checkpoint, device='cpu')
    env = Go2MaplessEnv(cfg, scene=args.scene, sensor_mode='lidar')
    results = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed+episode)
            planner = ReactivePlanner(cfg)
            trace = []
            reached = False
            hold = 0
            while True:
                cmd = np.zeros(3) if reached else action(env, planner, policy)
                _, _, terminated, truncated, info = env.step(cmd)
                trace.append(dict(time=float(env.data.time), **info))
                if info['collision'] or info['fallen']:
                    break
                if not reached and info['success']:
                    reached = True
                if reached:
                    hold += 1
                    if hold >= round(2/cfg['navigation_dt']):
                        break
                if truncated:
                    break
            stable_stop = bool(reached and info['distance'] < cfg['goal_tolerance']+.15 and
                not info['collision'] and not info['fallen'] and np.linalg.norm(env.data.qvel[:2]) < .20)
            result = dict(episode=episode, seed=args.seed+episode, reached=reached, stable_stop=stable_stop,
                success=stable_stop, distance=info['distance'], collision=info['collision'], fallen=info['fallen'],
                seconds=info['seconds'], terminal_speed_m_s=float(np.linalg.norm(env.data.qvel[:2])))
            results.append(result)
            (out/f'trace_{episode}.json').write_text(json.dumps(trace))
            print(json.dumps(result), flush=True)
    finally:
        env.close()
    (out/'summary.json').write_text(json.dumps(dict(arguments=vars(args), backend='MuJoCo',
        planner='PPO' if policy else 'reactive', episodes=results, successes=sum(r['success'] for r in results),
        total=len(results), config=cfg, hardware_connected=False,
        success_definition='Within goal tolerance, then 2s zero command; no collision/fall, final distance < tolerance+.15m and horizontal speed < .20m/s'), indent=2))


if __name__ == '__main__':
    main()
