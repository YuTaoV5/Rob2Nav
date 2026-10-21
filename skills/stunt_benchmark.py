"""Repeat a frozen Go2 stunt policy in independent MuJoCo episodes; never train here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['sideflip','upright_walk'],required=True)
    p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--motion',type=Path)
    p.add_argument('--backflip-source',required=True,type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(3100,3108)))
    p.add_argument('--physics-hz',type=int,choices=[200,500,1000,2000],default=1000)
    p.add_argument('--seconds',type=float)
    p.add_argument('--video-first',action='store_true')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if len(a.seeds)!=len(set(a.seeds)):raise ValueError('Use independent, unique seeds')
    policy=a.policy.resolve()
    policy_sha=hashlib.sha256(policy.read_bytes()).hexdigest()
    seconds=a.seconds if a.seconds is not None else (6. if a.kind=='sideflip' else 8.)
    a.output.mkdir(parents=True,exist_ok=False)
    episodes=[];evaluator_sha=None
    for index,seed in enumerate(a.seeds):
        folder=a.output/f'seed_{seed}'
        cmd=[sys.executable,'-m','rob2nav.mujoco_skill','--kind',a.kind,'--policy',str(policy),
            '--backflip-source',str(a.backflip_source.resolve()),'--seconds',str(seconds),
            '--physics-hz',str(a.physics_hz),'--seed',str(seed),'--output',str(folder)]
        if a.motion:cmd.extend(['--motion',str(a.motion.resolve())])
        if a.video_first and index==0:cmd.append('--video')
        with (a.output/f'seed_{seed}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'summary.json').read_text())
        if result['policy_sha256']!=policy_sha:raise ValueError('Policy changed during benchmark')
        if evaluator_sha is not None and result['evaluator_sha256']!=evaluator_sha:
            raise ValueError('Evaluator changed during benchmark')
        evaluator_sha=result['evaluator_sha256']
        episodes.append(dict(seed=seed,path=str(folder),success=bool(result['successes']),
            hardware_constraints_pass=result['hardware_constraints_pass'],metrics=result['metrics']))
        progress=dict(kind=a.kind,backend='MuJoCo',policy_sha256=policy_sha,evaluator_sha256=evaluator_sha,
            successes=sum(x['success'] for x in episodes),total=len(episodes),requested_seeds=a.seeds,
            complete=len(episodes)==len(a.seeds),physics_hz=a.physics_hz,policy_hz=50,seconds=seconds,
            hardware_constraints_passes=sum(x['hardware_constraints_pass'] for x in episodes),
            scope='Fixed scene and motor model; independent +/-0.002 rad initial joint perturbations only; not broad terrain robustness.',
            root_state_writes='initialization only; no in-flight stabilization or pose edits',hardware_connected=False,
            episodes=episodes)
        (a.output/'benchmark.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(dict(seed=seed,success=episodes[-1]['success'],successes=progress['successes'],total=len(episodes))),flush=True)


if __name__=='__main__':main()
