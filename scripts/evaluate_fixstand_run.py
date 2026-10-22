"""Export a completed run and enforce all four-seed motion/control gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def simulation_arguments(kind,run,source,motion=None):
    if kind in ('backflip','frontflip'):
        return ['skills.backflip.simulate','--source',str(source),'--onnx',str(run/'exported/policy.onnx'),'--seconds','5','--direction','front' if kind=='frontflip' else 'back']
    cmd=['rob2nav.mujoco_skill','--kind',kind,'--policy',str(run/'exported/policy.pt'),'--backflip-source',str(source),'--seconds','8']
    if kind=='sideflip':
        if motion is None:raise ValueError('Sideflip evaluation requires --motion matching the exported reference hash and direction')
        cmd+=['--motion',str(motion)]
    else:cmd+=['--command',*(['0','0','0'] if kind in ('upright_stand','upright_step') else ['.15','0','0'])]
    return cmd


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['sideflip','backflip','frontflip','upright_walk','upright_stand','upright_step'],required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--seed-start',type=int,required=True)
    cache=Path(os.environ.get('ROB2NAV_UPSTREAMS',str(Path(__file__).resolve().parents[2]/'Rob2Nav-upstreams')))
    p.add_argument('--source',type=Path,default=cache/'GO2_backflip');p.add_argument('--motion',type=Path)
    a=p.parse_args();run=a.run
    simulation_arguments(a.kind,run,a.source,a.motion)
    if not (run/'completion.json').is_file():raise ValueError('Training has not completed')
    model=max(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if a.kind in ('backflip','frontflip'):from skills.backflip.export_cpu import export
    else:from rob2nav.export_actor import export
    export(model,run/'exported');reports=[]
    for seed in range(a.seed_start,a.seed_start+4):
        out=run/f'seed_{seed}'
        cmd=simulation_arguments(a.kind,run,a.source,a.motion)
        subprocess.run([sys.executable,'-m',*cmd,'--physics-hz','1000','--fixstand-roundtrip','--diagnostics',
            '--seed',str(seed),'--output',str(out),*(['--video'] if seed==a.seed_start else [])],check=True)
        d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
            constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
        (run/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
    (run/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(r['passed'] for r in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')


if __name__=='__main__':main()
