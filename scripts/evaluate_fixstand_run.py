"""Export a completed run and enforce all four-seed motion/control gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['sideflip','backflip','upright_walk','upright_stand'],required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--seed-start',type=int,required=True)
    a=p.parse_args();run=a.run
    if not (run/'completion.json').is_file():raise ValueError('Training has not completed')
    model=max(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if a.kind=='backflip':from skills.backflip.export_cpu import export
    else:from rob2nav.export_actor import export
    export(model,run/'exported');reports=[];source='/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip'
    for seed in range(a.seed_start,a.seed_start+4):
        out=run/f'seed_{seed}'
        if a.kind=='backflip':cmd=['skills.backflip.simulate','--source',source,'--onnx',str(run/'exported/policy.onnx'),'--seconds','5']
        else:
            cmd=['rob2nav.mujoco_skill','--kind',a.kind,'--policy',str(run/'exported/policy.pt'),'--backflip-source',source,'--seconds','8']
            if a.kind=='sideflip':cmd+=['--motion','outputs/fixstand_finetune_20261021/sideflip.npz']
            else:cmd+=['--command',*(['0','0','0'] if a.kind=='upright_stand' else ['.15','0','0'])]
        subprocess.run([sys.executable,'-m',*cmd,'--physics-hz','1000','--fixstand-roundtrip','--diagnostics',
            '--seed',str(seed),'--output',str(out),*(['--video'] if seed==a.seed_start else [])],check=True)
        d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
            constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
        (run/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
    (run/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(r['passed'] for r in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')


if __name__=='__main__':main()
