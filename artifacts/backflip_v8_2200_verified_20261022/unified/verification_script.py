"""Isolated v7 deployment contract and strict, heading-varied MuJoCo checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-yaw',type=float,default=0.)
    p.add_argument('--entry-hold-diagnostic',type=float,help='Explicit counterfactual: does not match the v7 training contract')
    p.add_argument('--policy-run',type=Path,help='Completed, exported backflip run; default is archived v7')
    p.add_argument('--export-directory',type=Path,help='Explicit exported saved checkpoint; requires selected checkpoint and completed parent run')
    p.add_argument('--selected-checkpoint',type=Path,help='Exact checkpoint in policy-run whose hash must match export metadata')
    p.add_argument('--seed-start',type=int,default=12510)
    p.add_argument('--entry-torque-blend',type=float,default=0.);a=p.parse_args()
    root=Path('/root/gpufree-data/Rob2Nav');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.entry_hold_diagnostic is not None and not 0.<=a.entry_hold_diagnostic<=.4:p.error('Entry hold diagnostic requires [0,.4]')
    fixture=out/'fixture';shutil.copytree(root/'deployment',fixture/'deployment')
    config=fixture/'skills/navigation/config.json';config.parent.mkdir(parents=True)
    shutil.copyfile(root/'skills/navigation/config.json',config)
    parent=a.policy_run.resolve() if a.policy_run else root/'artifacts/backflip_v7_verified_20261022/outputs/fixstand_backflip_upstream_actor_v7_20261022'
    export_dir=a.export_directory.resolve() if a.export_directory else parent/'exported'
    if bool(a.export_directory)!=bool(a.selected_checkpoint) or (a.export_directory and not a.policy_run):
        raise ValueError('Custom export requires both an exact checkpoint and completed parent run')
    export=json.loads((export_dir/'export.json').read_text())
    selected=a.selected_checkpoint.resolve() if a.selected_checkpoint else None
    if selected:
        if selected.parent!=parent or selected.suffix!='.pt' or not selected.is_file():
            raise ValueError('Selected checkpoint must be a saved model inside the completed parent run')
        if hashlib.sha256(selected.read_bytes()).hexdigest()!=export['checkpoint_sha256']:
            raise ValueError('Selected checkpoint hash does not match the exported candidate')
    trained=export['fixstand_roundtrip']
    if not export['entry_training_contract_verified'] or trained.get('flip_direction','back')!='back':
        raise ValueError('Unverified backflip entry contract')
    if a.policy_run and not (parent/'completion.json').is_file():raise ValueError('Training has not completed')
    if hashlib.sha256((export_dir/'policy.onnx').read_bytes()).hexdigest()!=export['policy_sha256']:
        raise ValueError('Export policy hash mismatch')
    for name in ('walk_moe','backflip60'):
        f=fixture/f'deployment/skills/{name}.json';d=json.loads(f.read_text())
        source=export_dir/'policy.onnx' if name=='backflip60' else root/d['policy']
        target=fixture/f'deployment/models/{name}{source.suffix}';shutil.copyfile(source,target)
        d.update(policy=target.relative_to(fixture).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        if name=='backflip60':
            d.update(action_parameterization=trained['action_parameterization'],entry_hold_scale=trained['entry_hold_scale'],actuation='native_pd',
                     atomic_duration_s=5.,recovery_s=2.,simulation_enabled=True,entry_tolerance=.12,
                     motor_control=export['control_contract'],hardware_qualified=False,
                     entry_torque_blend_s=a.entry_torque_blend,
                     blocker='Isolated strict MuJoCo verification; hardware unqualified')
            if a.entry_hold_diagnostic is not None:
                d.update(entry_hold_scale=a.entry_hold_diagnostic,blocker='Changed entry hold counterfactual; NOT the trained v7 execution contract')
        f.write_text(json.dumps(d,indent=2)+'\n')
    sequence=out/'sequence.json';sequence.write_text(json.dumps(dict(steps=[
        dict(skill='stand',seconds=1),dict(skill='walk_moe',seconds=2,velocity=[.15,0,0]),
        dict(skill='backflip60',trigger=True),dict(skill='walk_moe',seconds=2,velocity=[.15,0,0]),
        dict(skill='stand',seconds=2)]),indent=2)+'\n')
    if a.smoke_yaw and not a.smoke:p.error('--smoke-yaw requires --smoke')
    cases=[(a.seed_start-1,a.smoke_yaw)] if a.smoke else [(a.seed_start+i,yaw) for i,yaw in enumerate([0,-15,15,30])]
    def run(case):
        seed,yaw=case;dest=out/f'seed_{seed}';log=(out/f'seed_{seed}.log').open('wb')
        with log:
            result=subprocess.run([sys.executable,'-c',"import sys; sys.path.insert(0, '/tmp/rob2nav_reference_stage_20261022'); from rob2nav.deployment.verify_mujoco import main; main()",'--root',str(fixture),
                '--asset','source','--sequence',str(sequence),'--seed',str(seed),'--initial-yaw-deg',str(yaw),
                '--video','--output',str(dest)],cwd=root,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
        summary=dest/'summary.json'
        if summary.exists():
            data=json.loads(summary.read_text())
            executed=Path('/tmp/rob2nav_reference_stage_20261022/rob2nav/deployment/core.py')
            data['execution_overrides']={'rob2nav/deployment/core.py':str(executed)}
            data['source_sha256']['rob2nav/deployment/core.py']=hashlib.sha256(executed.read_bytes()).hexdigest()
            summary.write_text(json.dumps(data,indent=2)+'\n')
        return dict(seed=seed,initial_yaw_deg=yaw,exit_code=result.returncode,
                    passed=result.returncode==0 and summary.exists() and json.loads(summary.read_text())['passed'])
    with ThreadPoolExecutor(max_workers=2) as pool:reports=list(pool.map(run,cases))
    report=dict(smoke=a.smoke,entry_torque_blend_s=a.entry_torque_blend,
                entry_hold_diagnostic=a.entry_hold_diagnostic,trained_execution_contract=a.entry_hold_diagnostic is None,
                policy_run=str(parent),policy_sha256=export['policy_sha256'],
                selected_checkpoint=str(selected) if selected else None,
                selected_checkpoint_sha256=export.get('checkpoint_sha256'),export_directory=str(export_dir),
                cases=reports,passed=all(r['passed'] for r in reports),hardware_qualified=False,
                isolated_core_sha256=hashlib.sha256(Path('/tmp/rob2nav_reference_stage_20261022/rob2nav/deployment/core.py').read_bytes()).hexdigest(),
                scope=('One heading smoke test' if a.smoke else 'Four heading cases')+'; fixed flat geometry, actual MoE walk/flip/walk; no physical robot qualification')
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
