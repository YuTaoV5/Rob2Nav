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
    p.add_argument('--entry-torque-blend',type=float,default=0.);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=out/'fixture';shutil.copytree(root/'deployment',fixture/'deployment')
    config=fixture/'skills/navigation/config.json';config.parent.mkdir(parents=True)
    shutil.copyfile(root/'skills/navigation/config.json',config)
    parent=root/'artifacts/backflip_v7_verified_20261022/outputs/fixstand_backflip_upstream_actor_v7_20261022'
    export=json.loads((parent/'exported/export.json').read_text())
    if not export['entry_training_contract_verified'] or export['fixstand_roundtrip']['entry_hold_scale']!=.1:
        raise ValueError('Unverified v7 entry contract')
    if hashlib.sha256((parent/'exported/policy.onnx').read_bytes()).hexdigest()!=export['policy_sha256']:
        raise ValueError('Export policy hash mismatch')
    for name in ('walk_moe','backflip60'):
        f=fixture/f'deployment/skills/{name}.json';d=json.loads(f.read_text())
        source=parent/'exported/policy.onnx' if name=='backflip60' else root/d['policy']
        target=fixture/f'deployment/models/{name}{source.suffix}';shutil.copyfile(source,target)
        d.update(policy=target.relative_to(fixture).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        if name=='backflip60':
            d.update(action_parameterization='backflip_entry_v3',entry_hold_scale=.1,actuation='native_pd',
                     atomic_duration_s=5.,recovery_s=2.,simulation_enabled=True,entry_tolerance=.12,
                     motor_control=export['control_contract'],hardware_qualified=False,
                     entry_torque_blend_s=a.entry_torque_blend,
                     blocker='Isolated strict MuJoCo verification; hardware unqualified')
        f.write_text(json.dumps(d,indent=2)+'\n')
    sequence=out/'sequence.json';sequence.write_text(json.dumps(dict(steps=[
        dict(skill='stand',seconds=1),dict(skill='walk_moe',seconds=2,velocity=[.15,0,0]),
        dict(skill='backflip60',trigger=True),dict(skill='walk_moe',seconds=2,velocity=[.15,0,0]),
        dict(skill='stand',seconds=2)]),indent=2)+'\n')
    cases=[(12509,0)] if a.smoke else [(12510,0),(12511,-15),(12512,15),(12513,30)]
    def run(case):
        seed,yaw=case;dest=out/f'seed_{seed}';log=(out/f'seed_{seed}.log').open('wb')
        with log:
            result=subprocess.run([sys.executable,'-m','rob2nav.deployment.verify_mujoco','--root',str(fixture),
                '--asset','source','--sequence',str(sequence),'--seed',str(seed),'--initial-yaw-deg',str(yaw),
                '--video','--output',str(dest)],cwd=root,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
        summary=dest/'summary.json'
        return dict(seed=seed,initial_yaw_deg=yaw,exit_code=result.returncode,
                    passed=result.returncode==0 and summary.exists() and json.loads(summary.read_text())['passed'])
    with ThreadPoolExecutor(max_workers=2) as pool:reports=list(pool.map(run,cases))
    report=dict(smoke=a.smoke,entry_torque_blend_s=a.entry_torque_blend,cases=reports,passed=all(r['passed'] for r in reports),hardware_qualified=False,
                scope=('One heading smoke test' if a.smoke else 'Four heading cases')+'; fixed flat geometry, actual MoE walk/flip/walk; no physical robot qualification')
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
