"""Re-run fixed upstream candidates and the exact web terrain presets on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    upstream=root.parent/'Rob2Nav-upstreams/GO2_backflip'
    cases=[]
    for name in ('model_4000','model_5000','model_5500','model_6000','model_6000_actor'):
        cases.append((name,'backflip',['skills.backflip.simulate','--source',str(upstream),'--onnx',str(upstream/(name+'.onnx')),
            '--physics-hz','1000','--seconds','5','--seed','12000','--fixstand-roundtrip','--diagnostics']))
    for skill in ('stairs','platform','mixed'):
        for mode in ('web','straight'):
            cases.append((skill+'_'+mode,skill,['go2_nav.terrain_run','--scene',skill,'--locomotion','moe','--no-jump',
                '--episodes','1','--seed','12000','--max-seconds','45',*(['--straight'] if mode=='straight' else [])]))
    for skill in ('lsd_sideflip','lsd_backflip'):
        cases.append((skill,skill,['rob2nav.backends.source_mujoco','--root',str(root),'--policy',skill,
            '--scene','empty','--seconds','6','--seed','12000']))
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl')
    def run(case):
        name,skill,command=case;start=time.monotonic()
        with (output/(name+'.log')).open('w') as log:
            try:
                proc=subprocess.run([sys.executable,'-m',*command,'--output',str(output/name)],cwd=root,env=env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=600)
                row=dict(name=name,skill=skill,exit_code=proc.returncode,seconds=time.monotonic()-start,diagnostic_only=True)
            except subprocess.TimeoutExpired:row=dict(name=name,skill=skill,error='candidate timed out',diagnostic_only=True)
        summary=output/name/'summary.json'
        if summary.is_file():
            data=json.loads(summary.read_text());data['diagnostic_only']=True
            data['display_name']='候选对比 · '+name
            summary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
            row.update(successes=data.get('successes'),total=data.get('total',data.get('episodes')),
                constraints=data.get('hardware_constraints_pass'),metrics=data.get('metrics'),results=data.get('results'))
        (output/(name+'.result.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,cases))
    (output/'comparison.json').write_text(json.dumps(dict(cases=results,diagnostic_only=True,hardware_connected=False),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
