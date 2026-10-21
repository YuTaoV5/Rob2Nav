"""Wait for serial training, then evaluate frozen endpoints on independent seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();run=a.run.resolve();root=Path(__file__).resolve().parents[1]
    source=root.parent/'Rob2Nav-upstreams/GO2_backflip'
    result=json.loads((run/'evaluation.json').read_text()) if (run/'evaluation.json').exists() else dict(status='waiting',skills={},hardware_connected=False)
    def save():
        tmp=run/'evaluation.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(run/'evaluation.json')
    def command(name,cmd):
        with (run/(name+'.log')).open('w') as log:subprocess.run(list(map(str,cmd)),cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    save()
    try:
        while True:
            status=json.loads((run/'status.json').read_text())
            stages={stage['name']:stage['status'] for stage in status['stages']}
            for skill in ('sideflip','upright','backflip'):
                if stages.get(skill)!='completed':continue
                if len(result['skills'].get(skill,{}).get('episodes',[]))==4:continue
                folder=run/skill;checkpoint=max(folder.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
                completion=json.loads((folder/'completion.json').read_text())
                if completion.get('execution_complete') is not True or completion.get('requested_iterations')!=status['budget']['iterations']:
                    raise RuntimeError(skill+' completion manifest does not match requested training')
                exported=folder/'validated_export'
                if not exported.exists():
                    exporter='skills.backflip.export_cpu' if skill=='backflip' else 'rob2nav.export_actor'
                    command(skill+'_export',[sys.executable,'-m',exporter,'--checkpoint',checkpoint,'--output',exported])
                entry=result['skills'].setdefault(skill,dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           episodes=[],all_action_and_endpoints_pass=False,hardware_qualified=False))
                result['status']='evaluating';save()
                for seed in range(6100,6104):
                    if any(row['seed']==seed for row in entry['episodes']):continue
                    out=folder/f'fixstand_seed{seed}'
                    retry=0
                    while out.exists() and not (out/'summary.json').exists():
                        retry+=1;out=folder/f'fixstand_seed{seed}_retry{retry}'
                    if not (out/'summary.json').exists():
                        if skill=='backflip':
                            cmd=[sys.executable,'-m','skills.backflip.simulate','--source',source,'--onnx',exported/'policy.onnx','--seconds','6']
                        else:
                            cmd=[sys.executable,'-m','rob2nav.mujoco_skill','--kind','sideflip' if skill=='sideflip' else 'upright_walk',
                                 '--policy',exported/'policy.pt','--backflip-source',source,'--seconds','8','--command','.1','0','0']
                            if skill=='sideflip':cmd+=['--motion',run/'sideflip.npz']
                        cmd+=['--fixstand-roundtrip','--physics-hz','1000','--seed',str(seed),'--output',out]
                        if seed==6100:cmd+=['--video']
                        command(f'{skill}_seed{seed}',cmd)
                    report=json.loads((out/'summary.json').read_text())
                    entry['episodes'].append(dict(seed=seed,summary=str(out/'summary.json'),
                        passed=report['successes']==1,metrics=report['metrics'],hardware_constraints_pass=report['hardware_constraints_pass']))
                    save()
                entry['all_action_and_endpoints_pass']=all(row['passed'] for row in entry['episodes'])
                # Hardware remains false even if this small fixed-seed simulation suite passes.
                save()
            if status['status']=='failed':
                result.update(status='training_failed',reason=status.get('reason'));save();return
            if len(result['skills'])==3 and all(len(s['episodes'])==4 for s in result['skills'].values()):
                result['status']='passed_simulation_only' if all(s['all_action_and_endpoints_pass'] for s in result['skills'].values()) else 'not_accepted'
                save();return
            if not a.wait:result['status']='training_not_finished';save();return
            time.sleep(15)
    except BaseException as exc:
        result.update(status='evaluation_failed',reason=repr(exc));save();raise


if __name__=='__main__':main()
