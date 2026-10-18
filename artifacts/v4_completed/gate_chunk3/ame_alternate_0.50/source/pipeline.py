"""Launch only after base-policy/phase checks, then run matched evaluations."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--run',required=True)
p.add_argument('--output',required=True);p.add_argument('--baseline-check',required=True)
p.add_argument('--model-check',required=True);p.add_argument('--smoke',required=True)
p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--python',default='/root/isaacsim/python.sh');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
def status(state,**extra):(out/'status.json').write_text(json.dumps({'state':state,**extra},indent=2))
base=json.loads(Path(a.baseline_check).read_text());model=json.loads(Path(a.model_check).read_text())
smoke=json.loads((Path(a.smoke)/'completed.json').read_text())
phases=[json.loads(s)['phase'] for s in (Path(a.smoke)/'phases.jsonl').read_text().splitlines()]
assert base['total']>=16 and base['successes']/base['total']>=.9 and base['max_zero_fusion_error']==0.
assert model['zero_fusion_exact'] and model['staged_unfreezing_verified']
assert model['checkpoint_sha256']==hashlib.sha256(Path(a.moe).read_bytes()).hexdigest()
assert smoke['phase']=='joint' and phases==['value','fusion','actor','joint']
status('checks_passed',baseline_successes=base['successes'],baseline_total=base['total'])
baseline=[sys.executable,'-m','moe_perceptive.suite','--moe',a.moe,'--output',str(out/'baseline'),
    '--python',a.python,'--video-narrow']
with (out/'baseline.log').open('w') as baseline_log:
    worker=subprocess.Popen(baseline,stdout=baseline_log,stderr=subprocess.STDOUT)
    train=[a.python,'-m','moe_perceptive.train','--headless','--moe',a.moe,'--output',a.run,
        '--num-envs',str(a.num_envs),'--iterations',str(a.iterations)]
    status('training',command=train)
    with (out/'train.log').open('w') as log:result=subprocess.run(train,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':'training','command':train,'returncode':result.returncode})
    baseline_return=worker.wait();records.append({'name':'baseline','command':baseline,'returncode':baseline_return})
    (out/'commands.json').write_text(json.dumps(records,indent=2))
if result.returncode or baseline_return:
    status('failed',training_returncode=result.returncode,baseline_returncode=baseline_return);raise SystemExit(1)
summary=json.loads((Path(a.run)/'completed.json').read_text())
checkpoint=Path(a.run)/f"model_{summary['last_iteration']}.pt"
command=[sys.executable,'-m','moe_perceptive.suite','--moe',a.moe,'--checkpoint',str(checkpoint),
    '--output',str(out/'trained'),'--python',a.python,'--video-narrow']
status('evaluating',checkpoint=str(checkpoint),command=command)
with (out/'trained.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
records.append({'name':'trained','command':command,'returncode':result.returncode})
(out/'commands.json').write_text(json.dumps(records,indent=2))
status('completed' if result.returncode==0 else 'failed',note='No success claim; inspect the per-terrain counts.')
raise SystemExit(result.returncode)
