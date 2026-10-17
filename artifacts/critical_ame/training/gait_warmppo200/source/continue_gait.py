"""Resume PPO only after a bounded value warmup verified a frozen actor."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
p.add_argument('--warmup',required=True);p.add_argument('--training-output',required=True)
p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=200)
p.add_argument('--python',default='/root/isaacsim/python.sh');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);warm=Path(a.warmup)
def status(state,**extra):(out/'status.json').write_text(json.dumps({'state':state,**extra},indent=2))
status('waiting_for_value_warmup');deadline=time.monotonic()+1200
while not (warm/'train_summary.json').exists():
    if time.monotonic()>deadline:
        status('failed',reason='value warmup timeout');raise TimeoutError('Value warmup did not complete')
    time.sleep(10)
summary=json.loads((warm/'train_summary.json').read_text())
assert summary['critic_only'] and summary['frozen_actor_unchanged']
args=json.loads((warm/'arguments.json').read_text())
checkpoint=warm/f"model_{summary['last_iteration']}.pt"
records=[]
commands=[('ppo',[a.python,'-m','critical_ame.train','--headless','--repair-rays','--resume',
    '--upstream',a.upstream,'--stage',str(args['stage']),'--seed',str(args['seed']),
    '--num-envs',str(args['num_envs']),'--iterations',str(a.iterations),
    '--checkpoint',str(checkpoint),'--output',a.training_output]),
    ('assessment',[sys.executable,'-m','critical_ame.assess_gait','--upstream',a.upstream,
    '--run',a.training_output,'--output',str(out/'assessment'),'--python',a.python])]
for name,command in commands:
    status('running',name=name,command=command)
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':command,'returncode':result.returncode})
    (out/'commands.json').write_text(json.dumps(records,indent=2))
    if result.returncode:
        status('failed',name=name,returncode=result.returncode);raise SystemExit(result.returncode)
status('completed',note='External initialization comparison; no capability inferred from command completion')
