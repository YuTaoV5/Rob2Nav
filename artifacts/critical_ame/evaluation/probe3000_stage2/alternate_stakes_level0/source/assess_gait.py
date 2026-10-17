"""Bounded assessment of a completed imitation initializer; no automatic PPO."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True)
p.add_argument('--run',required=True);p.add_argument('--output',required=True)
p.add_argument('--python',default='/root/isaacsim/python.sh');a=p.parse_args()
run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
deadline=time.monotonic()+1800
while not (run/'train_summary.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Initializer did not complete within 30 minutes')
    time.sleep(10)
summary=json.loads((run/'train_summary.json').read_text());ck=run/f"model_{summary['last_iteration']}.pt"
commands=[]
for stage,terrains,levels in [(1,['rough','stepping_stones','stairs_up'],['0']),
                              (2,['double_stakes','alternate_stakes'],['0','5'])]:
    command=[sys.executable,'-m','critical_ame.suite','--upstream',a.upstream,'--checkpoint',str(ck),
             '--stage',str(stage),'--terrains',*terrains,'--levels',*levels,'--num-envs','32',
             '--reset-jitter','.03','--yaw-jitter','.04','--repair-rays','--python',a.python,
             '--output',str(out/f'stage{stage}')]
    with (out/f'stage{stage}.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    commands.append({'command':command,'returncode':result.returncode})
    (out/'commands.json').write_text(json.dumps(commands,indent=2))
    if result.returncode:raise SystemExit(result.returncode)
(out/'completed.json').write_text(json.dumps({'note':'External-teacher initialization assessment, not a strict AME scratch run.'}))
