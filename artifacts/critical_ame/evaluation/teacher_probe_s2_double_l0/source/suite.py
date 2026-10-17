"""Sequential first-episode evaluation on the upstream stage's terrain families."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--upstream',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--stage',type=int,choices=[1,2],required=True)
p.add_argument('--levels',nargs='+',type=int,default=[0,2,5])
p.add_argument('--terrains',nargs='+')
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--seed',type=int,default=801)
p.add_argument('--velocity',type=float,default=.5);p.add_argument('--video',action='store_true')
p.add_argument('--reset-jitter',type=float,default=0.);p.add_argument('--yaw-jitter',type=float,default=0.)
p.add_argument('--repair-rays',action='store_true')
p.add_argument('--python',default='/root/isaacsim/python.sh');p.add_argument('--output',required=True)
a=p.parse_args()
families=a.terrains or (['stairs_down','stairs_up','slope_down','slope_up','boxes','rough','stepping_stones','gaps'] if a.stage==1 else ['stairs_down','stairs_up','double_stakes','alternate_stakes','bridge','gaps','rails','rough'])
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
for level in a.levels:
    for terrain in families:
        target=out/f'{terrain}_level{level}'
        command=[a.python,'-m','critical_ame.evaluate','--headless','--upstream',a.upstream,
            '--checkpoint',a.checkpoint,'--stage',str(a.stage),'--terrain',terrain,'--level',str(level),
            '--num-envs',str(a.num_envs),'--seed',str(a.seed),'--velocity',str(a.velocity),
            '--reset-jitter',str(a.reset_jitter),'--yaw-jitter',str(a.yaw_jitter),'--output',str(target)]
        if a.repair_rays:command+=['--repair-rays']
        if a.video:command+=['--video']
        with (out/f'{terrain}_level{level}.log').open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record={'command':command,'returncode':proc.returncode}
        if (target/'summary.json').exists():
            result=json.loads((target/'summary.json').read_text())
            record.update(terrain=terrain,level=level,successes=result['successes'],total=result['total'])
        records.append(record);(out/'commands.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
(out/'completed.json').write_text(json.dumps({'evaluations':len(records),'all_commands_passed':True}))
