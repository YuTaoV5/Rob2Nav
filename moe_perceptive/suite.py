"""Matched baseline/trained first-episode cases, including original 20 cm stakes."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--checkpoint')
p.add_argument('--output',required=True);p.add_argument('--python',default='/root/isaacsim/python.sh')
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--video-narrow',action='store_true')
p.add_argument('--memory-lifetime',type=float,default=1.);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
cases=[('flat',0.),('stairs',.5),('platform',.5)]+[(k,d) for k in ('ame_stakes','ame_alternate') for d in (0.,.5,1.)]
for kind,difficulty in cases:
    name=f'{kind}_{difficulty:.2f}';target=out/name
    command=[a.python,'-m','moe_perceptive.evaluate','--headless','--moe',a.moe,'--kind',kind,
        '--difficulty',str(difficulty),'--num-envs',str(a.num_envs),'--output',str(target),
        '--memory-lifetime',str(a.memory_lifetime)]
    if a.checkpoint:command+=['--checkpoint',a.checkpoint]
    if a.video_narrow and kind.startswith('ame_') and difficulty==1.:command+=['--video']
    with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record={'command':command,'returncode':result.returncode}
    if (target/'summary.json').exists():
        d=json.loads((target/'summary.json').read_text())
        record.update(kind=kind,difficulty=difficulty,successes=d['successes'],total=d['total'])
    records.append(record);(out/'commands.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
(out/'completed.json').write_text(json.dumps({'cases':len(records),'note':'Commands completed; see measured success counts.'}))
