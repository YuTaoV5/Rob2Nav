"""Controlled longer-memory adaptation: native smoke, warm start, matched suite."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--initializer',required=True)
p.add_argument('--output',required=True);p.add_argument('--run',required=True)
p.add_argument('--iterations',type=int,default=1000);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
def run(name,command):
    (out/'status.json').write_text(json.dumps({'state':name,'command':command},indent=2))
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':command,'returncode':r.returncode})
    (out/'commands.json').write_text(json.dumps(records,indent=2))
    if r.returncode:
        (out/'status.json').write_text(json.dumps({'state':'failed','stage':name,'returncode':r.returncode}))
        raise SystemExit(r.returncode)
common=['/root/isaacsim/python.sh','-m','moe_perceptive.train','--headless','--moe',a.moe,
    '--initialize-from',a.initializer,'--memory-lifetime','4','--warmup','0','--fusion','0','--actor','0']
smoke=str(Path(a.run).with_name(Path(a.run).name+'_smoke'))
run('smoke',common+['--num-envs','32','--iterations','2','--output',smoke])
if not (Path(smoke)/'completed.json').exists():raise RuntimeError('Native smoke did not finish')
run('training',common+['--num-envs','512','--iterations',str(a.iterations),'--output',a.run])
run('evaluation',['python3','-m','moe_perceptive.suite','--moe',a.moe,'--checkpoint',
    str(Path(a.run)/f'model_{a.iterations-1}.pt'),'--memory-lifetime','4',
    '--video-narrow','--output',str(out/'trained')])
(out/'status.json').write_text(json.dumps({'state':'completed','note':'Inspect case counts; completion is not traversal success.'}))
