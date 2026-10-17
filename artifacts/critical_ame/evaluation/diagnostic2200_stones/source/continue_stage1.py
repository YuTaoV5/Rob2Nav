"""Continue the active stage-1 run to 3000, then evaluate before stage 2.

This is a foreground experiment pipeline, not a scheduler. Failed commands stop
the pipeline, and no unsuccessful base policy is advanced to the complex stage.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--runs',required=True)
p.add_argument('--python',default='/root/isaacsim/python.sh');p.add_argument('--output',required=True)
a=p.parse_args();runs=Path(a.runs);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
records=[]
def status(state,**extra):
    (out/'status.json').write_text(json.dumps({'state':state,'utc':datetime.now(timezone.utc).isoformat(),**extra},indent=2))
def run(command,name):
    status('running',name=name,command=command)
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':command,'returncode':result.returncode})
    (out/'commands.json').write_text(json.dumps(records,indent=2))
    if result.returncode:
        status('failed',name=name,returncode=result.returncode)
        raise SystemExit(result.returncode)
status('waiting_for_1000')
deadline=time.monotonic()+1800
prior=runs/'stage1_fixed0701'
while not (prior/'train_summary.json').exists():
    if (prior/'invalid_rays.json').exists():
        status('failed',reason='An unrecovered invalid observation occurred');raise SystemExit(1)
    if time.monotonic()>deadline:status('failed',reason='Timed out waiting for active run');raise SystemExit(1)
    time.sleep(10)
assert json.loads((prior/'train_summary.json').read_text())['last_iteration']==999
checkpoint=prior/'model_999.pt'
if not checkpoint.exists():raise FileNotFoundError(checkpoint)
train=runs/'stage1_1000_3000'
command=[a.python,'-m','critical_ame.train','--headless','--repair-rays','--upstream',a.upstream,
    '--stage','1','--num-envs','512','--iterations','2000','--seed','42','--resume',
    '--checkpoint',str(checkpoint),'--output',str(train)]
# Early assessment runs independently while stage 1 continues learning.
early=[sys.executable,'-m','critical_ame.suite','--upstream',a.upstream,'--checkpoint',str(checkpoint),
    '--stage','1','--levels','0','--terrains','rough','stepping_stones','stairs_up','--num-envs','32',
    '--reset-jitter','.03','--yaw-jitter','.04','--repair-rays','--python',a.python,'--output',str(out/'eval1000')]
with (out/'eval1000.log').open('w') as log:
    worker=subprocess.Popen(early,stdout=log,stderr=subprocess.STDOUT)
run(command,'stage1_to3000')
worker.wait()
records.append({'name':'eval1000','command':early,'returncode':worker.returncode})
checkpoint=train/'model_2999.pt'
run([sys.executable,'-m','critical_ame.suite','--upstream',a.upstream,'--checkpoint',str(checkpoint),
    '--stage','1','--levels','2','--num-envs','32','--repair-rays','--reset-jitter','.03',
    '--yaw-jitter','.04','--python',a.python,'--output',str(out/'gate3000')],'stage1_gate3000')
gate=[]
for path in (out/'gate3000').glob('*/summary.json'):
    d=json.loads(path.read_text());failures=sum(r['failure'] for r in d['episodes'])
    gate.append({'terrain':d['args']['terrain'],'successes':d['successes'],'total':d['total'],'failures':failures,
        'passes':d['successes']/d['total']>=.8 and failures/d['total']<=.05})
passed=len(gate)==8 and all(g['passes'] for g in gate)
(out/'gate.json').write_text(json.dumps({'passed':passed,'criteria':'Each base family at level2: >=80% supported completion, <=5% failure. Additional validation XY +/-3 cm, yaw +/-0.04 rad.','terrains':gate},indent=2))
status('stage1_gate_passed' if passed else 'stage1_needs_improvement',gate=gate)
