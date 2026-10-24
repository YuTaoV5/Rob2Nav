"""Wait for one existing job; audit and evaluate it once. Never submits training."""
from pathlib import Path
import fcntl,json,subprocess,time,sys,os
root=Path(__file__).resolve().parents[2]
if str(root) not in sys.path:sys.path.insert(0,str(root))
from studio.pipeline_api import request
base=root/'outputs/rebound_after_smoke_20261024';base.mkdir(exist_ok=True)
lock=(base/'evaluation.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=base/'state.json'
def record(**data):
 state.write_text(json.dumps(dict(updated=time.time(),job='5bddd59c880c4f33',**data),indent=2))
try:
 if (base/'started.json').exists():raise RuntimeError('Existing evaluation attempt; inspect it instead of restarting')
 while True:
  jobs=request('jobs');job=next(j for j in jobs if j['id']=='5bddd59c880c4f33')
  if job['state']=='succeeded':break
  if job['state'] not in ('draft','queued','preflight','running'):
   raise RuntimeError('Training ended without success: '+job['state'])
  record(stage='waiting',training_state=job['state'])
  time.sleep(30)
 run=root/'outputs/studio/runs/5bddd59c880c4f33'
 completion=json.loads((run/'completion.json').read_text())
 assert completion['execution_complete'] and completion['requested_iterations']==24
 if (run/'exported').exists():raise RuntimeError('Export already exists; do not duplicate evaluation')
 command=['/root/gpufree-data/envs/go2-mapless/bin/python',str(root/'scripts/evaluate_fixstand_run.py'),
          '--kind','backflip_rebound','--run',str(run),'--seed-start','28500',
          '--source',str(root.parent/'Rob2Nav-upstreams/GO2_backflip')]
 # The exporter validates actual YAML, source hashes and all phase thresholds
 # before writing a policy. No relabeling or backfilling is permitted here.
 (base/'started.json').write_text(json.dumps(dict(command=command,started=time.time()),indent=2))
 with (base/'evaluation.log').open('w') as log:
  child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  record(stage='evaluating',pid=child.pid)
  code=child.wait()
 if code:raise RuntimeError('Evaluator exited with code '+str(code))
 result=json.loads((run/'evaluation.json').read_text())
 assert [e['seed'] for e in result['episodes']]==list(range(28500,28504))
 record(stage='completed',acceptance=result['status'],hardware_qualified=False)
except Exception as exc:
 record(stage='failed',error=str(exc));raise
