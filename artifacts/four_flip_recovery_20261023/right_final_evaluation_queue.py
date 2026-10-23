"""Finish the already-running right-flip job with its reserved independent evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path.cwd();sys.path.insert(0,str(root))
from studio.jobs import Jobs
jobs=Jobs(root);key='e4baa3665ca04e28';run=root/'outputs/studio/runs'/key
record=root/'artifacts/four_flip_recovery_20261023/right_final_evaluation_state.json'
while True:
    job=jobs.get(key)
    if job['state'] in ('succeeded','failed','canceled','unknown'):break
    time.sleep(5)
if job['state']!='succeeded':
    record.write_text(json.dumps(dict(status='not_evaluated',reason='Training not successful',training_state=job['state']),indent=2))
    raise SystemExit(1)
cfg=job['config'];options=cfg['recipe_options']
assert cfg['recipe_id']=='sideflip_native_ppo' and options['sideflip_direction']=='right'
assert not cfg['auto_evaluate'], 'Worker now owns evaluation; do not duplicate'
assert (run/'completion.json').is_file(), 'No completed run, never evaluate a partial checkpoint'
motion=root/options['motion']
cmd=['/root/gpufree-data/envs/go2-mapless/bin/python','scripts/evaluate_fixstand_run.py',
    '--kind','sideflip','--run',str(run),'--seed-start','20200',
    '--source',str(root.parent/'Rob2Nav-upstreams/GO2_backflip'),'--motion',str(motion)]
previous=run/'evaluation.json'
if previous.exists():
    existing=json.loads(previous.read_text())
    if existing.get('status') not in ('not_accepted','passed_simulation_only'):
        raise RuntimeError('Existing partial evaluation; inspect its real process before any recovery')
else:
    record.write_text(json.dumps(dict(status='evaluating',job=key,seeds=list(range(20200,20204)),command=cmd),indent=2))
    with (run/'independent_evaluation_20200.log').open('w') as log:
        subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),check=True)
result=json.loads(previous.read_text())
assert [r['seed'] for r in result['episodes']]==list(range(20200,20204))
record.write_text(json.dumps(dict(status=result['status'],job=key,episodes=result['episodes'],
    hardware_qualified=False,requires_video_review_and_unified_switch=True),indent=2))
