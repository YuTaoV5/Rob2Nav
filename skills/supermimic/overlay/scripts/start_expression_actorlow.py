"""One guarded LR-only 32x24 trial; do not inherit the failed WM-transfer flag."""
import json,subprocess,time,hashlib
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1'
OUT=ROOT/'actor_low_lr_trial';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'started_at':time.time()},f)
old=json.loads((ROOT/'warmstart_smoke_process.json').read_text())['command']
assert '--world-model-parent' not in old and '--actor-lr-scale' not in old
assert old[old.index('--iterations')+1]=='24' and old[old.index('--seed')+1]=='36201'
cmd=list(old);cmd[cmd.index('--run-name')+1]='expression_rhythm_actorlow_smoke_v4_20261028'
cmd+=['--actor-lr-scale','.01875']
record={'command':cmd,'baseline_command':old,'single_factor':'actor_lr_scale 1 -> .01875; WM/critic/optimizers fresh in both','training_seed':36201,'created_at':time.time(),'source_sha256':hashlib.sha256((BASE/'scripts/train_expression_supermimic.py').read_bytes()).hexdigest()}
with open(OUT/'training.log','wb') as log:
 p=subprocess.Popen(cmd,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record['pid']=p.pid;(ROOT/'actor_low_lr_smoke_process.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
