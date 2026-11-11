"""Eight sequential native closed-loop teacher probes; no student training."""
import os,json,subprocess,time,hashlib
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1'
OUT=ROOT/'native_independent';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'started_at':time.time()},f)
state={'phase':'screening','student_training_started':False,'trials':[]}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
try:
 for clip in ('ai4_dog_trot_00','solo8_crawl_slow'):
  for seed in range(36400,36404):
   dest=OUT/clip/str(seed);dest.mkdir(parents=True,exist_ok=True)
   env=dict(os.environ,KINE_SCREEN_SEED=str(seed),KINE_SCREEN_OUTPUT=str(dest))
   cmd=['/root/gpufree-data/envs/kine2go-teacher/bin/python','-u',str(ROOT/'eval_kine_independent.py'),clip]
   state['current']={'clip':clip,'seed':seed,'command':cmd};save()
   with open(dest/'evaluation.log','w') as log:subprocess.run(cmd,cwd='/root/gpufree-data/skill_teachers/kine2go',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
   row=json.loads((dest/('native_screen_'+clip+'.json')).read_text())['trials'][0]
   state['trials'].append(row);save()
 state['phase']='completed_pending_review'
except Exception as e:state['phase']='failed_requires_review';state['error']=repr(e)
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
