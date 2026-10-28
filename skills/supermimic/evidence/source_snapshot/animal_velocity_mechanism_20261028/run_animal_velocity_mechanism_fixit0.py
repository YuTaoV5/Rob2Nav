"""Sequential bounded reward0/reward2 smoke and frozen-parent physical comparison."""
import json,subprocess,hashlib,shutil
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_velocity_mechanism_v1_fixit0';OUT.mkdir(exist_ok=True);PY='/root/gpufree-data/envs/go2-supermimic/bin/python'
parent=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text())['checkpoint'];wm=ROOT/'animal_wm_response_mix_v1/rollouts_plus_response_wm_only.pt';cfg0=json.loads((ROOT/'animal_crawl_resetfix_v1/actual_animal_contract.json').read_text())['cfg']
expected=json.loads((ROOT/'animal_wm_response_mix_v1/state.json').read_text())['arms']['rollouts_plus_response']['artifact_sha256'];assert hashlib.sha256(wm.read_bytes()).hexdigest()==expected
with (OUT/'started.json').open('x') as f:json.dump({'parent_checkpoint':parent,'parent_sha256':hashlib.sha256(Path(parent).read_bytes()).hexdigest(),'wm_initialization':str(wm),'wm_sha256':expected,'scope':'32env32updates each arm; reward weight0 vs2; no formal training','trainer_source_sha256':hashlib.sha256((BASE/'scripts/train_animal_velocity_mechanism.py').read_bytes()).hexdigest()},f,indent=2)
state={'phase':'frozen_parent_evaluation','arms':{}}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def run(argv,log):
 with log.open('w') as f:r=subprocess.run(argv,cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'Exit {r.returncode}, log {log}')
def evaluate(ck,output):run([PY,'scripts/eval_animal_student_v2.py','--checkpoint',str(ck),'--output',str(output),'--seed','37210','--num-envs','4','--joint-perturb-std','.025'],output.with_suffix('.log'))
save()
try:
 old_parent=ROOT/'animal_velocity_mechanism_v1/parent_eval.json'
 for old in old_parent.parent.glob('parent_eval.*'):shutil.copyfile(old,OUT/old.name)
 state['parent_evaluation_reused_from']=str(old_parent);state['parent_evaluation']=json.loads((OUT/'parent_eval.json').read_text());save()
 for weight in [0.,2.]:
  name='reward'+str(int(weight));folder=OUT/name;folder.mkdir(exist_ok=True);state.update(phase='smoke_training',current_arm=name);save()
  cfg=dict(cfg0);cfg.update(num_envs=32,iterations=32,seed=37201,buffer_len=48,warmup_iterations=8,wm_batch=64,wm_updates=1,wm_archive_frac=0.,archive_size=128,imag_batch=32,critic_epochs=2,critic_minibatches=2,critic_warmup_iterations=0,actor_ramp_iterations=8,actor_lr_scale=.075,lr=.0001,lr_decay_iterations=32,log_interval=8,save_interval=16,compile=False,driver_fade=[20,28],actor_initialization=parent,world_model_initialization=str(wm),reference_velocity_weight=weight,reference_velocity_sigma=.25,explicit_critic_calibration_iterations=8,mechanism_output=str(folder/'result.json'),run_name='animal_velocity_mechanism_'+name+'_smoke32_fixit0_20261028')
  argv=[PY,'scripts/train_animal_velocity_mechanism.py']
  for k,v in cfg.items():
   argv.append('--'+k.replace('_','-'));argv.extend(str(x) for x in v) if isinstance(v,(list,tuple)) else argv.append(str(v).lower() if isinstance(v,bool) else str(v))
  (folder/'command.json').write_text(json.dumps({'argv':argv,'cfg':cfg},indent=2));run(argv,folder/'training.log');result=json.loads((folder/'result.json').read_text());state['arms'][name]={'training':result};state['phase']='student_only_evaluation';save();evaluate(result['checkpoint'],folder/'evaluation.json');state['arms'][name]['evaluation']=json.loads((folder/'evaluation.json').read_text());save()
 state['phase']='completed_pending_review'
except Exception as e:state.update(phase='failed_requires_review',error=repr(e))
save();print(json.dumps({'phase':state['phase'],'error':state.get('error')},indent=2))
