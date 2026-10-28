"""One bounded 800-update practical trial after audited smoke; not a causal proof."""
import json,subprocess,hashlib
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_reference_velocity_bounded_v2';OUT.mkdir(exist_ok=True);PY='/root/gpufree-data/envs/go2-supermimic/bin/python'
smoke=ROOT/'animal_velocity_mechanism_v1_fixit0';s=json.loads((smoke/'state.json').read_text());assert s['phase']=='completed_pending_review'
for a in s['arms'].values():
 r=a['training'];assert r['iterations']==32 and r['reward_max_error']==0 and r['critic_calibration_end']['actor_state_max_delta']==0 and r['critic_calibration_end']['actual_bc_steps_so_far']==0 and r['critic_calibration_end']['adam_actor_state_entries']==0 and len(r['actual_actor_steps'])==16
 audit=json.loads((Path(r['log_dir'])/'animal_audit.json').read_text());assert all(audit[k]==0 for k in ['reward_max_error','actor_target_history_error','actor_intended_action_history_error','clock_frame_error'])
cfg=json.loads((smoke/'reward2/command.json').read_text())['cfg'];cfg.update(num_envs=4096,iterations=800,seed=37211,buffer_len=400,warmup_iterations=100,explicit_critic_calibration_iterations=100,wm_batch=2048,wm_updates=2,wm_archive_frac=.25,archive_size=100000,imag_batch=4096,critic_epochs=5,critic_minibatches=4,actor_ramp_iterations=100,lr_decay_iterations=800,log_interval=50,save_interval=100,driver_fade=[400,650],mechanism_output=str(OUT/'result.json'),run_name='animal_reference_velocity_v2_bounded800_20261028')
source=BASE/'scripts/train_animal_velocity_mechanism.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==s['arms']['reward2']['training']['source_sha256']
with (OUT/'started.json').open('x') as f:json.dump({'cfg':cfg,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'parent_actor_sha256':hashlib.sha256(Path(cfg['actor_initialization']).read_bytes()).hexdigest(),'parent_wm_sha256':hashlib.sha256(Path(cfg['world_model_initialization']).read_bytes()).hexdigest(),'smoke':'passed chain only; all short functional runs 0/4','role':'bounded practical reward test; enlarged budget not a single-factor causal experiment'},f,indent=2)
state={'phase':'training','cfg':cfg,'evaluations':[]}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def run(argv,log):
 with log.open('w') as f:r=subprocess.run(argv,cwd=BASE,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'Exit {r.returncode}; log {log}')
argv=[PY,'scripts/train_animal_velocity_mechanism.py']
for k,v in cfg.items():
 argv.append('--'+k.replace('_','-'));argv.extend(str(x) for x in v) if isinstance(v,(tuple,list)) else argv.append(str(v).lower() if isinstance(v,bool) else str(v))
(OUT/'command.json').write_text(json.dumps({'argv':argv},indent=2));save()
try:
 run(argv,OUT/'training.log');r=json.loads((OUT/'result.json').read_text());state.update(phase='evaluating_student_only',checkpoint=r['checkpoint']);save()
 for seed in [37220,37221,37222,37223]:
  p=OUT/f'evaluation_{seed}.json';run([PY,'scripts/eval_animal_student_v2.py','--checkpoint',r['checkpoint'],'--output',str(p),'--seed',str(seed),'--num-envs','128','--joint-perturb-std','.025'],p.with_suffix('.log'));state['evaluations'].append(str(p));save()
 state['phase']='completed_pending_review'
except Exception as e:state.update(phase='failed_requires_review',error=repr(e))
save();print(json.dumps({'phase':state['phase'],'error':state.get('error')},indent=2))
