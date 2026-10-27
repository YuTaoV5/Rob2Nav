"""One bounded from-scratch crawl student trial and four perturbed teacher-OFF tests."""
from pathlib import Path
import json,subprocess,time,hashlib,os
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_crawl_formal_v1';PY='/root/gpufree-data/envs/go2-supermimic/bin/python'
OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'started_at':time.time()},f)
state={'phase':'preflight','capability_qualified':False,'video_published':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def run(command,log):
 state['command']=command;save()
 with open(OUT/log,'w') as f:
  p=subprocess.Popen(command,cwd=BASE,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);state['child_pid']=p.pid;save();assert p.wait()==0,f'{log} failed'
save()
try:
 evaluator=json.loads((ROOT/'animal_eval_v2_preflight/state.json').read_text());assert evaluator['phase']=='passed' and evaluator['distinct_initial_joint_states']
 smoke=list((BASE/'logs/supermimic').glob('*_animal_crawl_kine_teacher_smoke_v1_20261028'));assert len(smoke)==1
 audit=json.loads((smoke[0]/'animal_audit.json').read_text());assert audit['training_completed'] and all(audit[k]==0 for k in ('reward_max_error','actor_target_history_error','actor_intended_action_history_error','clock_frame_error')) and audit['bc_steps']>0
 old=json.loads((smoke[0]/'animal_contract.json').read_text())['source_sha256']
 assert old['train_animal_supermimic.py']==hashlib.sha256((BASE/'scripts/train_animal_supermimic.py').read_bytes()).hexdigest()
 assert old['rob2nav_animal.py']==hashlib.sha256((BASE/'rob2nav_animal.py').read_bytes()).hexdigest()
 teacher=json.loads((ROOT/'mujoco_teacher_crawl_motion_response.json').read_text());assert any(r>.5 for r in teacher['projected_reference_progress_ratio'])
 state['preflight']={'evaluator_distinct_initial_states':True,'smoke_audit':audit,'teacher_motion_response':teacher,'teacher_acceptance':'functional research teacher under user rule; physical failures and fourth heading failure retained','smoke_functional_failure_retained':True}
 sources=['rob2nav_animal.py','rob2nav_kine_teacher.py','rob2nav_kine_reference.py','scripts/train_animal_supermimic.py','scripts/eval_animal_student_v2.py']
 state['source_sha256']={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in sources}
 run_name='animal_crawl_kine_student_v1_bounded_20261028'
 command=[PY,'-u','scripts/train_animal_supermimic.py','--task','Go2-SuperMimic-Animal-Crawl-V1','--num-envs','4096','--iterations','2000','--buffer-len','400','--imag-batch','4096','--warmup-iterations','300','--critic-warmup-iterations','0','--actor-ramp-iterations','300','--lr','.0001','--actor-lr-scale','.25','--lr-decay','2000','--lr-final-frac','.3','--driver-policy',str(ROOT/'teachers/solo8_crawl_slow/logs/model.pt'),'--driver-frac','.5','--driver-fade','1200','1700','--seed','36711','--run-name',run_name]
 (OUT/'submitted_config.json').write_text(json.dumps({'command':command,'from_scratch':True,'separate_critic_freeze':False,'policy_start':300,'teacher_fade':[1200,1700],'final_student_only_updates':300,'eval_seeds':[36720,36721,36722,36723],'evaluation_joint_perturb_std':.025,'budget_interpretation':'scaled bounded practical trial, not single-factor causal comparison with 32x24'},indent=2))
 state['phase']='training';save();run(command,'training.log')
 logs=list((BASE/'logs/supermimic').glob('*_'+run_name));assert len(logs)==1
 ck=logs[0]/'model_2000.pt';assert ck.exists();actual=json.loads((logs[0]/'animal_audit.json').read_text());assert actual['training_completed'] and all(actual[k]==0 for k in ('reward_max_error','actor_target_history_error','actor_intended_action_history_error','clock_frame_error'))
 state.update(checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),training_audit=actual,phase='evaluating_student_only',evaluations=[]);save()
 for seed in range(36720,36724):
  target=OUT/f'evaluation_{seed}.json'
  run([PY,'-u','scripts/eval_animal_student_v2.py','--checkpoint',str(ck),'--seed',str(seed),'--num-envs','128','--joint-perturb-std','.025','--output',str(target)],f'evaluation_{seed}.log')
  state['evaluations'].append(str(target));save()
 state.update(phase='completed_pending_review',next='Read locomotion and physical outcomes separately; no auto publication or hardware promotion')
except Exception as exc:state.update(phase='failed_requires_review',error=repr(exc))
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
