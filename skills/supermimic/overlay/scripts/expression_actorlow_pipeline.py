"""Single guarded, bounded expression student trial, followed by teacher-OFF eval."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic')
ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'expression_actorlow_formal'
OUT.mkdir(exist_ok=True);(OUT/'started.json').open('x').write(json.dumps({'started_at':time.time(),'pid':__import__('os').getpid()}))
PY='/root/gpufree-data/envs/go2-supermimic/bin/python'
state={'phase':'preflight','capability_qualified':False,'video_published':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def run(cmd,log):
 state['command']=cmd;save()
 with open(OUT/log,'w') as f:
  subprocess.run(cmd,cwd=BASE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
save()
try:
 teacher=json.loads((ROOT/'expression_teacher_eval.json').read_text());assert teacher['functional_candidate']
 smoke=list((BASE/'logs/supermimic').glob('*_expression_rhythm_actorlow_smoke_v4_20261028/expression_audit.json'));assert len(smoke)==1
 audit=json.loads(smoke[0].read_text());assert audit['training_completed'] and audit['reward_max_error']==0 and audit['clock_command_error']==0
 pair=json.loads((ROOT/'actor_low_lr_pair/state.json').read_text());assert pair['phase']=='completed_pending_review' and pair['observed_initial_fields_equal']
 report=json.loads((ROOT/'actor_low_lr_pair/actor_low.json').read_text())
 baseline=json.loads((ROOT/'actor_low_lr_smoke_process.json').read_text())['command'];assert '--world-model-parent' not in baseline
 assert baseline[baseline.index('--actor-lr-scale')+1]=='.01875'
 contract=json.loads((smoke[0].parent/'expression_contract.json').read_text());assert contract['initialization']['world_model']=='from_scratch' and contract['initialization']['optimizers']=='fresh'
 sources=[BASE/'rob2nav_expression.py',BASE/'scripts/train_expression_supermimic.py',BASE/'scripts/eval_expression.py',BASE/'rob2nav_moe_teacher.py']
 state['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 state['preflight']={'teacher_functional':True,'teacher_physical_limits_passed':False,'teacher_emotion_perception_validated':False,'smoke_audit':audit,'smoke_path':str(smoke[0])}
 run_name='expression_rhythm_actorlow_v4_bounded_20261028'
 cmd=list(baseline)
 changes={'--num-envs':'4096','--iterations':'800','--buffer-len':'400','--imag-batch':'4096','--warmup-iterations':'200','--critic-warmup-iterations':'100','--actor-ramp-iterations':'100','--seed':'36211','--run-name':run_name}
 for k,v in changes.items():cmd[cmd.index(k)+1]=v
 j=cmd.index('--driver-fade');cmd[j+1:j+3]=['400','650']
 cmd+=['--lr','.0002','--lr-decay','800','--lr-final-frac','.3']
 (OUT/'submitted_config.json').write_text(json.dumps({'command':cmd,'actor_initialization_only':True,'world_model_critic_optimizers_fresh':True,'teacher_fade':[400,650],'final_student_only_updates':150,'interpretation':'scaled practical bounded trial; not a single-factor comparison with 32x24','styles':'rhythmic choreography, not verified human-perceived emotions'},indent=2))
 state['phase']='training';save();run(cmd,'training.log')
 logs=list((BASE/'logs/supermimic').glob('*_'+run_name));assert len(logs)==1
 ck=logs[0]/'model_800.pt';assert ck.exists()
 actual=json.loads((logs[0]/'expression_audit.json').read_text());assert actual['training_completed'] and actual['reward_max_error']==0 and actual['clock_command_error']==0
 state['checkpoint']=str(ck);state['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest();state['training_audit']=actual
 state['phase']='evaluating_student_only';save();state['evaluations']=[]
 for seed in range(36220,36224):
  target=OUT/f'evaluation_{seed}.json'
  run([PY,'-u','scripts/eval_expression.py','--checkpoint',str(ck),'--seed',str(seed),'--num-envs','128','--output',str(target)],f'evaluation_{seed}.log')
  state['evaluations'].append(str(target));save()
 state['phase']='completed_pending_review';state['next']='Review functional/physical outcomes and record uncut original-speed student video before publishing.'
except Exception as e:
 state['phase']='failed_requires_review';state['error']=repr(e)
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
