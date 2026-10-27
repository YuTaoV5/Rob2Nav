"""Single guarded, bounded expression student trial, followed by teacher-OFF eval."""
import json,subprocess,sys,time,hashlib
from pathlib import Path
BASE=Path('/root/gpufree-data/go2-supermimic')
ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'expression_formal'
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
 smoke=list((BASE/'logs/supermimic').glob('*_expression_rhythm_moe_smoke_v1_20261028/expression_audit.json'));assert len(smoke)==1
 audit=json.loads(smoke[0].read_text());assert audit['training_completed'] and audit['reward_max_error']==0 and audit['clock_command_error']==0 and audit['teacher_frames']>0 and audit['bc_steps']>0
 sources=[BASE/'rob2nav_expression.py',BASE/'scripts/train_expression_supermimic.py',BASE/'scripts/eval_expression.py',BASE/'rob2nav_moe_teacher.py']
 state['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
 state['preflight']={'teacher_functional':True,'teacher_physical_limits_passed':False,'teacher_emotion_perception_validated':False,'smoke_audit':audit,'smoke_path':str(smoke[0])}
 run_name='expression_rhythm_moe_student_v1_bounded_20261028'
 cmd=[PY,'-u','scripts/train_expression_supermimic.py','--task','Go2-SuperMimic-Expression-Rhythm-V1','--num-envs','4096','--iterations','2000','--buffer-len','400','--imag-batch','4096','--warmup-iterations','300','--critic-warmup-iterations','150','--actor-ramp-iterations','300','--lr','.0001','--actor-lr-scale','.25','--lr-decay','2000','--lr-final-frac','.3','--driver-policy','/root/gpufree-data/Rob2Nav/vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt','--driver-frac','.5','--driver-fade','1200','1700','--seed','36111','--run-name',run_name]
 (OUT/'submitted_config.json').write_text(json.dumps({'command':cmd,'from_scratch':True,'teacher_fade':[1200,1700],'final_student_only_updates':300,'styles':'rhythmic choreography, not verified human-perceived emotions'},indent=2))
 state['phase']='training';save();run(cmd,'training.log')
 logs=list((BASE/'logs/supermimic').glob('*_'+run_name));assert len(logs)==1
 ck=logs[0]/'model_2000.pt';assert ck.exists()
 actual=json.loads((logs[0]/'expression_audit.json').read_text());assert actual['training_completed'] and actual['reward_max_error']==0 and actual['clock_command_error']==0
 state['checkpoint']=str(ck);state['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest();state['training_audit']=actual
 state['phase']='evaluating_student_only';save();state['evaluations']=[]
 for seed in range(36120,36124):
  target=OUT/f'evaluation_{seed}.json'
  run([PY,'-u','scripts/eval_expression.py','--checkpoint',str(ck),'--seed',str(seed),'--num-envs','128','--output',str(target)],f'evaluation_{seed}.log')
  state['evaluations'].append(str(target));save()
 state['phase']='completed_pending_review';state['next']='Review functional/physical outcomes and record uncut original-speed student video before publishing.'
except Exception as e:
 state['phase']='failed_requires_review';state['error']=repr(e)
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
