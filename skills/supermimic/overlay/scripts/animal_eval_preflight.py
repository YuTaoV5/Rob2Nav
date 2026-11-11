"""Sequential checks of the evaluator; locomotion success is not a smoke gate."""
from pathlib import Path
import json,subprocess,time
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_eval_v2_preflight';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'started_at':time.time()},f)
state={'phase':'running','reports':[]}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
try:
 paths=list((BASE/'logs/supermimic').glob('*_animal_crawl_kine_teacher_smoke_v1_20261028/model_24.pt'));assert len(paths)==1
 for seed in (36712,36713):
  output=OUT/f'evaluation_{seed}.json';cmd=['/root/gpufree-data/envs/go2-supermimic/bin/python','-u','scripts/eval_animal_student_v2.py','--checkpoint',str(paths[0]),'--seed',str(seed),'--num-envs','4','--joint-perturb-std','.025','--output',str(output)]
  with open(OUT/f'evaluation_{seed}.log','w') as log:subprocess.run(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
  report=json.loads(output.read_text());assert report['physical_samples']==report['expected_physical_samples']==1920 and report['num_envs']==4 and not report['body_contact']['contact_buffer_overflow'];state['reports'].append(str(output));save()
 reports=[json.loads(Path(p).read_text()) for p in state['reports']];state['distinct_initial_joint_states']=reports[0]['initial_joint_state_sha256']!=reports[1]['initial_joint_state_sha256'];assert state['distinct_initial_joint_states']
 state['phase']='passed';state['locomotion_pass_counts']=[d['locomotion_passed'] for d in reports];state['meaning']='evaluator and seeded initial perturbation verified, not action qualification'
except Exception as e:state.update(phase='failed_requires_review',error=repr(e))
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
