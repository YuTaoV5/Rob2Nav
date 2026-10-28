"""Verified RoboGauge teacher -> one bounded student trial -> student-only audit."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
for name in ('source','parent','config','teacher-profile','smoke-run','teacher-evidence','output'):
 ap.add_argument('--'+name,type=Path,required=True)
ap.add_argument('--rob2nav',type=Path)
ap.add_argument('--teacher-bc-weight',type=float,default=.1)
ap.add_argument('--run-name',default='velocity_gait_moe_teacher_v1_bounded_20261026')
ap.add_argument('--seed',type=int,default=34011)
ap.add_argument('--eval-seed-start',type=int,default=34020)
ap.add_argument('--asset-id',default='velocity_gait_moe_teacher_v1_model1200')
ap.add_argument('--asset-title',default='MoE教师辅助 · SuperMimic学生第1200轮')
args=ap.parse_args()
source,parent,config,profile,smoke,evidence,output=[p.resolve() for p in (args.source,args.parent,args.config,args.teacher_profile,args.smoke_run,args.teacher_evidence,args.output)]
output.mkdir(parents=True,exist_ok=True)
lock=(output/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record=output/'state.json'
if record.exists():raise SystemExit('Existing teacher trial; inspect it instead of repeating')
state={'pipeline_pid':os.getpid(),'started_unix':time.time(),'capability_qualified':False}
def save(**items):
 state.update(items);tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(record)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def execute(cmd,log_name,phase):
 with (output/log_name).open('w') as log:
  p=subprocess.Popen(cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',MUJOCO_GL='egl'))
  save(phase=phase,active_pid=p.pid,active_command=cmd);status=p.wait()
 if status:raise RuntimeError(f'{phase} failed: {status}; see {log_name}')
try:
 g=json.loads((smoke/'gait_audit.json').read_text());t=json.loads((smoke/'teacher_audit.json').read_text())
 contract=json.loads((smoke/'gait_contract.json').read_text());tc=json.loads((smoke/'teacher_contract.json').read_text())
 parity=json.loads((evidence/'forward_parity.json').read_text());bench=json.loads((evidence/'teacher_strict.json').read_text())
 if not parity['passed'] or parity['batch1_export_max_error']>=1e-5:raise ValueError('Teacher parity failed')
 if not g['training_completed'] or g['shared_transition_reward_max_error']>=1e-5 or not g['command_padding_zero']:raise ValueError('Gait smoke failed')
 if (args.teacher_bc_weight>0 and t['teacher_bc_updates']<=0) or (args.teacher_bc_weight==0 and t['teacher_bc_updates']!=0) or t['teacher_frames']<=0 or t['bc_eligible_frames']<=0 or max(t['teacher_mapping_error'],t['teacher_command_error'],t['teacher_executed_action_error'])>=1e-6:raise ValueError('Teacher sampling/BC smoke failed')
 if tc['bc_weight']!=args.teacher_bc_weight:raise ValueError('BC setting differs from verified smoke')
 if bench['student_actor_used'] or bench['falls_total'] or bench['body_contact_audit']['nonfoot_ground_contact_samples']:raise ValueError('Teacher physics needs diagnosis before use')
 if contract['gait_settings']!=json.loads(config.read_text()) or Path(contract['cfg']['resume']).resolve()!=parent:raise ValueError('Smoke parent/settings mismatch')
 for name,expected in dict(contract['source_sha256'],**tc['source_sha256']).items():
  path=source/'scripts'/name if (source/'scripts'/name).exists() else source/name
  if sha(path)!=expected:raise ValueError(f'Source changed after smoke: {name}')
 spec=json.loads(profile.read_text());policy=profile.parents[2]/spec['policy']
 if sha(policy)!=spec['sha256'] or sha(policy)!=tc['teacher']['sha256']:raise ValueError('Teacher changed')
 save(phase='preflight_passed',parent=str(parent),parent_sha256=sha(parent),teacher_sha256=sha(policy),
      teacher_benchmark_limitations={'velocity_ratio':bench['max_joint_velocity_ratio'],'limit_excess_rad':bench['max_joint_limit_excess_rad'],'target_rate_ratio':bench['max_target_rate_ratio']},
      teacher_quality_filter=tc['bc_quality_filter'],teacher_fraction=.25,teacher_fade=[600,1000],teacher_bc_weight=args.teacher_bc_weight)
 name=args.run_name
 cmd=[sys.executable,'scripts/train_gait_moe.py','--teacher-profile',str(profile),'--teacher-bc-weight',str(args.teacher_bc_weight),'--teacher-bc-lr','0.0001',
      '--gait-config',str(config),'--task','Go2-Velocity-Gait-V2','--driver-policy',str(policy),'--driver-frac','0.25','--driver-fade','600','1000',
      '--num-envs','4096','--iterations','1200','--buffer-len','400','--imag-batch','4096','--seed',str(args.seed),
      '--warmup-iterations','100','--critic-warmup-iterations','50','--actor-ramp-iterations','100',
      '--lr','0.00005','--actor-lr-scale','0.25','--lr-decay-iterations','1200','--lr-final-frac','0.3',
      '--resume',str(parent),'--reset-critic','true','--continue-run','false','--log-interval','50','--save-interval','200','--run-name',name]
 execute(cmd,'formal.log','training')
 runs=[p for p in (source/'logs/supermimic').glob('*'+name) if p.stat().st_mtime>=state['started_unix']]
 if len(runs)!=1:raise ValueError('Missing or ambiguous final run')
 run=runs[0];checkpoint=run/'model_1200.pt';audit=json.loads((run/'teacher_audit.json').read_text())
 if not checkpoint.exists() or (args.teacher_bc_weight>0 and audit['teacher_bc_updates']<=0) or (args.teacher_bc_weight==0 and audit['teacher_bc_updates']!=0) or audit['teacher_executed_action_error']>=1e-6:raise ValueError('Final teacher audit failed')
 save(phase='student_only_evaluation',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),teacher_audit=audit)
 reports=[]
 for seed in range(args.eval_seed_start,args.eval_seed_start+4):
  path=output/f'strict_{seed}.json'
  execute([sys.executable,'scripts/eval_velocity.py','--checkpoint',str(checkpoint),'--seed',str(seed),'--num-envs','128','--stage-seconds','10',
           '--initialize-hold-targets','--audit-body-contact','--output',str(path)],f'strict_{seed}.log',f'student_only_{seed}')
  reports.append(json.loads(path.read_text()))
 execute([sys.executable,'scripts/eval_gait.py','--checkpoint',str(checkpoint),'--seed','33000','--num-envs','8','--stage-seconds','5',
          '--initialize-hold-targets','--output',str(output/'paired_strict.json'),'--gait-output',str(output/'paired_gait.json')],'paired.log','paired_gait')
 stem=args.asset_id
 execute([sys.executable,'scripts/video_velocity_latest.py','--checkpoint',str(checkpoint),'--seed',str(args.eval_seed_start+10),'--stage-seconds','4',
          '--output-dir',str(output/'video'),'--label','MoE-assisted student / 1200','--stem',stem],'video.log','student_only_recording')
 summary={'checkpoint':str(checkpoint),'video':str(output/'video'/(stem+'.mp4')),'independent_seeds':[r['seed'] for r in reports],
          'teacher_during_evaluation':False,'teacher_sha256':sha(policy),'capability_qualified':False}
 (output/'summary.json').write_text(json.dumps(summary,indent=2));save(phase='completed_pending_review',active_pid=None,summary=summary)
 if args.rob2nav:
  execute([sys.executable,'scripts/publish_gait_video.py','--trial',str(output),'--rob2nav',str(args.rob2nav.resolve()),
           '--id',stem,'--title',args.asset_title,'--build'],'publish.log','publishing')
  save(phase='completed_pending_review',active_pid=None,summary=summary,video_published=True)
except Exception as error:
 save(phase='failed',active_pid=None,error=str(error));raise
