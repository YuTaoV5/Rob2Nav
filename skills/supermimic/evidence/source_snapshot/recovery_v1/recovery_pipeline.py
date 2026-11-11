import json,sys,subprocess,os,time,fcntl,hashlib
from pathlib import Path
root=Path('/root/gpufree-data/go2-supermimic');out=root/'reproduction/recovery_teacher_student_v1/formal_trial';out.mkdir(parents=True,exist_ok=True);lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=out/'state.json'
if state.exists():raise SystemExit('Existing state, do not duplicate')
def save(**d):state.write_text(json.dumps(d,indent=2))
smoke=root/'logs/supermimic/2026-10-26_22-44-40_recovery_fr_teacher_smoke_v1';a=json.loads((smoke/'recovery_audit.json').read_text());assert a['reward_max_error']==0 and a['teacher_frames']>0 and a['bc_steps']>0
teacher='/root/gpufree-data/Rob2Nav-upstreams/FR-Net/logs/GO2_recovery/pretrained_recovery_v3/model_2000.pt';name='recovery_fr_teacher_student_v1_bounded_20261026';start=time.time()
cmd=[sys.executable,'-u','scripts/train_recovery_supermimic.py','--task','Go2-SuperMimic-Recovery-V1','--num-envs','4096','--iterations','4000','--buffer-len','400','--imag-batch','4096','--warmup-iterations','400','--critic-warmup-iterations','200','--actor-ramp-iterations','400','--driver-policy',teacher,'--driver-frac','.5','--driver-fade','2500','3500','--seed','35111','--lr','0.0001','--actor-lr-scale','.25','--lr-decay-iterations','4000','--lr-final-frac','.3','--run-name',name,'--log-interval','100','--save-interval','500']
def execute(cmd,log,phase):
 with (out/log).open('w') as f:p=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,MUJOCO_GL='egl'),stdout=f,stderr=subprocess.STDOUT);save(pid=os.getpid(),phase=phase,active_pid=p.pid,command=cmd,capability_qualified=False);rc=p.wait()
 if rc:raise RuntimeError(phase+' failed '+str(rc))
try:
 execute(cmd,'formal.log','training');runs=[p for p in (root/'logs/supermimic').glob('*'+name) if p.stat().st_mtime>=start];assert len(runs)==1;checkpoint=runs[0]/'model_4000.pt';assert checkpoint.exists();a=json.loads((runs[0]/'recovery_audit.json').read_text());assert a['reward_max_error']==0
 for seed in range(35120,35124):execute([sys.executable,'scripts/eval_recovery_supermimic.py','--checkpoint',str(checkpoint),'--seed',str(seed),'--num-envs','128','--output',str(out/f'eval_{seed}.json')],f'eval_{seed}.log',f'evaluation_{seed}')
 save(pid=os.getpid(),phase='completed_pending_review',checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),capability_qualified=False)
except Exception as e:save(pid=os.getpid(),phase='failed',error=str(e),capability_qualified=False);raise
