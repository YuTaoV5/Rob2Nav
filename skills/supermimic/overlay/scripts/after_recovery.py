import json,os,sys,time,subprocess,fcntl,shutil,html
from pathlib import Path
root=Path('/root/gpufree-data/go2-supermimic');trial=root/'reproduction/recovery_teacher_student_v1/formal_trial';out=trial/'publication';out.mkdir(parents=True,exist_ok=True);lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state=out/'state.json'
if state.exists():raise SystemExit('Existing recorder queue, inspect first')
def save(**x):state.write_text(json.dumps(x,indent=2))
save(pid=os.getpid(),phase='waiting_completion')
try:
 for _ in range(360):
  s=json.loads((trial/'state.json').read_text())
  if s['phase']=='failed':raise RuntimeError(s)
  if s['phase']=='completed_pending_review':break
  time.sleep(5)
 else:raise RuntimeError('Wait timed out without restarting training')
 save(pid=os.getpid(),phase='recording');video=trial/'video';env=dict(os.environ,MUJOCO_GL='egl',PATH='/root/gpufree-data/tools/node-v22.14.0-linux-x64/bin:'+os.environ['PATH'])
 with (out/'video.log').open('w') as f:subprocess.run([sys.executable,'scripts/video_recovery.py','--checkpoint',s['checkpoint'],'--output-dir',str(video)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 stem='recovery_student_v1_model4000';site=Path('/root/gpufree-data/Rob2Nav');assets=site/'artifacts/supermimic';assets.mkdir(exist_ok=True);mp4=assets/(stem+'.mp4');shutil.copyfile(video/(stem+'.mp4'),mp4)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(mp4),'-c:v','libvpx-vp9','-threads','2','-crf','32','-b:v','0','-an',str(assets/(stem+'.webm'))],check=True)
 subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(mp4),'-frames:v','1',str(assets/(stem+'.jpg'))],check=True)
 reports=[json.loads((trial/f'eval_{seed}.json').read_text()) for seed in range(35120,35124)];success=sum(x['functional_successes'] for x in reports);total=sum(x['functional_total'] for x in reports);meta=json.loads((video/(stem+'.json')).read_text());meta['functional_successes']=success;meta['functional_total']=total;meta['independent_seeds']=[x['seed'] for x in reports];(assets/(stem+'.json')).write_text(json.dumps(meta,indent=2))
 title='自恢复 · SuperMimic学生第4000轮';detail=f'12秒原速，教师已关闭；独立四种子恢复并稳定2秒 {success}/{total}。关节约束另报，研究候选、未获实机资格。';p=site/'studio/frontend/src/SuperMimicDemos.tsx';text=p.read_text();backup=out/'before';backup.mkdir(exist_ok=True)
 if stem not in text:
  shutil.copyfile(p,backup/p.name);assert 'const demos=[' in text;text=text.replace('const demos=[','const demos=[\n '+json.dumps({'id':stem,'title':title,'detail':detail},ensure_ascii=False)+',',1);p.write_text(text)
 p=site/'studio/frontend/public/supermimic.html';text=p.read_text()
 if stem not in text:
  shutil.copyfile(p,backup/p.name);base='/api/files/artifacts/supermimic/'+stem;card='<article><h2>'+html.escape(title)+'</h2><video controls playsinline preload="metadata" poster="'+base+'.jpg"><source src="'+base+'.webm" type="video/webm"><source src="'+base+'.mp4" type="video/mp4"></video><p>'+html.escape(detail)+'</p><a href="'+base+'.mp4">下载MP4</a></article>';assert '<div class="grid">' in text;p.write_text(text.replace('<div class="grid">','<div class="grid">'+card,1))
 save(pid=os.getpid(),phase='building');
 with (out/'build.log').open('w') as f:subprocess.run(['npm','run','build'],cwd=site/'studio/frontend',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 save(pid=os.getpid(),phase='completed',published=True,functional_successes=success,functional_total=total,capability_qualified=False)
except Exception as e:save(pid=os.getpid(),phase='failed',error=str(e));raise
