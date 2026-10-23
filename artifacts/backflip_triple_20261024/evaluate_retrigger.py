import os,json,pathlib,subprocess
p=pathlib.Path('outputs/backflip_triple_20261024')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
results=[]
for seed,yaw in zip(range(28030,28034),[0,90,180,-90]):
 out=p/f'retrigger_independent_{seed}'
 if out.exists():raise RuntimeError('Output exists: inspect, do not restart')
 cmd=['/root/gpufree-data/envs/go2-mapless/bin/python','-m','rob2nav.deployment.verify_mujoco','--asset','source','--sequence',str(p/'sequence.json'),'--seed',str(seed),'--initial-yaw-deg',str(yaw),'--max-seconds','150','--output',str(out)]
 (p/'retrigger_independent_state.json').write_text(json.dumps({'seed':seed,'yaw':yaw,'stage':'running','results':results}))
 if seed==28033:cmd += ['--video','--video-fps','50']
 with open(p/f'retrigger_independent_{seed}.log','w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 f=out/'summary.json';d=json.loads(f.read_text()) if f.exists() else {}
 results.append({'seed':seed,'yaw':yaw,'exit':r.returncode,'passed':d.get('passed',False),'flips':len(d.get('backflip_validation',[])),'summary':str(f)})
 (p/'retrigger_independent_state.json').write_text(json.dumps({'stage':'between_runs','results':results}))
 if r.returncode or not f.exists():raise RuntimeError('Execution failed: inspect before restarting')
(p/'retrigger_independent_state.json').write_text(json.dumps({'stage':'complete','passed':all(x['passed'] and x['flips']==3 for x in results),'results':results},indent=2))
