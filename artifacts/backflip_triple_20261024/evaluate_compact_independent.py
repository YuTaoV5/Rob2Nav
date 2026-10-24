import os,json,pathlib,subprocess
p=pathlib.Path('outputs/backflip_triple_20261024')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
results=[]
for seed,yaw in zip(range(28040,28044),[0,90,180,-90]):
 out=p/f'compact_independent_{seed}'
 if out.exists():raise RuntimeError('Output exists: inspect, do not restart')
 cmd=['/root/gpufree-data/envs/go2-mapless/bin/python',str(p/'compact_independent_probe.py'),str(seed),str(yaw)]
 (p/'compact_independent_state.json').write_text(json.dumps({'seed':seed,'yaw':yaw,'stage':'running','results':results}))
 with open(p/f'compact_independent_{seed}.log','w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
 f=out/'summary.json';d=json.loads(f.read_text()) if f.exists() else {}
 results.append({'seed':seed,'yaw':yaw,'exit':r.returncode,'passed':d.get('passed',False),'flips':len(d.get('backflip_validation',[])),'summary':str(f)})
 (p/'compact_independent_state.json').write_text(json.dumps({'stage':'between_runs','results':results}))
 if r.returncode or not f.exists():raise RuntimeError('Execution failed: inspect before restarting')
(p/'compact_independent_state.json').write_text(json.dumps({'stage':'complete','passed':all(x['passed'] and x['flips']==3 for x in results),'results':results},indent=2))
