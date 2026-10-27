"""Isolated teacher runtime, preserving the existing SuperMimic environment."""
import json, subprocess, time
from pathlib import Path
root=Path('/root/gpufree-data/go2-supermimic/reproduction/animal_expression_v1')
env=Path('/root/gpufree-data/envs/kine2go-teacher')
state={'phase':'preparing_teacher_runtime','training_started':False,'started_at':time.time()}
def save():
 (root/'runtime_state.json').write_text(json.dumps(state,indent=2))
save()
try:
 if not (env/'bin/python').exists():
  subprocess.run(['/root/gpufree-data/envs/go2-supermimic/bin/python','-m','venv','--system-site-packages',str(env)],check=True,timeout=90)
 cmd=[str(env/'bin/python'),'-m','pip','install','--disable-pip-version-check','--timeout','30','--retries','1','genesis-world==0.3.10','rsl-rl-lib==3.0.0','tyro==0.9.35']
 state['install_command']=cmd;save();subprocess.run(cmd,check=True,timeout=600)
 subprocess.run([str(env/'bin/python'),'-c','import genesis, rsl_rl; import importlib.metadata as m; print(m.version("genesis-world"),m.version("rsl-rl-lib"))'],check=True,timeout=120)
 state['phase']='runtime_ready_pending_native_evaluation'
except Exception as e:state['phase']='runtime_setup_failed';state['error']=repr(e)
state['finished_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
