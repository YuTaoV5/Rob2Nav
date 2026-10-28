import json,subprocess
from pathlib import Path
out=Path(__file__).parent
rows=json.loads((out/'started.json').read_text())['rows']
state={'phase':'running','training':False,'rows':[]}
for row in rows:
 state['current']=row
 (out/'state.json').write_text(json.dumps(state,indent=2))
 with (out/('prefix'+str(row['prefix'])+'.log')).open('w') as log:
  result=subprocess.run(['/root/gpufree-data/envs/go2-supermimic/bin/python',row['script']],stdout=log,stderr=subprocess.STDOUT)
 report=Path(row['output'])/'state.json'
 state['rows'].append({'config':row,'returncode':result.returncode,'report':json.loads(report.read_text()) if report.exists() else None})
 if result.returncode!=0 or state['rows'][-1]['report'].get('phase')!='completed_pending_review':
  state['phase']='failed_requires_review';break
else:state['phase']='completed_pending_review'
(out/'state.json').write_text(json.dumps(state,indent=2))
