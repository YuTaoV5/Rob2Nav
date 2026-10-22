from pathlib import Path
import json,subprocess,sys
root=Path.cwd();base=root/'outputs/backflip_v8_repeated_switch_20261023';reports=[]
for seed,yaw in [(23010,0),(23011,-15),(23012,15),(23013,30)]:
 out=base/f'seed_{seed}'
 if out.exists():raise RuntimeError('Do not overwrite independent evidence: '+str(out))
 cmd=[sys.executable,'-m','rob2nav.deployment.verify_mujoco','--asset','source','--sequence',str(base/'sequence.json'),'--seed',str(seed),'--initial-yaw-deg',str(yaw),'--max-seconds','300','--output',str(out)]
 if seed==23010:cmd+=['--video']
 with (base/f'seed_{seed}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Verification process failed: '+str(seed))
 d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,yaw_deg=yaw,passed=d['passed'],cycles=len(d['backflip_validation']),all_cycles_pass=all(x['passed'] for x in d['backflip_validation']),summary=str(out/'summary.json')))
 (base/'independent_evaluation.json').write_text(json.dumps(dict(status='evaluating' if len(reports)<4 else 'passed_simulation_only' if all(x['passed'] and x['cycles']==5 and x['all_cycles_pass'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2))
