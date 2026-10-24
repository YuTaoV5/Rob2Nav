from pathlib import Path
import yaml,json
from rob2nav.export_actor import ConfigLoader
root=Path(__file__).resolve().parents[2]; runs=root/'outputs/studio/runs'
result={}
for key in ('babdb90dceb74a46','b70bd35d97bc4e23'):
 env=yaml.load((runs/key/'params/env.yaml').read_text(),Loader=ConfigLoader)
 result[key]={k:env.get(k) for k in ('episode_length_s','entry_delay_seconds','entry_hold_scale','rebound','rebound_implementation_sha256','reward_scale_overrides')}
 result[key]['completion']=json.loads((runs/key/'completion.json').read_text())
 if key.startswith('bab'):
  assert env['episode_length_s']==9. and env['entry_delay_seconds']==1. and env['entry_hold_scale']==.15
  assert env['rebound']['cycles']==3 and env['rebound']['max_rebound_seconds']==.45
  result[key]['contract']=json.loads((runs/key/'fixstand.json').read_text())
 else:
  scales=env['rewards']['scales']; assert scales['lin_vel_z']==60 and scales['ang_vel_x']==-1
  assert env['episode_length_s']==5. and env['entry_delay_seconds']==1. and env['entry_hold_scale']==.1
  result[key]['scales']={k:scales[k] for k in ('lin_vel_z','ang_vel_x')}
result['left_evaluation']=json.loads((runs/'527c9e2a8c624243/evaluation.json').read_text())
result['left_summaries']={str(s):json.loads((runs/'527c9e2a8c624243'/('seed_'+str(s))/'summary.json').read_text()) for s in range(28200,28204)}
Path(__file__).with_name('actual_smoke_audit.json').write_text(json.dumps(result,indent=2))
print('Actual smoke settings verified; old rebound run lacks code hash, not backfilled. Left status:',result['left_evaluation']['status'])
