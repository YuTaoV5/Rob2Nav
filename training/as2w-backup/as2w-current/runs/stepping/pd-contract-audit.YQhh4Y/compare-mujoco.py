import json, statistics, collections
from pathlib import Path
out = Path(__file__).parent
base = out.parent/'support-aware-reward.bp4984'/'support-aware-mujoco-holdout-1199'
alt = out/'mujoco-85-32'
a,b = [json.loads((p/'contract.json').read_text()) for p in (base,alt)]
assert a['checkpoint_sha256']==b['checkpoint_sha256']
r={}
for course in ('up','down'):
    rows = [json.loads((p/'mujoco-stepping-diagnostic.json').read_text())[course]['rows'] for p in (base,alt)]
    seeds = {x['seed'] for x in rows[1]}
    rows[0] = [x for x in rows[0] if x['seed'] in seeds]
    assert [x['seed'] for x in rows[0]] == [x['seed'] for x in rows[1]]
    r[course] = {label:dict(pass_count=sum(x['passed'] for x in group),count=len(group),
          failures=dict(collections.Counter(str(x['failure']) for x in group)),
          median_progress_m=statistics.median(x['forward_progress_m'] for x in group))
          for label,group in zip(('55_2','85_3.2'),rows)}
(out/'comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(r)
