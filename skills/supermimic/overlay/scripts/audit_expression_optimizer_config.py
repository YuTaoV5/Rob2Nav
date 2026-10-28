"""Read actual saved configuration/optimizer LR; no updates or model conversion."""
import json,hashlib
from pathlib import Path
import torch
BASE=Path('/root/gpufree-data/go2-supermimic');rows=[]
paths=[('parent','2026-10-26_02-47-27_velocity_gait_moe_dataonly_v1_bounded_20261026/model_1200.pt'),('actor_only','2026-10-28_01-18-47_expression_rhythm_actorinit_smoke_v2_20261028/model_24.pt'),('actor_wm','2026-10-28_01-47-47_expression_rhythm_actor_wm_smoke_v3_20261028/model_24.pt')]
for label,name in paths:
 p=BASE/'logs/supermimic'/name;d=torch.load(p,map_location='cpu',weights_only=False)
 rows.append({'role':label,'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cfg':{k:d['cfg'].get(k) for k in ('task','lr','actor_lr_scale','critic_warmup_iterations','actor_ramp_iterations','seed','num_envs','iterations')},'saved_actor_lr':d['opt_actor']['param_groups'][0]['lr']})
report={'models':rows,'actor_only_to_parent_final_lr_ratio':rows[1]['saved_actor_lr']/rows[0]['saved_actor_lr'],'training_started':False,'causal_mechanism_proven':False}
out=BASE/'reproduction/animal_expression_v1/optimizer_lr_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
