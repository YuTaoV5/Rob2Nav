"""Synthetic-input actor compatibility only; no training or physics qualification."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
torch.set_num_threads(2);torch.manual_seed(36200)
BASE=Path('/root/gpufree-data/go2-supermimic')
p=next((BASE/'logs/supermimic').glob('*velocity_gait_moe_dataonly_v1_bounded_20261026/model_1200.pt'))
ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['obs_dim']==350
actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).eval();actor.load_state_dict(ck['actor'])
# Observation histories are term-major, NOT five contiguous 70D frames.
offset=0
for name,dim in st.OBS_TERMS:
 if name=='target':break
 offset+=dim*5
assert offset==210 and dim==28
ids=torch.tensor([offset+h*28+j for h in range(5) for j in range(3,9)])
assert float(actor.norm.mean[ids].abs().max())==0,'Old supposedly-unused inputs were not zero; review mapping'
old=torch.randn(128,350)*.1;old[:,ids]=0
new=old.clone()
for h in range(5):
 phase=torch.rand(128)*6.283185307;styles=torch.nn.functional.one_hot(torch.arange(128)%4,4)
 new[:,offset+h*28+3:offset+h*28+9]=torch.cat([styles,phase.sin()[:,None],phase.cos()[:,None]],1)
safe=copy.deepcopy(actor)
with torch.no_grad():
 baseline=actor(old);unsafe=actor(new)
 safe.net[0].weight[:,ids]=0;safe.norm.mean[ids]=0;safe.norm.var[ids]=1
 result=safe(new);err=float((result-baseline).abs().max())
 assert torch.isfinite(result).all() and err<1e-5
report={'parent':str(p),'parent_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'synthetic 128-input inference parity, no real physics, no trained model saved','actor_hidden':list(ck['cfg']['actor_hidden']),'history_layout':'term-major','target_offset':offset,'new_input_columns':ids.tolist(),'old_new_channel_variance_max':float(actor.norm.var[ids].max()),'naive_new_command_max_action_change':float((unsafe-baseline).abs().max()),'safe_initialization_max_action_error':err,'new_channel_normalizer_mean':0,'new_channel_normalizer_variance':1,'new_channel_first_layer_weights':0,'critic_transfer_validated':False,'world_model_transfer_validated':False,'training_started':False,'capability_qualified':False}
out=BASE/'reproduction/animal_expression_v1/warmstart_actor_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if '--save-initialization-only' in sys.argv:
 cfg=dict(ck['cfg']);cfg['task']='Go2-SuperMimic-Expression-Rhythm-V1'
 target=out.parent/'warmstart_actor_initialization_only.pt'
 torch.save({'cfg':cfg,'actor':safe.state_dict(),'obs_dim':350,'initialization_only':True,'new_training_performed':False,'parent_sha256':report['parent_sha256'],'transfer_audit':report},target)
 print('INITIALIZATION_ONLY_NOT_NEW_TRAINING',str(target),flush=True)
