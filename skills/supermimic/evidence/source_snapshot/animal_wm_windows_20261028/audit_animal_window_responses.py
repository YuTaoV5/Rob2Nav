"""Frozen action finite differences after episode-window WM fit; no tuning."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
ROOT=Path(__file__).resolve().parents[1]/'reproduction/animal_expression_v1';OUT=ROOT/'animal_wm_windows_v1';path=OUT/'response_audit.json'
if path.exists():raise RuntimeError('Existing report; do not overwrite')
torch.set_float32_matmul_precision('highest');ck=torch.load(json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text())['checkpoint'],map_location='cuda',weights_only=False);rows=[]
for arm in ['unconstrained','delayed_input']:
 artifact=OUT/(arm+'_wm_only.pt');new=torch.load(artifact,map_location='cuda',weights_only=False);wm=WorldModel(tuple(ck['cfg']['wm_hidden'])).cuda().eval();wm.load_state_dict(new['world_model']);wm.requires_grad_(False)
 for name in ['animal_action_response_audit','animal_action_response_prefix120_seed36726','animal_action_response_prefix240_seed36727']:
  d=torch.load(ROOT/name/'branches.pt',map_location='cuda',weights_only=False);b=d['records'][0];a=b['a0'];neutral=wm.in_norm.mean[st.STATE_DIM:st.STATE_DIM+12][None].expand_as(a)
  with torch.no_grad():
   def pred(x):
    s1,_=wm(b['s0'],neutral if arm=='delayed_input' else x,b['a_prev'],b['xi'],b['relief']);s2,_=wm(s1,neutral if arm=='delayed_input' else a,x,b['xi'],b['relief']);return s1[:,st.V_YAW][:,:2],s2[:,st.V_YAW][:,:2]
   j1=[];j2=[]
   for j in range(12):
    delta=torch.zeros_like(a);delta[:,j]=.05;p=pred(a+delta);m=pred(a-delta);j1.append((p[0]-m[0])/.1);j2.append((p[1]-m[1])/.1)
   one=torch.stack(j1);two=torch.stack(j2);truth=d['actual_two_step'];rows.append({'arm':arm,'dataset':name,'one_step_response_norm':float(one.norm()),'two_step_relative_error':float((two-truth).norm()/truth.norm()),'two_step_cosine':float(torch.nn.functional.cosine_similarity(two.flatten(),truth.flatten(),dim=0)),'wm_artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()})
report={'policy_updated':False,'physics_executed':False,'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':'Fixed baseline state finite difference; physical branches retain prior recorded start mismatches/noise. Three local counterfactual response sets only, not a new functional trial. No further fitting or threshold changes.'};path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
