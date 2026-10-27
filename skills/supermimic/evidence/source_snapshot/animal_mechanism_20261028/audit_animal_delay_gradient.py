"""Offline derivative comparison on saved physics branches; no new rollout/update."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
ROOT=Path(__file__).resolve().parents[1]/'reproduction/animal_expression_v1';folder=ROOT/'animal_action_response_audit'
data=torch.load(folder/'branches.pt',map_location='cpu',weights_only=False);s=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(s['checkpoint'],map_location='cuda',weights_only=False)
torch.set_float32_matmul_precision('highest');wm=WorldModel(tuple(ck['cfg']['wm_hidden'])).cuda().eval();wm.load_state_dict(ck['world_model']);wm.requires_grad_(False)
b=data['records'][0];t={k:b[k].cuda() for k in ('s0','a0','a_prev','xi','relief')};truth=data['actual_two_step'].cuda();rows=[]
for masked in (False,True):
 a=t['a0'].clone().requires_grad_(True)
 s1,_=wm(t['s0'],a.detach() if masked else a,t['a_prev'],t['xi'],t['relief']);s2,_=wm(s1,t['a0'],a,t['xi'],t['relief'])
 jac=torch.stack([torch.autograd.grad(s2[:,st.V_YAW][:,j].sum(),a,retain_graph=True)[0] for j in range(2)],-1).permute(1,0,2)
 rows.append({'mask_current_action_gradient_in_first_transition':masked,'response_cosine':float(torch.nn.functional.cosine_similarity(truth.flatten(),jac.flatten(),dim=0)),'relative_response_error':float((truth-jac).norm()/truth.norm().clamp(min=1e-8)),'predicted_response_norm':float(jac.norm()),'actual_response_norm':float(truth.norm())})
out={'training_started':False,'physics_executed':False,'source_branch_artifact_sha256':hashlib.sha256((folder/'branches.pt').read_bytes()).hexdigest(),'rows':rows,'limitations':'One 40-step-prefix local audit with recorded repeat/prefix differences, not global calibration or proof of improved learning; no original model or policy modified.'}
(ROOT/'animal_delay_gradient_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
