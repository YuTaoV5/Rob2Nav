"""Matched offline WM-only ablation: unconstrained vs explicit one-frame input delay."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_delay_structure_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'seed':36811,'updates_per_arm':100,'train_prefixes':[40,120],'heldout_prefix':240,'policy_updated':False,'physics_executed':False},f)
torch.manual_seed(36811);torch.set_float32_matmul_precision('highest')
f=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(f['checkpoint'],map_location='cuda',weights_only=False)
parent=WorldModel(tuple(ck['cfg']['wm_hidden'])).cuda().eval();parent.load_state_dict(ck['world_model']);names=['animal_action_response_audit','animal_action_response_prefix120_seed36726','animal_action_response_prefix240_seed36727'];datasets=[];sha={}
for n in names:
 p=ROOT/n/'branches.pt';sha[n]=hashlib.sha256(p.read_bytes()).hexdigest();datasets.append(torch.load(p,map_location='cuda',weights_only=False))
state={'phase':'running','policy_updated':False,'physics_executed':False,'arms':{}}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def predict(wm,d,delayed):
 r=d['records'][2:];b=d['records'][0];x={k:torch.cat([v[k] for v in r]) for k in ['s0','a0','a_prev','xi','relief']};n=b['a0'].shape[0];a2=b['a0'].repeat(len(r),1);neutral=wm.in_norm.mean[st.STATE_DIM:st.STATE_DIM+12][None].expand_as(x['a0'])
 s1,_=wm(x['s0'],neutral if delayed else x['a0'],x['a_prev'],x['xi'],x['relief']);s2,_=wm(s1,neutral if delayed else a2,x['a0'],x['xi'],x['relief'])
 v1=s1[:,st.V_YAW][:,:2].reshape(12,2,n,2);v2=s2[:,st.V_YAW][:,:2].reshape(12,2,n,2);y1=torch.cat([v['s1'] for v in r]);y2=torch.cat([v['s2'] for v in r]);scale=wm.out_norm.var[:st.STATE_DIM].sqrt()+1e-6
 error=((s1-y1)/scale).square().mean()+((s2-y2)/scale).square().mean()
 return (v1[:,0]-v1[:,1])/.1,(v2[:,0]-v2[:,1])/.1,error
for delayed in [False,True]:
 arm='explicit_delayed_input' if delayed else 'unconstrained_input';wm=copy.deepcopy(parent);opt=torch.optim.Adam(wm.parameters(),lr=1e-5);rows=[];state['current_arm']=arm;state['arms'][arm]={'history':rows};save()
 for step in range(101):
  if step in [0,20,100]:
   metrics=[]
   with torch.no_grad():
    for d in datasets:
     j1,j2,err=predict(wm,d,delayed);truth=d['actual_two_step'];metrics.append({'two_step_relative_error':float((j2-truth).norm()/truth.norm()),'two_step_cosine':float(torch.nn.functional.cosine_similarity(j2.flatten(),truth.flatten(),dim=0)),'one_step_response_norm':float(j1.norm()),'true_one_step_response_norm':float(d['actual_one_step'].norm()),'normalized_two_transition_state_mse':float(err)})
   rows.append({'updates':step,'train40':metrics[0],'train120':metrics[1],'heldout240':metrics[2]});save()
  if step==100:break
  loss=0
  for d in datasets[:2]:
   j1,j2,err=predict(wm,d,delayed);truth=d['actual_two_step'];denom=truth.square().mean().clamp(min=1e-8);loss=loss+err+(j2-truth).square().mean()/denom+(j1-d['actual_one_step']).square().mean()/denom
  opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(wm.parameters(),10);assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
 artifact=OUT/(arm+'_wm_only.pt');torch.save({'world_model':wm.state_dict(),'optimizer':opt.state_dict(),'explicit_delayed_input':delayed},artifact);state['arms'][arm]['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();save()
state.update(phase='completed_pending_review',updates_per_arm=100,learning_rate=1e-5,seed=36811,branch_sha256=sha,parent_sha256=hashlib.sha256(Path(f['checkpoint']).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),deployed=False,controller_qualified=False,limitations='Matched objective and budget within this pair only. Changed objective relative to prior calibration100. Three local prefix/seed datasets, not global calibration or functional validation. Explicit arm holds current action slot at fixed parent running mean; previous-action slot carries effective control. Actor and original checkpoint unchanged. WM artifacts server only.')
save();print(json.dumps(state,indent=2))
