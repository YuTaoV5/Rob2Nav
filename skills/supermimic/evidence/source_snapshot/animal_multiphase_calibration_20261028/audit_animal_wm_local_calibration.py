"""Offline local WM-only fit; prefix240 held out; never a qualified controller."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_wm_local_calibration_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'training':'offline WM only, 100 Adam updates, no policy or physics','split':'prefix40+120 train, prefix240 held out'},f)
torch.manual_seed(36801);torch.set_float32_matmul_precision('highest')
final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(final['checkpoint'],map_location='cuda',weights_only=False)
wm=WorldModel(tuple(ck['cfg']['wm_hidden'])).cuda().eval();wm.load_state_dict(ck['world_model']);teacher=copy.deepcopy(wm).requires_grad_(False)
folders=['animal_action_response_audit','animal_action_response_prefix120_seed36726','animal_action_response_prefix240_seed36727'];data=[];hashes={}
for name in folders:
 p=ROOT/name/'branches.pt';hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();d=torch.load(p,map_location='cuda',weights_only=False);data.append(d)
def prediction(net,d):
 rows=d['records'];base=rows[0];r=rows[2:];keys=['s0','a0','a_prev','xi','relief'];x={k:torch.cat([z[k] for z in r]) for k in keys};n=base['a0'].shape[0]
 s1,_=net(x['s0'],x['a0'],x['a_prev'],x['xi'],x['relief']);a2=base['a0'].repeat(len(r),1);s2,_=net(s1,a2,x['a0'],x['xi'],x['relief']);v1=s1[:,st.V_YAW][:,:2].reshape(12,2,n,2);v2=s2[:,st.V_YAW][:,:2].reshape(12,2,n,2)
 return s1,s2,(v1[:,0]-v1[:,1])/.1,(v2[:,0]-v2[:,1])/.1
with torch.no_grad():anchors=[prediction(teacher,d)[:2] for d in data]
opt=torch.optim.Adam(wm.parameters(),lr=1e-5);history=[]
def metrics(d):
 with torch.no_grad():
  s1,s2,j1,j2=prediction(wm,d);truth=d['actual_two_step'];real1=d['actual_one_step'];return {'two_step_relative_error':float((j2-truth).norm()/truth.norm()),'two_step_cosine':float(torch.nn.functional.cosine_similarity(j2.flatten(),truth.flatten(),dim=0)),'one_step_response_norm':float(j1.norm()),'true_one_step_response_norm':float(real1.norm()),'true_two_step_response_norm':float(truth.norm())}
for step in range(101):
 if step in (0,20,100):history.append({'step':step,'train40':metrics(data[0]),'train120':metrics(data[1]),'heldout240':metrics(data[2])});(OUT/'state.json').write_text(json.dumps({'phase':'running','history':history},indent=2))
 if step==100:break
 loss=0
 for i in (0,1):
  s1,s2,j1,j2=prediction(wm,data[i]);truth=data[i]['actual_two_step'];true1=data[i]['actual_one_step'];response=(j2-truth).square().mean()/truth.square().mean().clamp(min=1e-8)
  # Physics one-frame delay: immediate response is near zero; fit measured data, do not hard mask.
  immediate=(j1-true1).square().mean()/truth.square().mean().clamp(min=1e-8)
  nominal=((s1-anchors[i][0]).square().mean()+(s2-anchors[i][1]).square().mean())
  loss=loss+response+immediate+10*nominal
 opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(wm.parameters(),10));assert torch.isfinite(loss) and grad==grad;opt.step()
report={'phase':'completed_pending_review','policy_updated':False,'physics_executed':False,'offline_wm_updates':100,'learning_rate':1e-5,'seed':36801,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'branch_sha256':hashes,'parent_checkpoint_sha256':hashlib.sha256(Path(final['checkpoint']).read_bytes()).hexdigest(),'history':history,'world_model_globally_calibrated':False,'controller_qualified':False,'limitations':'Three local prefix/seed datasets only; prefix240 held out from optimization, not a functional trial. No deployment or actor update. Nominal anchors constrain only sampled states.'}
(OUT/'state.json').write_text(json.dumps(report,indent=2));torch.save({'world_model':wm.state_dict(),'optimizer':opt.state_dict(),'report':report},OUT/'offline_wm_only.pt');print(json.dumps(report,indent=2))
