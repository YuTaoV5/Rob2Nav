"""Matched bounded WM-only fit: passive+pulse rollouts, with/without response loss."""
import sys,json,hashlib,copy,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'animal_wm_response_mix_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'seed':37101,'updates_per_arm':200,'lr':1e-5,'passive_batch':64,'active_fork_batch':1,'window':8,'normalization':'frozen parent stats','train_prefixes':[30,90,150,240],'validation_prefix':360,'test_prefix':450,'test_eps':.05,'diagnostic_only_eps':.15,'policy_updated':False},f)
torch.manual_seed(37101);torch.set_float32_matmul_precision('highest');final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());base=torch.load(final['checkpoint'],map_location='cuda',weights_only=False);init=ROOT/'animal_wm_windows_v1/delayed_input_wm_only.pt';ck=torch.load(init,map_location='cuda',weights_only=False);parent=WorldModel(tuple(base['cfg']['wm_hidden'])).cuda().eval();parent.load_state_dict(ck['world_model']);groups={};source_hashes={}
def windows(t,ids):
 out={}
 for k,v in t.items():
  x=v[:,ids];shape=x.shape[2:];out[k]=x.reshape(60,8,len(ids),*shape).transpose(1,2).reshape(60*len(ids),8,*shape).cuda()
 return out
for seed in [36820,36821,36822,36823]:
 p=ROOT/f'animal_delay_trajectories_v1/trajectory_{seed}.pt';source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();t=torch.load(p,map_location='cpu',weights_only=False)['tensors'];assert not t['reset_mask'].any()
 if seed<36822:groups.setdefault('train',[]).append(windows(t,list(range(16))))
 else:groups['validation' if seed==36822 else 'test']=windows(t,list(range(8)))
groups['train']={k:torch.cat([v[k] for v in groups['train']]) for k in groups['train'][0]};forks={}
for prefix in [30,90,150,240,360,450]:
 for eps in [.05,.15]:
  key=f'{prefix}_{eps:.2f}';p=ROOT/f'animal_paired_excitation_v1/pulse_prefix{prefix}_eps{eps:.2f}.pt';source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();d=torch.load(p,map_location='cuda',weights_only=False);t=d['tensors'];x={k:v.transpose(0,1) for k,v in t.items()};x['reset_mask']=torch.zeros((104,8),device='cuda',dtype=torch.bool);forks[key]={'x':x,'truth':d['actual_two_step'],'first_truth':d['actual_one_step'],'epsilon':eps,'prefix':prefix}
train_keys=[k for k,v in forks.items() if v['prefix']<360];state={'phase':'running','policy_updated':False,'physics_executed':False,'arms':{},'initialization_sha256':hashlib.sha256(init.read_bytes()).hexdigest()}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
def roll_loss(wm,x,idx):
 t={k:v[idx] for k,v in x.items()};return wm.loss(t['s0'][:,0],t['a_intended'],t['a_previous'],t['xi'][:,0],t['relief'],t['s1'],t['pose_delta'],(~t['reset_mask']).float())
def response(wm,d):
 x=d['x'];s1,_=wm(x['s0'][:,0],x['a_intended'][:,0],x['a_previous'][:,0],x['xi'][:,0],x['relief'][:,0]);s2,_=wm(s1,x['a_intended'][:,1],x['a_previous'][:,1],x['xi'][:,0],x['relief'][:,1]);a=s1.reshape(26,4,-1);b=s2.reshape(26,4,-1);one=torch.stack([(a[2+2*j,:,st.V_YAW][:,:2]-a[3+2*j,:,st.V_YAW][:,:2])/(2*d['epsilon']) for j in range(12)]);two=torch.stack([(b[2+2*j,:,st.V_YAW][:,:2]-b[3+2*j,:,st.V_YAW][:,:2])/(2*d['epsilon']) for j in range(12)]);return one,two
@torch.no_grad()
def evaluate(wm,include_test):
 nominal={}
 for name in ['validation']+(['test'] if include_test else []):
  x=groups[name];n=len(x['s0']);nominal[name]=sum(float(roll_loss(wm,x,slice(i,i+64)))*min(64,n-i) for i in range(0,n,64))/n
 responses=[]
 for key,d in forks.items():
  if d['prefix']==450 and not include_test:continue
  one,two=response(wm,d);truth=d['truth'];responses.append({'fork':key,'split':'train' if d['prefix']<360 else 'validation' if d['prefix']==360 else 'test' if d['epsilon']==.05 else 'diagnostic_only','relative_response_error':float((two-truth).norm()/truth.norm().clamp(min=1e-8)),'response_cosine':float(torch.nn.functional.cosine_similarity(two.flatten(),truth.flatten(),dim=0)),'one_step_response_norm':float(one.norm()),'true_one_step_response_norm':float(d['first_truth'].norm())})
 return {'nominal_8step_loss':nominal,'responses':responses}
for active in [False,True]:
 torch.manual_seed(37101);name='rollouts_only' if not active else 'rollouts_plus_response';wm=copy.deepcopy(parent);original=wm.forward
 def delayed(self,s,a,a_prev,xi,relief):return original(s,self.in_norm.mean[st.STATE_DIM:st.STATE_DIM+12][None].expand_as(a),a_prev,xi,relief)
 wm.forward=types.MethodType(delayed,wm);opt=torch.optim.Adam(wm.parameters(),lr=1e-5);state['current_arm']=name;state['updates']=0;state['arms'][name]={'history':[]};save()
 for update in range(201):
  if update in [0,50,200]:state['arms'][name]['history'].append({'updates':update,'metrics':evaluate(wm,update in [0,200])});save()
  if update==200:break
  idx=torch.randint(len(groups['train']['s0']),(64,),device='cuda');key=train_keys[int(torch.randint(len(train_keys),(1,)).item())];d=forks[key];nom=roll_loss(wm,groups['train'],idx);pulse=roll_loss(wm,d['x'],slice(None));one,two=response(wm,d);response_loss=(two-d['truth']).square().mean()/d['truth'].square().mean().clamp(min=1e-8);loss=nom+pulse+(response_loss if active else 0)
  opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(wm.parameters(),10);assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
  if (update+1)%25==0:state.update(updates=update+1,last_loss=float(loss),last_nominal=float(nom),last_pulse=float(pulse),last_response=float(response_loss));save()
 p=OUT/(name+'_wm_only.pt');torch.save({'world_model':wm.state_dict(),'optimizer':opt.state_dict(),'explicit_delayed_input':True,'response_objective':active},p);state['arms'][name]['artifact_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();save()
state.update(phase='completed_pending_review',updates_per_arm=200,seed=37101,learning_rate=1e-5,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_artifact_sha256=source_hashes,controller_qualified=False,deployed=False,limitations='Matched initialization/seed/budget; both arms learn passive plus pulse absolute 8-step transitions; only response arm adds normalized two-step finite-difference mismatch. Pulse predictions use each branch own recorded s0, not an assertion of identical physics states. 450/.15 retained diagnostic only due start mismatch; different amplitudes remain separate. Test only at fixed baseline/final, no heldout fitting/norm updates. No actor, physical rollout, or functional qualification. WM artifacts server-only.')
save();print(json.dumps(state,indent=2))
