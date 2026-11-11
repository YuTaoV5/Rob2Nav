"""Bounded eight-step WM-only fit on real episode-split simulation trajectories."""
import sys,json,hashlib,copy,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from go2_supermimic.supermimic.train import WorldModel
from go2_supermimic.supermimic import state as st
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'reproduction/animal_expression_v1';DATA=ROOT/'animal_delay_trajectories_v1';OUT=ROOT/'animal_wm_windows_v1';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'updates_per_arm':200,'batch_windows':64,'lr':1e-5,'seed':36901,'window':8,'train_seeds':[36820,36821],'validation_seed':36822,'test_seed':36823,'holdout_envs':'0:8 student only','normalization':'frozen original parent stats; no heldout update','policy_updated':False},f)
torch.manual_seed(36901);torch.set_float32_matmul_precision('highest');final=json.loads((ROOT/'animal_crawl_resetfix_v1/state.json').read_text());ck=torch.load(final['checkpoint'],map_location='cuda',weights_only=False);parent=WorldModel(tuple(ck['cfg']['wm_hidden'])).cuda().eval();parent.load_state_dict(ck['world_model']);groups={};hashes={};contract=[]
def windows(t,ids):
 out={}
 for key,val in t.items():
  x=val[:,ids];shape=x.shape[2:];out[key]=x.reshape(60,8,len(ids),*shape).transpose(1,2).reshape(60*len(ids),8,*shape).cuda()
 return out
for seed in [36820,36821,36822,36823]:
 path=DATA/f'trajectory_{seed}.pt';hashes[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest();payload=torch.load(path,map_location='cpu',weights_only=False);t=payload['tensors'];assert not t['reset_mask'].any();history=float((t['a_previous'][1:]-t['a_intended'][:-1]).abs().max());phase=float((t['phase1']-t['phase0']-1).abs().max());assert history==phase==0 and torch.equal(t['a_previous'],t['a_executed'])
 a=windows(t,list(range(16)));context=float((a['xi']-a['xi'][:,:1]).abs().max());assert context==0;contract.append({'seed':seed,'action_history_error':history,'clock_error':phase,'context_change_within_8steps':context})
 if seed<36822:
  for role,ids in [('student',list(range(8))),('teacher',list(range(8,16)))]:groups.setdefault('train_'+role,[]).append(windows(t,ids))
 else:groups['validation_student' if seed==36822 else 'test_student']=windows(t,list(range(8)))
for role in ['student','teacher']:
 parts=groups['train_'+role];groups['train_'+role]={k:torch.cat([d[k] for d in parts]) for k in parts[0]}
groups['train_all']={k:torch.cat([groups['train_student'][k],groups['train_teacher'][k]]) for k in groups['train_student']}
clipping={}
with torch.no_grad():
 for name,g in groups.items():
  if name=='train_all':continue
  flatten=lambda k:g[k].reshape(-1,g[k].shape[-1]);raw=parent._inputs(flatten('s0'),flatten('a_intended'),flatten('a_previous'),flatten('xi'),flatten('relief'));z=(raw-parent.in_norm.mean)/(parent.in_norm.var.sqrt()+1e-6);clip=z.abs()>parent.in_norm.clip
  clipping[name]={'windows':len(g['s0']),'feature_fraction_outside_clip':float(clip.float().mean()),'rows_with_any_clipped_feature':float(clip.any(-1).float().mean()),'max_abs_standardized_input':float(z.abs().max()),'current_action_clipped_fraction':float(clip[:,st.STATE_DIM:st.STATE_DIM+12].float().mean()),'previous_action_clipped_fraction':float(clip[:,st.STATE_DIM+12:st.STATE_DIM+24].float().mean())}
state={'phase':'preflight_passed','contract':contract,'input_clip_audit':clipping,'arms':{},'policy_updated':False,'physics_executed':False,'normalization_updated':False}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
def loss(wm,g,idx):
 x={k:v[idx] for k,v in g.items()};return wm.loss(x['s0'][:,0],x['a_intended'],x['a_previous'],x['xi'][:,0],x['relief'],x['s1'],x['pose_delta'],(~x['reset_mask']).float())
def evaluate(wm,g):
 total=0.;velocity=0.;joint=0.;count=0
 with torch.no_grad():
  for start in range(0,len(g['s0']),64):
   idx=slice(start,start+64);x={k:v[idx] for k,v in g.items()};n=len(x['s0']);total+=float(loss(wm,g,idx))*n;s=x['s0'][:,0]
   for j in range(8):
    s,_=wm(s,x['a_intended'][:,j],x['a_previous'][:,j],x['xi'][:,0],x['relief'][:,j]);assert torch.isfinite(s).all();velocity+=float((s[:,st.V_YAW]-x['s1'][:,j,st.V_YAW]).square().mean())*n;joint+=float((s[:,st.Q]-x['s1'][:,j,st.Q]).square().mean())*n
   count+=n
 return {'normalized_8step_state_pose_loss':total/count,'raw_8step_linear_velocity_rmse':(velocity/(8*count))**.5,'raw_8step_joint_position_rmse':(joint/(8*count))**.5,'windows':count}
for delayed in [False,True]:
 torch.manual_seed(36901);name='delayed_input' if delayed else 'unconstrained';wm=copy.deepcopy(parent)
 if delayed:
  original=wm.forward
  def forward(self,s,a,a_prev,xi,relief):
   neutral=self.in_norm.mean[st.STATE_DIM:st.STATE_DIM+12][None].expand_as(a);return original(s,neutral,a_prev,xi,relief)
  wm.forward=types.MethodType(forward,wm)
 opt=torch.optim.Adam(wm.parameters(),lr=1e-5);state['current_arm']=name;state['arms'][name]={'history':[]};save()
 for update in range(201):
  if update in [0,50,200]:
   scored=['train_student','train_teacher','validation_student']+(['test_student'] if update in [0,200] else []);metrics={group:evaluate(wm,groups[group]) for group in scored};state['arms'][name]['history'].append({'updates':update,'metrics':metrics});save()
  if update==200:break
  idx=torch.randint(len(groups['train_all']['s0']),(64,),device='cuda');l=loss(wm,groups['train_all'],idx);opt.zero_grad();l.backward();grad=torch.nn.utils.clip_grad_norm_(wm.parameters(),10);assert torch.isfinite(l) and torch.isfinite(grad);opt.step()
  if (update+1)%25==0:state['updates']=update+1;state['last_training_loss']=float(l);save()
 p=OUT/(name+'_wm_only.pt');torch.save({'world_model':wm.state_dict(),'optimizer':opt.state_dict(),'explicit_delayed_input':delayed},p);state['arms'][name]['artifact_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();save()
state.update(phase='completed_pending_review',updates_per_arm=200,branch_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(Path(final['checkpoint']).read_bytes()).hexdigest(),deployed=False,controller_qualified=False,limitations='Only fixed reference initial poses; no added initial perturbation. Test/validation student-only complete episodes. Current normalization frozen from original training; no heldout stats update. Raw rollout prediction does not certify action derivatives, policy control or functional crawling. WM artifacts server-only, original actor/checkpoint unchanged.')
save();print(json.dumps(state,indent=2))
