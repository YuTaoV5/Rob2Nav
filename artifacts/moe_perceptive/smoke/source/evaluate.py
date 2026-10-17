"""First-episode native evaluation of original/fused MoE; optional full video."""
import argparse,json,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--checkpoint')
p.add_argument('--kind',choices=['flat','stairs','platform','ame_stakes','ame_alternate'],required=True)
p.add_argument('--difficulty',type=float,default=0.);p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--seed',type=int,default=801);p.add_argument('--output',required=True);p.add_argument('--video',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.video:a.enable_cameras=True
app=AppLauncher(a).app
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from .task import make_cfg
from .model import PerceptiveActor
from .mdp import match_joint_limits,FEET
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)


def capture(env):
    r=env.scene['robot'];s=env.scene['contact_forces'];ids=[r.body_names.index(n) for n in FEET]
    cids=[s.body_names.index(n) for n in FEET]
    env._eval_state={'position':(r.data.root_pos_w-env.scene.env_origins).clone(),
        'upright':-r.data.projected_gravity_b[:,2].clone(),
        'feet':(r.data.body_pos_w[:,ids]-env.scene.env_origins[:,None]).clone(),
        'forces':s.data.net_forces_w[:,cids].clone()}
    return torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)


cfg=make_cfg(a.num_envs,a.seed,a.kind,True,a.difficulty);cfg.sim.device=a.device or 'cuda:0'
cfg.terminations.capture=TerminationTermCfg(func=capture)
env=ManagerBasedRLEnv(cfg=cfg,render_mode='rgb_array' if a.video else None)
match_joint_limits(env.scene['robot']);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
actor=PerceptiveActor(a.moe).to(env.device).eval()
phase='base';checkpoint_sha=None
if a.checkpoint:
    checkpoint=torch.load(a.checkpoint,map_location=env.device,weights_only=False)
    actor.load_state_dict({k.removeprefix('actor.'):v for k,v in checkpoint['model_state_dict'].items() if k.startswith('actor.')},strict=True)
    phase=checkpoint['infos']['phase'];actor.set_phase(phase)
    actor.residual_bound=checkpoint['infos']['residual_bound']
    checkpoint_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
env._moe_anchor=actor.anchor
obs=wrapped.get_observations();done_first=torch.zeros(a.num_envs,device=env.device,dtype=torch.bool)
records=[];trace=[];frames=0;writer=None;maximum_zero_error=0.
if a.video:
    import imageio.v2 as imageio
    writer=imageio.get_writer(out/'episode_0.mp4',fps=25,macro_block_size=1)
try:
    with torch.inference_mode():
        for step in range(int(cfg.episode_length_s/env.step_dt)+5):
            policy_obs=obs['policy'];base=actor.base(policy_obs[:,:225]);actions=actor(policy_obs) if a.checkpoint else base
            if not a.checkpoint:
                maximum_zero_error=max(maximum_zero_error,float((actor(policy_obs)-base).abs().max()))
            if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite policy')
            obs,_,done,_=wrapped.step(actions);state=env._eval_state
            ended=done.bool()&~done_first
            for i in ended.nonzero().flatten().tolist():
                complete=bool(env.termination_manager.get_term('route_complete')[i]);failed=bool(env.termination_manager.terminated[i])
                records.append({'env':i,'success':complete and not failed and float(state['upright'][i])>.7,
                    'route_complete':complete,'failure':failed,'position':state['position'][i].cpu().tolist(),
                    'upright':float(state['upright'][i]),'seconds':(step+1)*env.step_dt,
                    'termination_terms':[n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[i])]})
            if not done_first[0]:
                trace.append({'time':(step+1)*env.step_dt,**{k:v[0].cpu().tolist() for k,v in state.items()},
                    'action':actions[0].cpu().tolist(),'base_action':base[0].cpu().tolist()})
                trace[-1]['known_fraction']=float((policy_obs[0,225+693:225+2*693]>.5).float().mean())
                if writer and step%2==0 and not ended[0]:
                    pos=env.scene['robot'].data.root_pos_w[0].cpu().numpy()
                    env.sim.set_camera_view(pos+np.array([2.3,2.5,1.7]),pos+np.array([.5,0.,0.]))
                    writer.append_data(env.render());frames+=1
            done_first|=ended
            if done_first.all():break
    result={'args':vars(a),'phase':phase,'checkpoint_sha256':checkpoint_sha,
        'moe_sha256':hashlib.sha256(Path(a.moe).read_bytes()).hexdigest(),
        'episodes':records,'successes':sum(r['success'] for r in records),'total':len(records),
        'max_zero_fusion_error':maximum_zero_error if not a.checkpoint else None,'video_frames':frames,
        'success_definition':'First episode: route >3.5 m, then >=0.3 s with >=2 support feet, upright >0.9 and speed <0.25 m/s; no failure termination.',
        'note':'Shared fixed terrain difficulty/seed; reset XY +/-3 cm, yaw +/-0.04 rad. No physical sensor drivers.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'trace.json').write_text(json.dumps(trace))
    torch.jit.script(actor.cpu()).save(str(out/'actor.pt'))
    print('EVALUATION',result['successes'],'/',result['total'],flush=True)
finally:
    if writer:writer.close()
    wrapped.close();app.close()
