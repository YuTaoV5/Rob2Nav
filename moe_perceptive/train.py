"""Train pretrained MoE terrain fusion with value/fusion/actor/joint phases."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--warmup',type=int,default=50);p.add_argument('--fusion',type=int,default=300)
p.add_argument('--actor',type=int,default=300);p.add_argument('--seed',type=int,default=42)
p.add_argument('--resume');p.add_argument('--kind',default='mixed',choices=['mixed','flat','stairs','platform','ame_stakes','ame_alternate'])
p.add_argument('--initialize-from',help='Load policy/critic only into a new experiment; do not restore optimizer or curriculum.')
p.add_argument('--memory-lifetime',type=float,default=1.)
p.add_argument('--navigation',action='store_true');p.add_argument('--teacher-map',action='store_true')
p.add_argument('--sensor-rig',choices=['original','near_down60','near_down60_front_lidar'],default='original')
p.add_argument('--course',choices=['original','wide_deep'],default='original')
p.add_argument('--arrival-aware',action='store_true',help='Experimental distance-aware velocity and braking rewards; new runs only.')
p.add_argument('--numeric-debug',action='store_true',help='Stop at the first nonfinite observation, rollout value or backward operation.')
p.add_argument('--foothold-aware',action='store_true',help='Experimental privileged descending-foot support reward; no action correction.')
p.add_argument('--spatial-attention',action='store_true',help='Soft nominal leg neighborhoods for the four terrain heads.')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from rsl_rl.runners import OnPolicyRunner
from .model import register_policy
from .task import make_cfg
from .mdp import match_joint_limits,JOINTS,DEFAULT
from .provenance import capture

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.autograd.set_detect_anomaly(a.numeric_debug)
if a.memory_lifetime<=0:raise ValueError('Memory lifetime must be positive')
if a.resume and a.initialize_from:raise ValueError('Choose resume or initialize-from')
if a.teacher_map and not a.navigation:raise ValueError('Teacher map requires explicit navigation experiment')
out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
meta={**capture(out,a),'torch':torch.__version__,'actor_observation':'5x45 history + 3x693 visible height/mask/age',
      'simulation_sensors':f'simplified depth/lidar rays, ideal odometry, {a.memory_lifetime:g} s observed-return memory',
      'protocol':'pretrained 137k MoE, new terrain attention, progressively unfrozen PPO; not AME scratch replication'}
meta['perception_role']='privileged full-map teacher; not deployable sensor policy' if a.teacher_map else 'observed-return student'
if a.navigation:
    meta['actor_observation']+=' + clipped relative XY goal and sin/cos heading'
    meta['training_geometry']='Explicit shallow-pit precursor; evaluation must use original AME pits.'
(out/'manifest.json').write_text(json.dumps(meta,indent=2))
cfg=make_cfg(a.num_envs,a.seed,a.kind);cfg.sim.device=a.device or 'cuda:0'
cfg.moe_memory_lifetime=a.memory_lifetime
from .rig import configure
configure(cfg,a.sensor_rig)
if a.navigation:
    from .navigation import apply
    apply(cfg,a.teacher_map,training=True)
from .sparse_course import configure as configure_course
configure_course(cfg,a.course,training=True)
if a.arrival_aware:
    if not a.navigation:raise ValueError('Arrival-aware requires navigation')
    from .arrival import configure as configure_arrival
    configure_arrival(cfg)
if a.foothold_aware:
    if not (a.navigation and a.teacher_map):raise ValueError('Foothold probe requires explicit privileged teacher')
    from .foothold import configure as configure_foothold
    configure_foothold(cfg)
if a.course=='wide_deep':
    meta['training_geometry']='Wide-support/small-gap precursor; real 2m pit at every level; final geometry delegates to original AME.'
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
dump_yaml(str(out/'env.yaml'),cfg)
env=ManagerBasedRLEnv(cfg=cfg);match_joint_limits(env.scene['robot'])
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
agent=UnitreeGo2RoughPPORunnerCfg();agent.seed=a.seed;agent.device=cfg.sim.device
agent.obs_groups={'policy':['policy'],'critic':['critic']};agent.save_interval=100
agent.policy.class_name='MoEPerceptiveActorCritic';agent.policy.init_noise_std=.15
agent.policy.noise_std_type='log';agent.policy.actor_obs_normalization=False;agent.policy.critic_obs_normalization=False
agent.policy.critic_hidden_dims=[256,128,64]
agent.algorithm.learning_rate=2e-4;agent.algorithm.schedule='fixed'
agent.algorithm.entropy_coef=.001;agent.algorithm.num_learning_epochs=4;agent.algorithm.num_mini_batches=4
train_cfg=agent.to_dict();train_cfg['policy']['moe_checkpoint']=a.moe
train_cfg['policy']['goal_conditioned']=a.navigation
train_cfg['policy']['spatial_prior']=a.spatial_attention
dump_yaml(str(out/'agent.yaml'),train_cfg);register_policy()
runner=OnPolicyRunner(wrapped,train_cfg,log_dir=str(out),device=cfg.sim.device)
policy=runner.alg.policy;actor=policy.actor;env._moe_anchor=actor.anchor
# Explicit stable groups permit a smaller learning rate for pretrained weights.
old_actor=list(actor.base.actor.parameters());old_encoder=list(actor.base.student_moe_encoder.parameters())
excluded={id(v) for v in old_actor+old_encoder+list(actor.anchor.parameters())}
new=[v for v in policy.parameters() if id(v) not in excluded]
runner.alg.optimizer=torch.optim.Adam([
    {'params':new,'lr':2e-4,'name':'new'}, {'params':old_actor,'lr':2e-5,'name':'pretrained_actor'},
    {'params':old_encoder,'lr':1e-5,'name':'pretrained_encoder'}])
if a.initialize_from:
    ck=torch.load(a.initialize_from,map_location='cpu',weights_only=False)
    if ck['infos']['provenance']['moe_sha256']!=meta['moe_sha256']:raise ValueError('Initializer MoE mismatch')
    state=ck['model_state_dict']
    if a.navigation and not ck['infos']['provenance']['args'].get('navigation',False):
        old=state['critic.0.weight'];new=policy.state_dict()['critic.0.weight'].clone().zero_()
        if new.shape[0]!=old.shape[0] or new.shape[1]!=old.shape[1]+4:raise ValueError('Unexpected critic resize')
        new[:,:old.shape[1]]=old;state['critic.0.weight']=new
        meta['critic_initialization']='Old critic input columns copied exactly; four new goal columns zero.'
    # RSL's ActorCritic override returns bool. Use the PyTorch base method to
    # inspect missing/unexpected keys explicitly during architecture migration.
    missing,unexpected=torch.nn.Module.load_state_dict(policy,state,strict=False)
    # Only the new zero-initialized goal adapter may be absent in a v1/v2 initializer.
    if unexpected or set(missing)-{'actor.goal_projection.weight','actor.goal_projection.bias'}:
        raise ValueError(f'Initializer state mismatch: {missing}, {unexpected}')
    meta['initialized_from_sha256']=hashlib.sha256(Path(a.initialize_from).read_bytes()).hexdigest()
    meta['initialization_note']='Policy and critic only; fresh Adam, RNG, episodes and curriculum.'
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
if a.resume:
    ck=torch.load(a.resume,map_location='cpu',weights_only=False)
    for key in ('moe_sha256','protocol'):
        if ck['infos']['provenance'][key]!=meta[key]:raise ValueError('Resume provenance mismatch')
    old=ck['infos']['provenance']['args']
    for key in ('seed','num_envs','kind','warmup','fusion','actor'):
        if old[key]!=vars(a)[key]:raise ValueError('Resume changed '+key)
    if old.get('memory_lifetime',1.)!=a.memory_lifetime:raise ValueError('Resume changed memory lifetime')
    for key in ('navigation','teacher_map'):
        if old.get(key,False)!=vars(a)[key]:raise ValueError('Resume changed '+key)
    if old.get('sensor_rig','original')!=a.sensor_rig:raise ValueError('Resume changed sensor rig')
    if old.get('course','original')!=a.course:raise ValueError('Resume changed course')
    if old.get('arrival_aware',False)!=a.arrival_aware:raise ValueError('Resume changed arrival reward profile')
    if old.get('foothold_aware',False)!=a.foothold_aware:raise ValueError('Resume changed foothold reward profile')
    if old.get('spatial_attention',False)!=a.spatial_attention:raise ValueError('Resume changed spatial attention')
    runner.load(a.resume)
    levels=ck['terrain_levels'].to(env.device);t=env.scene.terrain
    t.terrain_levels[:]=levels;t.env_origins[:]=t.terrain_origins[levels,t.terrain_types]
    env.scene.env_origins[:]=t.env_origins;env.reset()
    # Resume per-level qualification as well as terrain indices. Older files
    # lack these fields and intentionally requalify from zero.
    for name,value in ck.get('course_state',{}).items():
        if name not in ('_deep_ema','_deep_attempts'):raise ValueError('Unknown course state '+name)
        if value.shape!=(env.num_envs,):raise ValueError('Course state shape mismatch')
        setattr(env,name,value.to(env.device).clone())
    random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
    torch.set_rng_state(ck['rng']['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng']['cuda']])
start_iteration=runner.current_learning_iteration
initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
base_initial={k:v.detach().cpu().clone() for k,v in actor.base.state_dict().items()}
phase='';rollout_calls=0;current_iteration=start_iteration


def phase_at(it):
    if it<a.warmup:return 'value'
    if it<a.warmup+a.fusion:return 'fusion'
    if it<a.warmup+a.fusion+a.actor:return 'actor'
    return 'joint'


def set_phase(name):
    global phase
    if name==phase:return
    actor.set_phase(name);policy.log_std.requires_grad_(name!='value');phase=name
    for g,rate in zip(runner.alg.optimizer.param_groups,(2e-4,2e-5,1e-5)):g['lr']=rate
    record={'iteration':current_iteration,'phase':name,'residual_bound_rad':actor.residual_bound*.25,
        'trainable_parameters':sum(v.numel() for v in policy.parameters() if v.requires_grad)}
    with (out/'phases.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print('PHASE',record,flush=True)


original_act=runner.alg.act
def act(obs):
    global rollout_calls,current_iteration
    if rollout_calls%agent.num_steps_per_env==0:
        current_iteration=start_iteration+rollout_calls//agent.num_steps_per_env
        set_phase(phase_at(current_iteration))
    rollout_calls+=1
    if a.numeric_debug:
        for name,value in obs.items():
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite observation '+name)
    value=original_act(obs)
    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite action')
    return value
runner.alg.act=act
original_update=runner.alg.update
def update():
    if a.numeric_debug:
        for name in ('rewards','returns','advantages','values','actions','actions_log_prob','mu','sigma'):
            value=getattr(runner.alg.storage,name,None)
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite rollout '+name)
    result=original_update()
    with torch.no_grad():policy.log_std.clamp_(math.log(.03),math.log(.5))
    for name,value in policy.named_parameters():
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite parameter '+name)
    if phase=='value':
        for k,v in actor.state_dict().items():torch.testing.assert_close(v.cpu(),initial[k],rtol=0,atol=0)
    if phase in ('value','fusion'):
        for k,v in actor.base.state_dict().items():torch.testing.assert_close(v.cpu(),base_initial[k],rtol=0,atol=0)
    return result
runner.alg.update=update


def save(path,infos=None):
    path=Path(path);temporary=path.with_suffix('.tmp')
    payload={'model_state_dict':policy.state_dict(),'optimizer_state_dict':runner.alg.optimizer.state_dict(),
        'iter':runner.current_learning_iteration+1,'infos':{'provenance':meta,'phase':phase,
        'residual_bound':actor.residual_bound,'frozen_checks_passed':True},
        'terrain_levels':env.scene.terrain.terrain_levels.cpu(),
        'course_state':{name:getattr(env,name).detach().cpu().clone()
                        for name in ('_deep_ema','_deep_attempts') if hasattr(env,name)},
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
    torch.save(payload,temporary);temporary.replace(path);print('CHECKPOINT',str(path),flush=True)
runner.save=save
original_log=runner.log
def log(locs,*args,**kwargs):
    original_log(locs,*args,**kwargs)
    row={'iteration':locs['it'],'phase':phase,'steps_per_s':agent.num_steps_per_env*a.num_envs/(locs['collection_time']+locs['learn_time']),
        'std_min':float(policy.log_std.exp().min()),'std_max':float(policy.log_std.exp().max()),
        'loss':{k:float(v) for k,v in locs['loss_dict'].items()},
        'terrain_histograms':[]}
    terrain=env.scene.terrain
    for col in range(8):
        row['terrain_histograms'].append(torch.bincount(terrain.terrain_levels[terrain.terrain_types==col],minlength=10).tolist())
    with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    (out/'status.json').write_text(json.dumps(row,indent=2))
runner.log=log
actual=env.action_manager.get_term('joint_pos')
assert actual._joint_names==JOINTS
torch.testing.assert_close(actual._offset[0],torch.tensor(DEFAULT,device=env.device))
(out/'io.json').write_text(json.dumps({'joints':JOINTS,'default':DEFAULT,'action_scale':.25,'control_dt':env.step_dt,
    'policy_dim':2308 if a.navigation else 2304,'critic_dim':3697 if a.navigation else 3693,'kp':20.,'kd':.5,
    'asset_joint_names':env.scene['robot'].joint_names,
    'joint_limits':env.scene['robot'].data.joint_pos_limits[0,[env.scene['robot'].joint_names.index(n) for n in JOINTS]].tolist()},indent=2))
started=time.monotonic()
try:
    runner.learn(a.iterations,init_at_random_ep_len=False)
    (out/'completed.json').write_text(json.dumps({'last_iteration':runner.current_learning_iteration,'additional_iterations':a.iterations,
        'elapsed_s':time.monotonic()-started,'phase':phase,'note':'Training completion is not a terrain success result'},indent=2))
except Exception as error:
    (out/'failed.json').write_text(json.dumps({'error':repr(error),'iteration':current_iteration,'phase':phase},indent=2));raise
finally:wrapped.close();app.close()
