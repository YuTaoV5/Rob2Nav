import hashlib, json, math, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
import genesis as gs
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
urdf,pose,load=map(Path,sys.argv[1:4])
out=Path(sys.argv[4]); out.mkdir(exist_ok=False)
n_envs=int(sys.argv[5])
p=json.loads(pose.read_text())['best']
static=json.loads(load.read_text())['axes']
gs.init(backend=gs.gpu)
dt=.005
scene=gs.Scene(sim_options=gs.options.SimOptions(dt=dt),rigid_options=gs.options.RigidOptions(enable_neutral_collision=True,batch_links_info=True),show_viewer=False)
ground=scene.add_entity(gs.morphs.Plane())
r=scene.add_entity(gs.morphs.URDF(file=str(urdf),fixed=False,merge_fixed_links=False,pos=(0,0,p['base_height_m']+.002)))
scene.build(n_envs=n_envs)
names=list(p['joints_rad'])
ids=[r.get_joint(n).dofs_idx_local[0] for n in names]
T=lambda x: torch.as_tensor(x,dtype=torch.float32,device=gs.device)
ref=T(list(p['joints_rad'].values()))
caps=T([static[n]['cap_nm'] for n in names])
ff=T([static[n]['torque_nm'] for n in names])
lim={j.get('name'):j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
leg=[i for i,n in enumerate(names) if not n.endswith('foot')]
lo=T([{'hip':0,'thigh':.8,'calf':-1.6}[names[i].split('_')[-1]]*.1+.9*float(lim[names[i]]['lower']) for i in leg])
hi=T([{'hip':0,'thigh':.8,'calf':-1.6}[names[i].split('_')[-1]]*.1+.9*float(lim[names[i]]['upper']) for i in leg])
rear=[names.index(n) for n in ('RL_foot','RR_foot')]
wheels=[i for i,n in enumerate(names) if n.endswith('foot')]

class HoldEnv:
    def __init__(self):
        self.device = str(gs.device)
        self.num_envs = n_envs
        self.num_actions = 14
        self.max_episode_length = 500
        self.cfg = dict(stage='hold', scope='Genesis development training', physics_dt=dt, decimation=4, mass_scale=[.95,1.05],wheel_friction_ratio=[.8,1.2],motor_torque_scale=[.9,1.1],delay_physics_steps=[1,2],push_force_n=[5,20],push_duration_s=.04,push_start_s=[1,7])
        self.episode_length_buf = torch.zeros(n_envs,device=gs.device,dtype=torch.long)
        self.action = torch.zeros((n_envs,14),device=gs.device)
        self.filtered_v = torch.zeros(n_envs,device=gs.device)
        self.initial_x = torch.zeros_like(self.filtered_v)
        self.pending = torch.zeros((n_envs,16),device=gs.device)
        self.rear_links = [r.get_link(name).idx for name in ('RL_foot','RR_foot')]
        self.generator = torch.Generator(device=gs.device).manual_seed(71931)
        r.set_dofs_force_range(-caps,caps,dofs_idx_local=ids)
        self.eval_probe = None
        self.nominal_mass = r.get_mass().clone()
        self.wheel_links = [link.idx_local for link in (r.get_link(names[i]) for i in wheels)]
        self.motor_scale = torch.ones((n_envs,16),device=gs.device)
        self.extra_delay = torch.zeros(n_envs,device=gs.device,dtype=torch.bool)
        self.older = torch.zeros_like(self.pending)
        self.push_link = r.get_link('base_link')
        self.push_force = torch.zeros((n_envs,3),device=gs.device)
        self.push_tick = torch.zeros_like(self.episode_length_buf)
        self.reset(torch.arange(n_envs,device=gs.device))
        self.eval_probe = None

    def reset(self, env_ids):
        count = len(env_ids)
        q = ref.expand(count,-1).clone()
        q[:,leg] = (q[:,leg]+.004*(2*torch.rand((count,12),device=gs.device,generator=self.generator)-1)).clamp(lo,hi)
        pitch = p['pitch_rad'] + .02*torch.rand(count,device=gs.device,generator=self.generator)-.02
        z = torch.zeros_like(pitch)
        r.set_pos(T([0,0,p['base_height_m']+.002]).repeat(count,1),envs_idx=env_ids)
        r.set_quat(torch.stack((torch.cos(pitch/2),z,torch.sin(pitch/2),z),1),envs_idx=env_ids)
        r.set_dofs_position(q,dofs_idx_local=ids,envs_idx=env_ids)
        r.zero_all_dofs_velocity(envs_idx=env_ids)
        self.initial_x[env_ids] = r.get_pos(envs_idx=env_ids)[:,0]
        self.episode_length_buf[env_ids] = 0
        self.filtered_v[env_ids] = 0
        self.action[env_ids] = 0
        self.pending[env_ids] = 0
        self.older[env_ids] = 0
        sample = torch.rand((count,25),device=gs.device,generator=self.generator)
        r.set_mass(self.nominal_mass[env_ids]*(.95+.10*sample[:,0]),envs_idx=env_ids)
        r.set_friction_ratio(.8+.4*sample[:,1:5],links_idx_local=self.wheel_links,envs_idx=env_ids)
        self.motor_scale[env_ids] = .90+.20*sample[:,5:21]
        self.extra_delay[env_ids] = sample[:,21]>.5
        direction = 2*math.pi*sample[:,22]
        force = 5+15*sample[:,23]
        self.push_force[env_ids] = torch.stack((force*direction.cos(),force*direction.sin(),torch.zeros_like(force)),1)
        self.push_tick[env_ids] = (50+300*sample[:,24]).long()

    def state(self):
        q = r.get_dofs_position(dofs_idx_local=ids)
        dq = r.get_dofs_velocity(dofs_idx_local=ids)
        w,x,y,z = r.get_quat().unbind(1)
        pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
        roll = torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        return q,dq,pitch,roll,r.get_ang(),r.get_vel()

    def get_observations(self):
        q,dq,pitch,roll,omega,v = self.state()
        displacement = r.get_pos()[:,0]-self.initial_x
        obs = torch.cat((q[:,leg]-ref[leg],dq*.1,torch.stack((pitch-p['pitch_rad'],roll,self.filtered_v,displacement),1),
                         omega*.25,v,self.action,self.pending/caps),1)
        return TensorDict({'policy':obs},batch_size=[n_envs])

    def step(self,actions):
        previous = self.action.clone()
        finite_action = torch.isfinite(actions).all(1)
        self.action.copy_(torch.where(finite_action[:,None],actions,0).clamp(-1,1))
        target = (ref[leg]+.08*self.action[:,:12]).clamp(lo,hi)
        failed = ~finite_action
        reward = torch.zeros(n_envs,device=gs.device)
        for _ in range(4):
            q,dq,pitch,roll,omega,v = self.state()
            err = pitch-p['pitch_rad']
            x = r.get_pos()
            contacts = r.get_contacts(with_entity=ground)
            valid = contacts['valid_mask']
            allowed = torch.zeros_like(valid)
            loads = []
            for link in self.rear_links:
                a,b = contacts['link_a']==link,contacts['link_b']==link
                allowed |= a|b
                loads.append((torch.where(a,contacts['force_a'][...,2],
                                         torch.where(b,contacts['force_b'][...,2],0))*valid).sum(1))
            failed |= ((valid&~allowed).any(1) | (err.abs()>.25) | (x[:,2]<.3) |
                       ((q[:,leg]<lo)|(q[:,leg]>hi)).any(1) | (dq[:,wheels].abs()>98.47).any(1))
            self.filtered_v.lerp_(v[:,0],dt/(dt+.10))
            tau = torch.zeros_like(q)
            tau[:,leg] = ff[leg]+55*(target-q[:,leg])-2*dq[:,leg]
            hip = (32*roll+4*omega[:,0]).clamp(-6,6)
            thigh = (-12*roll-2*omega[:,0]).clamp(-6,6)
            tau[:,names.index('RL_hip')] -= hip
            tau[:,names.index('RR_hip')] -= hip
            tau[:,names.index('RL_thigh')] += thigh
            tau[:,names.index('RR_thigh')] -= thigh
            drive = 40*(err+.04)+6*omega[:,1]+10*self.filtered_v+1.5*(x[:,0]-self.initial_x)
            tau[:,rear] = drive[:,None]+2*self.action[:,12:]
            applied = (torch.where(self.extra_delay[:,None],self.older,self.pending)*self.motor_scale).clamp(-caps,caps)*~failed[:,None]
            r.control_dofs_force(applied,dofs_idx_local=ids)
            self.older.copy_(self.pending)
            self.pending.copy_(tau.clamp(-caps,caps))
            min_load = torch.stack(loads,1).amin(1)
            if self.eval_probe is not None:
                self.eval_probe['peak_speed'] = torch.maximum(self.eval_probe['peak_speed'],v[:,0].abs())
                self.eval_probe['peak_displacement'] = torch.maximum(self.eval_probe['peak_displacement'],(x[:,0]-self.initial_x).abs())
                self.eval_probe['peak_torque_ratio'] = torch.maximum(self.eval_probe['peak_torque_ratio'],(applied/caps).abs().amax(1))
                if self.eval_step >= 13:
                    self.eval_probe['min_load'] = torch.minimum(self.eval_probe['min_load'],min_load)
                    self.eval_probe['settled_peak_speed'] = torch.maximum(self.eval_probe['settled_peak_speed'],v[:,0].abs())
            support = min_load.clamp(0,80)/80
            reward += (~failed)*(1+.6*support-2*err.square()-roll.square()
                                  -4*v[:,0].square()-2*(x[:,0]-self.initial_x).square()
                                  -.03*(applied/caps).square().mean(1))
            pushing = (self.episode_length_buf>=self.push_tick)&(self.episode_length_buf<self.push_tick+2)&~failed
            self.push_link.apply_external_force(self.push_force*pushing[:,None])
            scene.step()
        self.episode_length_buf += 1
        timeout = (self.episode_length_buf>=self.max_episode_length)&~failed
        done = timeout|failed
        reward = reward/4-.05*(self.action-previous).square().mean(1)-2*failed
        env_ids = done.nonzero().flatten()
        if len(env_ids) and self.eval_probe is None:
            self.reset(env_ids)
        return self.get_observations(),reward,done,{'time_outs':timeout}

torch.manual_seed(71931)
env = HoldEnv()
cfg = dict(num_steps_per_env=16, save_interval=1000, logger='tensorboard',
           obs_groups=dict(actor=['policy'],critic=['policy']),
           actor=dict(class_name='MLPModel',hidden_dims=[128,128],distribution_cfg=dict(class_name='GaussianDistribution',init_std=.35)),
           critic=dict(class_name='MLPModel',hidden_dims=[128,128]),
           algorithm=dict(class_name='PPO',num_learning_epochs=3,num_mini_batches=4,learning_rate=.0003,
                          gamma=.99,lam=.95,entropy_coef=.003,clip_param=.2))
(out/'config.json').write_text(json.dumps(dict(runner=cfg, env=env.cfg, num_envs=n_envs, seed=71931, updates=int(sys.argv[6]), eval_interval=int(sys.argv[7]))))
for source in (Path(__file__),pose,load):
    shutil.copyfile(source,out/source.name)
(out/'input_hashes.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__),urdf,pose,load)}))
runner=OnPolicyRunner(env,cfg,log_dir=str(out),device=env.device)
def evaluate(path, policy, count):
    # A fixed reset RNG stream isolates development probes from training resets.
    state = env.generator.get_state()
    env.generator.manual_seed(371944)
    ids = torch.arange(n_envs,device=gs.device)
    env.reset(ids)
    env.eval_probe = {key:torch.zeros(n_envs,device=gs.device) for key in
                      ('peak_speed','settled_peak_speed','peak_displacement','peak_torque_ratio')}
    env.eval_probe['min_load'] = torch.full((n_envs,),float('inf'),device=gs.device)
    alive = torch.ones(n_envs,dtype=torch.bool,device=gs.device)
    with torch.inference_mode():
        obs = env.get_observations()
        for step in range(500):
            env.eval_step = step
            obs, _, done, info = env.step(policy(obs))
            alive &= ~(done & ~info['time_outs'])
    metrics = env.eval_probe
    passed = alive & (metrics['settled_peak_speed']<=.10) & (metrics['min_load']>=10.)
    record = dict(failed=int((~alive).sum()), scope='Genesis development probe',update=count,seed=371944,
                  survived=int(alive.sum()),passed=int(passed.sum()),envs=n_envs,
                  thresholds=dict(hold_s=10,settle_s=.26,max_speed_m_s=.10,min_rear_load_n=10),
                  samples={key:value.tolist() for key,value in metrics.items()})
    path.write_text(json.dumps(record))
    print('HOLD_EVAL',count,record['survived'],record['passed'],flush=True)
    env.eval_probe = None
    env.generator.set_state(state)
    env.reset(ids)

updates=int(sys.argv[6])
interval=int(sys.argv[7])
if updates<1 or interval<1:
    raise ValueError('positive update and evaluation counts required')
for completed in range(0,updates,interval):
    block=min(interval,updates-completed)
    runner.current_learning_iteration=completed
    runner.learn(num_learning_iterations=block)
    count=completed+block
    runner.save(str(out/f'hold-{count:05d}.pt'),infos={'completed_updates':count})
    evaluate(out/f'hold-{count:05d}.json',runner.get_inference_policy(),count)
