"""FR-Net checkpoint sim-to-sim screening. No learning and no mid-rollout state edits."""
import argparse,json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,torch,mujoco
ap=argparse.ArgumentParser();ap.add_argument('--root',default='/root/gpufree-data');ap.add_argument('--seeds',default='35010');ap.add_argument('--seconds',type=float,default=10);ap.add_argument('--output',required=True);args=ap.parse_args();root=Path(args.root)
source=root/'Rob2Nav-upstreams/FR-Net';weight=source/'logs/GO2_recovery/pretrained_recovery_v3/model_2000.pt';modelpath=root/'Rob2Nav/deployment/models/go2_isaac.xml'
torch.set_num_threads(1);d=torch.load(weight,map_location='cpu',weights_only=True);net=torch.nn.Sequential(torch.nn.Linear(45,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12));net.load_state_dict({k.removeprefix('actor.'):v for k,v in d['model_state_dict'].items() if k.startswith('actor.')});net.eval()
urdf=ET.parse(source/'resources/robots/go2/urdf/go2.urdf').getroot();joints=[j for j in urdf.findall('joint') if j.get('type') in ('revolute','continuous')];names=[j.get('name') for j in joints];assert len(names)==12
m=mujoco.MjModel.from_xml_path(str(modelpath));m.opt.timestep=.005
jid=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]);assert (jid>=0).all();qadr=m.jnt_qposadr[jid];vadr=m.jnt_dofadr[jid]
aids=np.array([next(i for i in range(m.nu) if m.actuator_trnid[i,0]==j) for j in jid]);default=np.array([0 if '_hip_' in n else (1.5 if '_thigh_' in n else -2.4) for n in names]);limits=np.array([float(j.find('limit').get('velocity')) for j in joints]);effort=np.array([float(j.find('limit').get('effort')) for j in joints]);hard=np.array([[float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))] for j in joints]);base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base');assert base>=0
poses=[('back',np.pi,0),('left',np.pi/2,0),('right',-np.pi/2,0),('nose',0,np.pi/2),('tail',0,-np.pi/2),('upright',0,0)];rows=[]
for seed in map(int,args.seeds.split(',')):
 rng=np.random.default_rng(seed)
 for label,roll,pitch in poses:
  data=mujoco.MjData(m);data.qpos[:3]=[0,0,.35];quat=np.zeros(4);mujoco.mju_euler2Quat(quat,np.array([roll,pitch,rng.uniform(-np.pi,np.pi)]),'xyz');data.qpos[3:7]=quat;data.qpos[qadr]=default;mujoco.mj_forward(m,data)
  previous=np.zeros(12);target=default.copy();peak_v=peak_limit=peak_target=0.;stable=0.;recovered=False;recovery_time=None;contact_after=False
  for step in range(round(args.seconds/.005)):
   if step%4==0:
    rot=data.xmat[base].reshape(3,3);motion=np.zeros(6);mujoco.mj_objectVelocity(m,data,mujoco.mjtObj.mjOBJ_BODY,base,motion,1)
    obs=np.r_[motion[:3]*.25,rot.T@np.array([0,0,-1]),np.zeros(3),data.qpos[qadr]-default,data.qvel[vadr]*.05,previous].clip(-100,100).astype(np.float32)
    with torch.inference_mode():action=net(torch.from_numpy(obs)).numpy().clip(-10,10)
    new=default+action*.25;peak_target=max(peak_target,float(np.max(np.abs(new-target)/.02/limits)));target=new;previous=action.copy()
   torque=np.clip(30*(target-data.qpos[qadr])-.8*data.qvel[vadr],-effort,effort);data.ctrl[aids]=torque;mujoco.mj_step(m,data)
   peak_v=max(peak_v,float(np.max(np.abs(data.qvel[vadr])/limits)));peak_limit=max(peak_limit,float(np.maximum(hard[:,0]-data.qpos[qadr],0).max()),float(np.maximum(data.qpos[qadr]-hard[:,1],0).max()))
   rot=data.xmat[base].reshape(3,3);motion=np.zeros(6);mujoco.mj_objectVelocity(m,data,mujoco.mjtObj.mjOBJ_BODY,base,motion,1)
   feet=set();nonfoot=False
   for c in data.contact:
    if c.dist>0:continue
    n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1) or '';n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2) or ''
    if m.geom_bodyid[c.geom1]==0 or m.geom_bodyid[c.geom2]==0:
     robotgeom=c.geom2 if m.geom_bodyid[c.geom1]==0 else c.geom1;bodyname=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[robotgeom]) or '';geomname=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,robotgeom) or ''
     if 'foot' in (bodyname+geomname).lower():feet.add(bodyname+geomname)
     else:nonfoot=True
   good=rot[2,2]>.95 and data.xpos[base,2]>.24 and np.linalg.norm(motion[3:5])<.2 and np.linalg.norm(motion[:3])<.5 and len(feet)>=3 and not nonfoot
   stable=stable+.005 if good else 0.
   if stable>=2 and not recovered:recovered=True;recovery_time=(step+1)*.005
  rows.append(dict(seed=seed,pose=label,recovered=recovered,recovery_time_s=recovery_time,max_joint_velocity_ratio=peak_v,max_joint_limit_excess_rad=peak_limit,max_target_rate_ratio=peak_target,physical_gate_passed=recovered and peak_v<=1.00001 and peak_limit<=1e-5 and peak_target<=1.00001));print(rows[-1],flush=True)
report=dict(source='https://github.com/lu-yidan/FR-Net',checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),joint_order=names,kp=30,kd=.8,action_scale=.25,action_clip=10,control_dt=.02,physics_dt=.005,scope='MuJoCo sim-to-sim screening, not native Isaac verification; native fixed PD nominal, no gain randomization',criterion='after initial fallen contact, upright height>.24 and 3 foot support, speed<.2 angular<.5 and no nonfoot ground contact continuously2s; full rollout joint/target gates',trials=rows,teacher_qualified=all(x['physical_gate_passed'] for x in rows),capability_qualified=False)
p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2));print('TEACHER_SCREEN',report['teacher_qualified'],flush=True)
