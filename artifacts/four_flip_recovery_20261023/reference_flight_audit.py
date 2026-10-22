"""Kinematic advisory only: test free-flight reference consistency, never replay as physics."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np

root=Path.cwd(); source=root.parent/'Rob2Nav-upstreams/GO2_backflip'
path=root/'artifacts/web_continuity_iteration_20261023/inputs/side_v12/sideflip.npz'
meta=json.loads(path.with_suffix('.json').read_text()); ref=np.load(path,allow_pickle=False)
model=mujoco.MjModel.from_xml_path(str(source/'mujoco/go2/go2_isaac.xml'));data=mujoco.MjData(model)
base=model.body('base').id; anchor=meta['body_names'].index('base')
qids=[model.jnt_qposadr[model.joint(n).id] for n in meta['joint_names']]
vids=[model.jnt_dofadr[model.joint(n).id] for n in meta['joint_names']]
feet=[meta['body_names'].index(n+'_foot') for n in ['FR','FL','RR','RL']]
floor=set(np.flatnonzero(model.geom_bodyid==0));com=[];momentum=[];free=[]
dt=1./float(ref['fps'][0])
for i in range(len(ref['joint_pos'])):
    data.qpos[:3]=ref['body_pos_w'][i,anchor]
    quat=np.asarray(ref['body_quat_w'][i,anchor],dtype=float);data.qpos[3:7]=quat/np.linalg.norm(quat)
    data.qpos[qids]=ref['joint_pos'][i]
    data.qvel[:3]=ref['body_lin_vel_w'][i,anchor]
    mujoco.mj_forward(model,data)
    data.qvel[3:6]=data.xmat[base].reshape(3,3).T@ref['body_ang_vel_w'][i,anchor]
    data.qvel[vids]=ref['joint_vel'][i]
    mujoco.mj_forward(model,data);mujoco.mj_subtreeVel(model,data)
    com.append(data.subtree_com[base].copy());momentum.append(data.subtree_angmom[base].copy())
    floor_contact=any((c.geom1 in floor or c.geom2 in floor) and c.dist<=.002 for c in data.contact)
    free.append(not floor_contact and bool(np.all(ref['body_pos_w'][i,feet,2]>.05)))
com=np.asarray(com);momentum=np.asarray(momentum);free=np.asarray(free)
# Exclude contact transitions and every finite-difference stencil crossing one.
mask=np.zeros(len(free),dtype=bool)
for i in range(2,len(free)-2):mask[i]=bool(free[i-2:i+3].all())
acc=np.gradient(np.gradient(com,dt,axis=0),dt,axis=0)
hdot=np.gradient(momentum,dt,axis=0)
residual=np.linalg.norm(acc-model.opt.gravity,axis=1)
rows=[dict(time_s=i*dt,com_acceleration_m_s2=acc[i].tolist(),
    gravity_residual_m_s2=float(residual[i]),angular_momentum=momentum[i].tolist(),
    angular_momentum_derivative_Nm=hdot[i].tolist()) for i in np.flatnonzero(mask)]
report=dict(scope='Kinematic advisory, not simulation or skill acceptance; FK state writes are diagnostic only',
    mujoco_version=mujoco.__version__,
    mjcf_sha256=hashlib.sha256((source/'mujoco/go2/go2_isaac.xml').read_bytes()).hexdigest(),
    diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    method='Whole-robot COM and angular momentum; no floor contact and all foot centers >5cm; two-frame interior stencil',
    reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fps=1/dt,
    free_reference_frames=int(free.sum()),interior_frames=int(mask.sum()),
    gravity_m_s2=model.opt.gravity.tolist(),rows=rows,
    qualifications='Finite differences and reference velocities have discretization error; this is not an infeasibility proof')
if mask.any():report.update(max_com_gravity_residual_m_s2=float(residual[mask].max()),
    median_com_gravity_residual_m_s2=float(np.median(residual[mask])),
    max_angular_momentum_derivative_Nm=float(np.linalg.norm(hdot[mask],axis=1).max()))
# Same COM finite-difference check on recorded free-base physics, to estimate the
# sampling-method floor. This failed skill is not a positive skill benchmark.
tracepath=root/'outputs/studio_diagnostics/left_v12_final_damping_probe/kd_1/case/trace.json'
trace=json.loads(tracepath.read_text());actual_com=[];actual_free=[]
footids=[model.body(n+'_foot').id for n in ['FR','FL','RR','RL']]
for row in trace:
    data.qpos[:3]=row['position'];data.qpos[3:7]=row['quaternion'];data.qpos[qids]=row['joints']
    mujoco.mj_forward(model,data)
    actual_com.append(data.subtree_com[base].copy())
    touching=any((c.geom1 in floor or c.geom2 in floor) and c.dist<=.002 for c in data.contact)
    actual_free.append(not touching and bool(np.all(data.xpos[footids,2]>.05)))
actual_com=np.asarray(actual_com);actual_mask=np.zeros(len(trace),dtype=bool)
for i in range(2,len(trace)-2):actual_mask[i]=all(actual_free[i-2:i+3])
actual_acc=np.gradient(np.gradient(actual_com,dt,axis=0),dt,axis=0)
actual_residual=np.linalg.norm(actual_acc-model.opt.gravity,axis=1)
report['physical_trace_sampling_control']=dict(scope='Failed real physics skill, only free-flight COM sampling control',
    trace_sha256=hashlib.sha256(tracepath.read_bytes()).hexdigest(),interior_frames=int(actual_mask.sum()),
    median_com_gravity_residual_m_s2=float(np.median(actual_residual[actual_mask])) if actual_mask.any() else None,
    max_com_gravity_residual_m_s2=float(actual_residual[actual_mask].max()) if actual_mask.any() else None)
out=root/'artifacts/four_flip_recovery_20261023/reference_flight_audit.json'
out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
