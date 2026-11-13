import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco as mj
import numpy as np

urdf, pose = map(Path, sys.argv[1:3])
tree = ET.parse(urdf).getroot()
compiler = ET.SubElement(ET.SubElement(tree, 'mujoco'), 'compiler')
compiler.set('fusestatic', 'false')
root_link = ET.Element('link', name='world')
tree.insert(0, root_link)
joint = ET.SubElement(tree,'joint',name='floating_base',type='floating')
ET.SubElement(joint,'parent',link='world')
ET.SubElement(joint,'child',link='base_link')
assets = {file.name:file.read_bytes() for file in (urdf.parent/'meshes').glob('*.STL')}
model = mj.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'),assets)
best = json.loads(pose.read_text())['best']
data = mj.MjData(model)
base = model.joint('floating_base')
assert base.type[0] == mj.mjtJoint.mjJNT_FREE and model.nv == 22
offset=int(base.qposadr[0])
pitch=best['pitch_rad']
data.qpos[offset:offset+7]=[0,0,best['base_height_m']+.002,np.cos(pitch/2),0,np.sin(pitch/2),0]
axes = {}
for name, angle in best['joints_rad'].items():
    axis=model.joint(name)
    qpos=int(axis.qposadr[0])
    dof=int(axis.dofadr[0])
    data.qpos[qpos]=angle
    axes[name]=dict(qpos=qpos,dof=dof)
mj.mj_forward(model,data)
assert len({axis['dof'] for axis in axes.values()})==16
assert np.isfinite(data.qfrc_bias).all()
print(json.dumps(dict(scope='floating-base asset and axis import probe',mass_kg=float(model.body_mass.sum()),axes=axes,
                      rear_links={name:model.body(name).id for name in ('RL_foot','RR_foot')})))
import onnxruntime as ort
config=json.loads(Path('static-load.json').read_text())['axes']
names=list(axes)
leg=[i for i,n in enumerate(names) if not n.endswith('_foot')]
rear=[names.index(n) for n in ('RL_foot','RR_foot')]
qi=[axes[n]['qpos'] for n in names]
vi=[axes[n]['dof'] for n in names]
ref=data.qpos[qi].copy()
cap=np.array([config[n]['cap_nm'] for n in names])
ff=np.array([config[n]['torque_nm'] for n in names])
anchor=np.array([dict(hip=0.,thigh=.8,calf=-1.6)[names[i].split('_')[-1]] for i in leg])
limits=np.array([model.joint(names[i]).range for i in leg])
lo,hi=(.1*anchor[:,None]+.9*limits).T
# Recompile the imported robot with an explicit contact plane.
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    xml=Path(tmp)/'imported.xml'
    mj.mj_saveLastXML(str(xml),model)
    world=ET.parse(xml).getroot()
    ET.SubElement(world.find('worldbody'),'geom',name='ground',type='plane',size='0 0 .1')
    model=mj.MjModel.from_xml_string(ET.tostring(world,encoding='unicode'),assets)
q_init=data.qpos.copy()
data=mj.MjData(model)
data.qpos[:]=q_init
model.opt.timestep=.005
mj.mj_forward(model,data)
floor=model.geom('ground').id
rear_bodies={model.body(names[i]).id for i in rear}
opts=ort.SessionOptions()
opts.intra_op_num_threads=1
session=ort.InferenceSession(sys.argv[3],sess_options=opts,providers=['CPUExecutionProvider'])
action=np.zeros(14)
pending=np.zeros(16)
filtered=0.
peak=0.
minimum_load=np.inf
first_unloaded=None
violations=[]
wheel_axes=[i for i,n in enumerate(names) if n.endswith("_foot")]
peak_wheel=np.zeros(4)
trace=[]
body_names={i:model.body(i).name for i in rear_bodies}
for tick in range(2000):
    q,dq=data.qpos[qi],data.qvel[vi]
    x=data.qpos[:3]
    w,a,b,c=data.qpos[3:7]
    err=np.arcsin(np.clip(2*(w*b-c*a),-1,1))-pitch
    roll=np.arctan2(2*(w*a+b*c),1-2*(a*a+b*b))
    v=data.qvel[:3]
    omega=data.xmat[model.body('base_link').id].reshape(3,3)@data.qvel[3:6]
    contact_bodies={int(model.geom_bodyid[g2 if g1==floor else g1])
                    for g1,g2 in data.contact.geom if floor in (g1,g2)}
    loads=dict.fromkeys(rear_bodies,0.)
    slip=dict.fromkeys(rear_bodies,None)
    for index,(g1,g2) in enumerate(data.contact.geom):
        if floor in (g1,g2):
            body=int(model.geom_bodyid[g2 if g1==floor else g1])
            if body in loads:
                force=np.zeros(6)
                mj.mj_contactForce(model,data,index,force)
                loads[body]+=float(force[0])
                if force[0]>1:
                    jac=np.zeros((3,model.nv))
                    mj.mj_jac(model,data,jac,None,data.contact[index].pos,body)
                    velocity=jac@data.qvel
                    tangential=float(np.linalg.norm(velocity[:2]))
                    slip[body]=max(slip[body] or 0.,tangential)
    if tick>=52:
        minimum_load=min(minimum_load,*loads.values())
        if min(loads.values())<10 and first_unloaded is None:
            first_unloaded=data.time
    peak_wheel=np.maximum(peak_wheel,np.abs(dq[wheel_axes]))
    violations=[name for name,bad in (
        ('non_rear_contact',bool(contact_bodies-rear_bodies)),
        ('joint_angle',np.any(q[leg]<lo) or np.any(q[leg]>hi)),
        ('pitch',abs(err)>.25),('base_height',x[2]<.3),
        ('wheel_speed',np.any(np.abs(dq[wheel_axes])>98.47))) if bad]
    trace.append(dict(t=float(data.time),vx=float(v[0]),roll=float(roll),
                      load={body_names[i]:loads[i] for i in rear_bodies},
                      slip_m_s={body_names[i]:slip[i] for i in rear_bodies},
                      wheel_rad_s={names[i]:float(dq[i]) for i in rear},
                      queued_nm={names[i]:float(pending[i]) for i in rear}))
    if violations:
        break
    if tick%4==0:
        obs=np.r_[q[leg]-ref[leg],dq*.1,err,roll,filtered,x[0],omega*.25,v,action,pending/cap]
        action=np.clip(session.run(None,{'policy':obs.astype(np.float32)[None,:]})[0][0],-1,1)
        target=np.clip(ref[leg]+.08*action[:12],lo,hi)
    filtered+=(.005/.105)*(v[0]-filtered)
    tau=np.zeros(16)
    tau[leg]=ff[leg]+55*(target-q[leg])-2*dq[leg]
    hip=np.clip(32*roll+4*omega[0],-6,6)
    thigh=np.clip(-12*roll-2*omega[0],-6,6)
    tau[[8,12]]-=hip
    tau[[9,13]]+=np.array([thigh,-thigh])
    tau[rear]=40*(err+.04)+6*omega[1]+10*filtered+1.5*x[0]+2*action[12:]
    data.qfrc_applied[vi]=pending
    pending=np.clip(tau,-cap,cap)
    peak=max(peak,abs(v[0]))
    mj.mj_step(model,data)
print(json.dumps(dict(scope='nominal MuJoCo diagnostic',held_s=data.time,peak_speed=peak,first_load_below_10n_s=first_unloaded,minimum_normal_load_n=minimum_load,violations=violations,
                      ground_links=[model.body(i).name for i in sorted(contact_bodies)],
                      last_wheel_rad_s=dict(zip((names[i] for i in wheel_axes),dq[wheel_axes].tolist())),
                      peak_abs_wheel_rad_s=dict(zip((names[i] for i in wheel_axes),peak_wheel.tolist())))))

if len(sys.argv)>4:
    Path(sys.argv[4]).write_text(json.dumps(trace))
