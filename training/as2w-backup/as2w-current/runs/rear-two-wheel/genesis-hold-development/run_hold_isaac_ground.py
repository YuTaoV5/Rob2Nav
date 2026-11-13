import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device='cpu').app
try:
    import torch
    import isaaclab.sim as sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    urdf, pose, effort = map(Path, sys.argv[1:4])
    best = json.loads(pose.read_text())['best']
    limits = json.loads(effort.read_text())['axes']
    names = list(best['joints_rad'])
    context = sim.SimulationContext(sim.SimulationCfg(dt=.005, device='cpu'))
    sim.GroundPlaneCfg().func('/World/Ground', sim.GroundPlaneCfg())
    spawn = sim.UrdfFileCfg(
        asset_path=str(urdf), fix_base=False, merge_fixed_joints=False,
        joint_drive=sim.UrdfFileCfg.JointDriveCfg(
            gains=sim.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
        activate_contact_sensors=True,
    )
    robot = Articulation(ArticulationCfg(
        prim_path='/World/Robot', spawn=spawn, init_state=ArticulationCfg.InitialStateCfg(joint_pos=best['joints_rad']),
        actuators={'effort': ImplicitActuatorCfg(
            joint_names_expr=['.*'], stiffness=0., damping=0.,
            effort_limit_sim={name:limits[name]['cap_nm'] for name in names})},
    ))
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Usd
    ground=context.stage.GetPrimAtPath('/World/Ground')
    plane=next(str(prim.GetPath()) for prim in Usd.PrimRange(ground) if prim.GetTypeName()=='Plane')
    sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Robot/.*',update_period=0.,
                                        filter_prim_paths_expr=[plane]))
    context.reset()
    rear_sensor=[sensor.body_names.index(n) for n in ('RL_foot','RR_foot')]
    other_sensor=[i for i in range(len(sensor.body_names)) if i not in rear_sensor]
    ids = [robot.joint_names.index(name) for name in names]
    assert len(set(ids)) == len(robot.joint_names) == 16
    pitch = torch.tensor(best['pitch_rad']/2)
    root = robot.data.default_root_state.clone()
    root[:,2] = best['base_height_m']+.002
    root[:,3:7] = torch.tensor([pitch.cos(),0.,pitch.sin(),0.])
    robot.write_root_state_to_sim(root)
    q = torch.tensor([list(best['joints_rad'].values())])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    import hashlib
    actor=ReferenceEvaluator(sys.argv[4])
    ref=q[0].numpy().copy()
    leg=[i for i,n in enumerate(names) if not n.endswith('_foot')]
    wheels=[i for i,n in enumerate(names) if n.endswith('_foot')]
    rear=[11,15]
    cap=np.array([limits[n]['cap_nm'] for n in names])
    ff=np.array([limits[n]['torque_nm'] for n in names])
    bounds=robot.data.joint_pos_limits[0,ids].numpy()[leg]
    anchor=np.array([dict(hip=0.,thigh=.8,calf=-1.6)[names[i].split('_')[-1]] for i in leg])
    lo,hi=(.9*bounds+.1*anchor[:,None]).T
    action=np.zeros(14)
    pending=np.zeros(16)
    filtered=0.
    peak=0.
    reason=None
    trace=[]
    first_unload=None
    first_other=None
    for tick in range(2000):
        q=robot.data.joint_pos[0,ids].numpy()
        dq=robot.data.joint_vel[0,ids].numpy()
        x=robot.data.root_link_pos_w[0].numpy()
        w,a,b,c=robot.data.root_link_quat_w[0].numpy()
        omega=robot.data.root_link_ang_vel_w[0].numpy()
        v=robot.data.root_link_lin_vel_w[0].numpy()
        err=np.arcsin(np.clip(2*(w*b-c*a),-1,1))-best['pitch_rad']
        roll=np.arctan2(2*(w*a+b*c),1-2*(a*a+b*b))
        if (q[leg]<lo).any() or (q[leg]>hi).any():
            reason='joint_angle'
        elif abs(err)>.25 or x[2]<.3:
            reason='base_pose'
        elif (np.abs(dq[wheels])>98.47).any():
            reason='wheel_speed'
        forces=sensor.data.force_matrix_w[0].sum(dim=1)
        loads=forces[rear_sensor,2].tolist()
        other=[sensor.body_names[i] for i in other_sensor if forces[i].norm()>1.]
        if tick>=52 and min(loads)<10 and first_unload is None:
            first_unload=tick*.005
        if tick>0 and other and first_other is None:
            first_other=tick*.005
        trace.append(dict(loads_n=loads,other_contact=other,t=tick*.005, height=float(x[2]), pitch_error=float(err), roll=float(roll), vx=float(v[0]), wheel_rad_s=dq[wheels].tolist(), applied_nm=pending.tolist()))
        if reason:
            break
        if tick%4==0:
            obs=np.r_[q[leg]-ref[leg],dq*.1,err,roll,filtered,x[0],omega*.25,v,action,pending/cap]
            action=np.clip(actor.run(None,{'policy':obs.astype(np.float32)[None,:]})[0][0],-1,1)
            if not np.isfinite(action).all():
                raise ValueError('Nonfinite action')
            target=np.clip(ref[leg]+.08*action[:12],lo,hi)
        filtered+=(.005/.105)*(v[0]-filtered)
        tau=np.zeros(16)
        tau[leg]=ff[leg]+55*(target-q[leg])-2*dq[leg]
        hip=np.clip(32*roll+4*omega[0],-6,6)
        thigh=np.clip(-12*roll-2*omega[0],-6,6)
        tau[[8,12]]-=hip
        tau[[9,13]]+=[thigh,-thigh]
        tau[rear]=40*(err+.04)+6*omega[1]+10*filtered+1.5*x[0]+2*action[12:]
        robot.set_joint_effort_target(torch.tensor(pending[None,:],dtype=torch.float32),joint_ids=ids)
        pending=np.clip(tau,-cap,cap)
        robot.write_data_to_sim()
        context.step(render=False)
        robot.update(.005)
        sensor.update(.005)
        peak=max(peak,abs(float(v[0])))
    result=dict(scope='Isaac Lab nominal diagnostic; ground-filtered contact force telemetry',physics_steps=tick+int(reason is None),
                peak_speed_m_s=peak,first_unload_s=first_unload,first_other_contact_s=first_other,first_guard=reason,onnx_sha256=hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest())
    output=Path(sys.argv[5])
    output.write_text(json.dumps(result))
    output.with_suffix(".trace.json").write_text(json.dumps(trace))
    print('ISAAC_HOLD',json.dumps(result),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    app.close()
    raise
else:
    app.close()
