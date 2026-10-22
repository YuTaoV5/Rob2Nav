"""Record upstream policy inference with real joint states; no hardware control."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import imageio.v2 as imageio
from ..assets import source


def scene_data(model,joint_names,policy):
    geometry=[]
    for i in range(model.ngeom):
        if model.geom_bodyid[i]!=0 or int(model.geom_type[i]) not in (0,5,6):continue
        geometry.append({'type':int(model.geom_type[i]),'size':model.geom_size[i].tolist(),
            'position':model.geom_pos[i].tolist(),'quaternion':model.geom_quat[i].tolist()})
    return dict(geometries=geometry,joint_names=joint_names,backend='MuJoCo · '+policy,
        role='上游权重本地推理；未完成独立能力验收',quaternion_order='wxyz')


def record(out,model,data,step,names,qids,seconds,policy,entry,seed,contract=None):
    out.mkdir(parents=True,exist_ok=False)
    if contract is not None:(out/'io.json').write_text(json.dumps(contract,indent=2))
    (out/'scene.json').write_text(json.dumps(scene_data(model,names,policy)))
    renderer=mujoco.Renderer(model,height=480,width=640);camera=mujoco.MjvCamera();camera.distance=2.5;camera.elevation=-18;camera.azimuth=135
    initial=data.qpos[:3].copy();trace=[];minimum_upright=1.;maximum_height=float(initial[2]);next_frame=0.;rotation_travel=np.zeros(3);last_time=float(data.time)
    try:
        with (out/'replay.jsonl').open('w',buffering=1) as stream,imageio.get_writer(out/'episode_0.mp4',fps=25,macro_block_size=1) as video:
            while data.time<seconds-1e-6:
                step()
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():raise FloatingPointError('Nonfinite simulation state')
                q=data.qpos[3:7];upright=float(1-2*(q[1]**2+q[2]**2));minimum_upright=min(minimum_upright,upright);maximum_height=max(maximum_height,float(data.qpos[2]))
                rotation_travel+=np.abs(data.qvel[3:6])*(float(data.time)-last_time);last_time=float(data.time)
                frame=dict(time=float(data.time),position=data.qpos[:3].tolist(),quaternion=q.tolist(),joints=data.qpos[qids].tolist(),upright=upright)
                trace.append(frame);stream.write(json.dumps(frame)+'\n')
                if data.time+1e-6>=next_frame:
                    camera.lookat[:]=data.qpos[:3];renderer.update_scene(data,camera=camera);video.append_data(renderer.render());next_frame+=.04
            metrics={'distance_x_m':float(data.qpos[0]-initial[0]),'max_root_height_m':maximum_height,'minimum_upright':minimum_upright,
                'terminal_upright':trace[-1]['upright'],'seconds':float(data.time),'frames':len(trace),'integrated_abs_angular_motion_rad':rotation_travel.tolist()}
            (out/'trace.json').write_text(json.dumps(trace))
            (out/'summary.json').write_text(json.dumps(dict(args={'kind':policy,'seed':seed},backend='MuJoCo',successes=0,total=1,episodes=[],
                diagnostic_only='Inference recording only; no validated skill-pass definition',metrics=metrics,source_commit=entry['commit'],
                success_definition='Not scored: inspect trajectory, clip tracking and independent repeat trials before claiming capability.',
                note='Policy executes in simulation only; no motor bridge or hardware connection.'),indent=2))
    finally:renderer.close()


def pie(root,out,scene,seconds,seed):
    folder,entry=source(root,'pie');sys.path.insert(0,str(folder))
    module=importlib.import_module('deploy.pie.sim2sim.go2_pie_sim2sim')
    model,bindings=module.build_model('flat' if scene=='empty' else 'stairs')
    data=module.initialize_data(model,bindings);module.validate_runtime_contract(model,bindings,data)
    policy=module.PieOnnxPolicy(folder/entry['checkpoint'],provider='cpu');depth=module.DepthCamera(model,bindings.camera_id)
    runtime=module.Sim2SimRuntime(model,bindings,data,policy,depth,module.VelocityCommand(.5,0.))
    contract=dict(robot='go2',policy='pie',joints=list(module.ACTUATED_JOINT_NAMES),
        observation={'proprio':45,'history':450,'depth':[2,60,86],'recurrent':128},action_dim=12,
        policy_hz=50,physics_hz=1/model.opt.timestep,kp=module.JOINT_STIFFNESS.tolist(),kd=module.JOINT_DAMPING.tolist(),
        source_commit=entry['commit'])
    try:record(out,model,data,runtime.step,list(module.ACTUATED_JOINT_NAMES),bindings.joint_qpos_adr,seconds,'PIE',entry,seed,contract)
    finally:depth.close()


def lsd(root,out,choice,seconds,seed,physics_hz=None):
    folder,entry=source(root,'lsd');sys.path.insert(0,str(folder))
    import torch
    from src.scripts.rl_sdk import RLPolicy
    from src.scripts.rl_data import RobotState,Control
    torch.set_num_threads(4)
    names={'lsd_loco':'go2_loco','lsd_jump':'go2_jump','lsd_backflip':'go2_back_filp','lsd_sideflip':'go2_silde_filp'}
    skill=RLPolicy('unitree_rl_lab/go2/'+names[choice]);skill.model=skill.models[0]
    loco=skill if choice=='lsd_loco' else RLPolicy('unitree_rl_lab/go2/go2_loco')
    loco.model=loco.models[0]
    # MuJoCo 3.2 mutates an instance; newer releases return a parsed spec.
    spec=mujoco.MjSpec()
    loaded=spec.from_file(str(folder/'robot_description/mjcf/go2/go2.xml'))
    if loaded is not None:spec=loaded
    spec.worldbody.add_geom(name='rob2nav_flat_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.1],friction=[1.,.005,.0001],rgba=[.65,.75,.9,1.])
    spec.worldbody.add_light(pos=[0,0,5],dir=[0,0,-1])
    model=spec.compile();model.opt.timestep=1./physics_hz if physics_hz else float(skill.params.dt);data=mujoco.MjData(model)
    physical_names=skill.params.joint_controller_names
    policy_names=[physical_names[i] for i in skill.params.joint_mapping]
    contract=dict(robot='go2',policy=choice,joints=policy_names,
        observation='45 proprio' if choice=='lsd_loco' else '69 motion-reference conditioned',action_dim=12,
        policy_hz=1/(float(skill.params.dt)*int(skill.params.decimation)),physics_hz=1/model.opt.timestep,
        kp=skill.params.rl_kp.tolist(),kd=skill.params.rl_kd.tolist(),source_commit=entry['commit'],
        prelude='0.8s fixed PD stance, not learned recovery')
    joints=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in policy_names])
    if (joints<0).any() or len(set(joints))!=12:raise ValueError('Go2 joint mapping mismatch')
    qids=model.jnt_qposadr[joints];vids=model.jnt_dofadr[joints]
    aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j)[0] for j in joints])
    data.qpos[:3]=[0.,0.,.32];data.qpos[3:7]=[1.,0.,0.,0.];data.qpos[qids]=loco.params.default_dof_pos.numpy()[0]
    mujoco.mj_forward(model,data)
    gyro_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'imu_gyro');gyro_adr=model.sensor_adr[gyro_id]
    state=RobotState();control=Control();control.x=.3 if choice=='lsd_loco' else 0.
    def sense():
        state.imu.quaternion[:]=data.qpos[3:7].tolist();state.imu.gyroscope[:]=data.sensordata[gyro_adr:gyro_adr+3].tolist()
        for i in range(12):state.motor_state.q[i]=float(data.qpos[qids[i]]);state.motor_state.dq[i]=float(data.qvel[vids[i]])
    # A PD stance prelude mirrors the deployment get-up state. It is not learned recovery.
    target=loco.params.default_dof_pos.numpy()[0];kp=loco.params.fixed_kp.numpy()[0];kd=loco.params.fixed_kd.numpy()[0]
    for _ in range(round(.8/model.opt.timestep)):
        data.ctrl[aids]=kp*(target-data.qpos[qids])-kd*data.qvel[vids];mujoco.mj_step(model,data)
    data.time=0.;sense();skill.reset(state);active=skill
    def step():
        nonlocal active
        sense()
        if active is skill and skill.motion_loader is not None and skill.rl_time>=skill.motion_loader.duration:
            active=loco;active.reset(state)
        with torch.inference_mode():active.run_inference(state,control)
        target=active.output_dof_pos.numpy()[0];velocity=active.output_dof_vel.numpy()[0]
        if not np.isfinite(target).all():raise FloatingPointError('Nonfinite policy action')
        for _ in range(round(float(active.params.dt)*int(active.params.decimation)/model.opt.timestep)):
            data.ctrl[aids]=active.params.rl_kp.numpy()[0]*(target-data.qpos[qids])+active.params.rl_kd.numpy()[0]*(velocity-data.qvel[vids])
            mujoco.mj_step(model,data)
    record(out,model,data,step,policy_names,qids,seconds,choice,entry,seed,contract)


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=['pie','lsd_loco','lsd_jump','lsd_backflip','lsd_sideflip'],required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--physics-hz',type=int,choices=[200,500,1000])
    p.add_argument('--scene',choices=['empty','stairs'],default='stairs');p.add_argument('--seconds',type=float,default=10);p.add_argument('--seed',type=int,default=801);a=p.parse_args()
    if not 1<=a.seconds<=120:raise ValueError('Invalid duration')
    np.random.seed(a.seed)
    if a.policy=='pie':pie(Path(a.root),Path(a.output),a.scene,a.seconds,a.seed)
    else:lsd(Path(a.root),Path(a.output),a.policy,a.seconds,a.seed,a.physics_hz)

if __name__=='__main__':main()
