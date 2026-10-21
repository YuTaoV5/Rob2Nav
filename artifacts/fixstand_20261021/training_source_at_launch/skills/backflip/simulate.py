"""Run the pinned native 60-input backflip policy with explicit integration refinement."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from rob2nav.mujoco_skill import assert_physics_step

PIN='954aa61ceb62c515689677f44a21e32109f6b62d'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--onnx',required=True,type=Path)
    p.add_argument('--physics-hz',type=int,choices=[200,500,1000,2000],default=1000)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--initial-joint-noise',type=float,default=.002)
    p.add_argument('--video',action='store_true')
    p.add_argument('--fixstand-roundtrip',action='store_true')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if not 0<a.seconds<=60 or not 0<=a.initial_joint_noise<=.05:raise ValueError('Invalid evaluation duration/noise')
    source=a.source.resolve();policy=a.onnx.resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('Wrong upstream commit')
    file=source/'mujoco/go2/play_onnx.py'
    pristine=subprocess.check_output(['git','-C',str(source),'show',PIN+':mujoco/go2/play_onnx.py'])
    if hashlib.sha256(file.read_bytes()).digest()!=hashlib.sha256(pristine).digest():
        raise ValueError('Upstream playback source was modified')
    spec=importlib.util.spec_from_file_location('_rob2nav_backflip_playback',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.SIM_DT=1./a.physics_hz;module.DECIMATION=a.physics_hz//50;module.POLICY_DT=.02
    a.output.mkdir(parents=True,exist_ok=False)
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(source/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.attrib['name'] in module.JOINT_NAMES}
    rng=np.random.default_rng(a.seed);resets=0;trace=[];rotation=0.;angular_travel=0.
    max_speed=0.;max_excess=0.;air=0.;max_air=0.;bad=False;minimum_up=1.;max_height=0.
    original_reset=module.reset_robot;original_step=mujoco.mj_step
    writer=None;renderer=None;next_trace=0.;next_frame=0.
    def reset(model,data,qids):
        nonlocal resets
        resets+=1
        if resets>1:raise RuntimeError('Unexpected physical reset during policy rollout')
        original_reset(model,data,qids)
        data.qpos[qids]+=rng.uniform(-a.initial_joint_noise,a.initial_joint_noise,12)
        mujoco.mj_forward(model,data)
    def step(model,data,*args,**kwargs):
        nonlocal rotation,angular_travel,max_speed,max_excess,air,max_air,bad,minimum_up,max_height,writer,renderer,next_trace,next_frame
        before_step=float(data.time)
        result=original_step(model,data,*args,**kwargs)
        assert_physics_step(data,before_step,model.opt.timestep)
        qids,dids=module.joint_addresses(model)
        velocity=np.array([float(limits[n]['velocity']) for n in module.JOINT_NAMES])
        rotation+=float(data.qvel[4])*model.opt.timestep
        angular_travel+=float(np.linalg.norm(data.qvel[3:6]))*model.opt.timestep
        max_speed=max(max_speed,float(np.max(np.abs(data.qvel[dids])/velocity)))
        ids=np.array([model.joint(n).id for n in module.JOINT_NAMES])
        max_excess=max(max_excess,float(max(0,np.max(np.maximum(data.qpos[qids]-model.jnt_range[ids,1],model.jnt_range[ids,0]-data.qpos[qids])))))
        ground={i for i in range(model.ngeom) if model.geom_bodyid[i]==0};contacts=[]
        for c in data.contact:
            x,y=int(c.geom1),int(c.geom2)
            if x in ground and y not in ground:contacts.append(model.body(int(model.geom_bodyid[y])).name)
            elif y in ground and x not in ground:contacts.append(model.body(int(model.geom_bodyid[x])).name)
        air=0. if contacts else air+model.opt.timestep;max_air=max(max_air,air)
        bad|=any('foot' not in n and 'calf' not in n for n in contacts)
        up=float(data.xmat[model.body('base').id].reshape(3,3)[2,2]);minimum_up=min(minimum_up,up);max_height=max(max_height,float(data.qpos[2]))
        if data.time>=next_trace:
            trace.append(dict(time=float(data.time),position=data.qpos[:3].tolist(),quaternion=data.qpos[3:7].tolist(),
                joints=data.qpos[qids].tolist(),upright=up,speed=float(np.linalg.norm(data.qvel[:2])),angular_speed=float(np.linalg.norm(data.qvel[3:6]))))
            from skills.fixstand import POSE
            trace[-1].update(standard_pose_error=float(np.max(abs(data.qpos[qids]-[POSE[n] for n in module.JOINT_NAMES]))),
                max_joint_speed=float(np.max(abs(data.qvel[dids]))),
                all_feet_ground=all(any(n.startswith(leg+'_') for n in contacts) for leg in ('FR','FL','RR','RL')))
            next_trace+=.02
        if a.video and data.time>=next_frame:
            if writer is None:
                import imageio.v2 as imageio
                writer=imageio.get_writer(a.output/'episode_0.mp4',fps=25,macro_block_size=1)
                renderer=mujoco.Renderer(model,height=480,width=640)
            camera=mujoco.MjvCamera();camera.distance=2.3;camera.azimuth=130;camera.elevation=-15;camera.lookat[:]=data.qpos[:3]
            renderer.update_scene(data,camera=camera);writer.append_data(renderer.render());next_frame+=.04
        return result
    argv=sys.argv
    module.reset_robot=reset;mujoco.mj_step=step
    sys.argv=[str(file),'--onnx',str(policy),'--headless','--no-realtime','--start','--duration',str(a.seconds)]
    try:
        with (a.output/'source_playback.log').open('w') as log,redirect_stdout(log):module.main()
    finally:
        module.reset_robot=original_reset;mujoco.mj_step=original_step;sys.argv=argv
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    if not trace or resets!=1:raise RuntimeError('Missing dynamics rollout or unexpected initialization')
    tail=trace[-100:];dominance=abs(rotation)/max(angular_travel,1e-8)
    success=bool(5.5<abs(rotation)<7.1 and dominance>.75 and max_air>=.08 and max_height>.45 and minimum_up<-.8 and not bad and
        all(x['upright']>.9 and x['position'][2]>.22 and x['speed']<.25 and x['angular_speed']<1. for x in tail))
    metrics=dict(body_y_rotation_rad=rotation,axis_dominance=dominance,airtime_s=max_air,max_height=max_height,
        terminal_upright=trace[-1]['upright'],unexpected_body_ground_contact=bad,max_joint_speed_ratio=max_speed,max_joint_limit_excess_rad=max_excess)
    if a.fixstand_roundtrip:
        from rob2nav.fixstand_validation import assess
        metrics['action_completed']=bool(success);metrics['fixstand']=assess(trace,initial_end=.35)
        success=bool(success and metrics['fixstand']['passed'])
    summary=dict(args=dict(kind='backflip',seed=a.seed,physics_hz=a.physics_hz,initial_joint_noise=a.initial_joint_noise),backend='MuJoCo',
        role='locally trained Go2 backflip; pinned native observation/controller',successes=int(success),total=1,episodes=[dict(success=success,**metrics)],metrics=metrics,
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),source_commit=PIN,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy_hz=50,physics_hz=a.physics_hz,
        joint_names=list(module.JOINT_NAMES),physical_resets=resets,root_state_writes='initialization only; no in-flight reset/stabilizing force',
        hardware_constraints_pass=max_speed<=1. and max_excess<=.001,hardware_connected=False)
    (a.output/'trace.json').write_text(json.dumps(trace))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
