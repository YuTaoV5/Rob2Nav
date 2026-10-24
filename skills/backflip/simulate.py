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


def main(extension=None):
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--onnx',required=True,type=Path)
    p.add_argument('--physics-hz',type=int,choices=[200,500,1000,2000],default=1000)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--initial-joint-noise',type=float,default=.002)
    p.add_argument('--initial-yaw-deg',type=float,default=0.)
    p.add_argument('--initial-state',type=Path,help='Read-only captured-entry diagnostic; applied only at initialization')
    p.add_argument('--direction',choices=['back','front'],default='back')
    p.add_argument('--entry-hold-diagnostic',type=float,help='Counterfactual entry residual limit; changes the trained contract and cannot qualify a policy')
    p.add_argument('--video',action='store_true')
    p.add_argument('--fixstand-roundtrip',action='store_true')
    p.add_argument('--diagnostics',action='store_true',help='Record read-only per-substep failure diagnostics')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if not 0<a.seconds<=60 or not 0<=a.initial_joint_noise<=.05:raise ValueError('Invalid evaluation duration/noise')
    if not np.isfinite(a.initial_yaw_deg) or abs(a.initial_yaw_deg)>180:raise ValueError('Invalid initial heading')
    if a.initial_state is not None and (a.initial_yaw_deg or a.initial_joint_noise):raise ValueError('Captured state requires zero additional yaw/noise')
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
    observed=None
    if a.initial_state is not None:
        from .initial_state import validate
        observed=validate(json.loads(a.initial_state.read_text()),module.JOINT_NAMES)
    export_file=policy.parent/'export.json'
    exported=json.loads(export_file.read_text()) if export_file.exists() else {}
    controller=None
    if 'control_contract' in exported:
        from .control import configure_playback
        controller=configure_playback(module,exported['control_contract'])
    contract=exported.get('fixstand_roundtrip') or {}
    if contract.get('action_parameterization')=='backflip_rebound_v1' and extension is None:
        raise ValueError('This is the single-flip evaluator; rebound requires contact-cycle execution and per-cycle acceptance')
    if a.direction!=contract.get('flip_direction','back'):raise ValueError('Requested direction does not match the trained actor contract')
    delayed=contract.get('action_parameterization') in ('backflip_entry_v2','backflip_entry_v3')
    if a.entry_hold_diagnostic is not None and (not delayed or not 0.<=a.entry_hold_diagnostic<=.4):
        raise ValueError('Entry counterfactual requires delayed-entry playback and [0,.4]')
    if extension is not None:extension.configure(module,exported,a)
    entry_policy_trace=[]
    if delayed:
        if hashlib.sha256(policy.read_bytes()).hexdigest()!=exported['policy_sha256']:
            raise ValueError('Backflip export hash mismatch')
        if contract['entry_seconds']!=1. or contract['release_seconds']!=[1.,1.2]:
            raise ValueError('Unsupported backflip entry contract')
        hold_scale=float(contract.get('entry_hold_scale',.4))
        if not 0.<=hold_scale<=.4:raise ValueError('Unsupported backflip hold scale')
        if a.entry_hold_diagnostic is not None:hold_scale=a.entry_hold_diagnostic
        module.PHASE_DURATION=3.
        original_phase=module.phase_features
        module.phase_features=lambda t:original_phase(min(t-1.,2.))
        original_load=module.load_onnx
        class EntrySession:
            def __init__(self,session):self.session=session;self.calls=0
            def run(self,*args,**kwargs):
                from skills.fixstand import backflip_entry_weight
                result=self.session.run(*args,**kwargs)
                time=self.calls*.02;weight=float(backflip_entry_weight(time));self.calls+=1
                raw=result[0].copy()
                result[0]=(1.-weight)*result[0]+weight*hold_scale*np.tanh(result[0])
                if a.diagnostics and time<=1.2:
                    entry_policy_trace.append(dict(time=time,raw_action=raw.reshape(-1).tolist(),
                        processed_action=result[0].reshape(-1).tolist()))
                return result
        def load(path):
            session,inp,out,name=original_load(path)
            return EntrySession(session),inp,out,name
        module.load_onnx=load
    a.output.mkdir(parents=True,exist_ok=False)
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(source/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.attrib['name'] in module.JOINT_NAMES}
    from rob2nav.motion_diagnostics import MotionDiagnostics
    diagnostics=MotionDiagnostics(module.JOINT_NAMES,[float(limits[n]['velocity']) for n in module.JOINT_NAMES],1) if a.diagnostics else None
    rng=np.random.default_rng(a.seed);resets=0;trace=[];rotation=0.;angular_travel=0.
    max_speed=0.;max_excess=0.;air=0.;max_air=0.;bad=False;minimum_up=1.;max_height=0.
    ground_reader=None
    original_reset=module.reset_robot;original_step=mujoco.mj_step
    writer=None;renderer=None;next_trace=0.;next_frame=0.
    def reset(model,data,qids):
        nonlocal resets
        resets+=1
        if resets>1:raise RuntimeError('Unexpected physical reset during policy rollout')
        original_reset(model,data,qids)
        if observed is not None:
            _,dids=module.joint_addresses(model)
            data.qpos[:3]=observed['root_position'];data.qpos[3:7]=observed['root_quaternion']
            data.qpos[qids]=observed['joint_position'];data.qvel[dids]=observed['joint_velocity']
            data.qvel[:3]=observed['root_linear_velocity_world'];data.qvel[3:6]=observed['root_angular_velocity_body']
        else:
            yaw=np.radians(a.initial_yaw_deg)
            data.qpos[3:7]=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            data.qpos[qids]+=rng.uniform(-a.initial_joint_noise,a.initial_joint_noise,12)
        mujoco.mj_forward(model,data)
    def step(model,data,*args,**kwargs):
        nonlocal rotation,angular_travel,max_speed,max_excess,air,max_air,bad,minimum_up,max_height,writer,renderer,next_trace,next_frame,ground_reader
        before_step=float(data.time)
        if extension is not None:extension.before_step()
        result=original_step(model,data,*args,**kwargs)
        assert_physics_step(data,before_step,model.opt.timestep)
        qids,dids=module.joint_addresses(model)
        if diagnostics is not None:diagnostics.sample(model,data,float(data.time),dids,data.qvel[3:6])
        velocity=np.array([float(limits[n]['velocity']) for n in module.JOINT_NAMES])
        rotation+=float(data.qvel[4])*model.opt.timestep
        angular_travel+=float(np.linalg.norm(data.qvel[3:6]))*model.opt.timestep
        max_speed=max(max_speed,float(np.max(np.abs(data.qvel[dids])/velocity)))
        ids=np.array([model.joint(n).id for n in module.JOINT_NAMES])
        max_excess=max(max_excess,float(max(0,np.max(np.maximum(data.qpos[qids]-model.jnt_range[ids,1],model.jnt_range[ids,0]-data.qpos[qids])))))
        if ground_reader is None:
            from rob2nav.ground_contacts import GroundContactReader
            ground_reader=GroundContactReader(model)
        contact_frame=ground_reader.sample(data);contacts=contact_frame.bodies;foot_forces=contact_frame.forces
        air=0. if contacts else air+model.opt.timestep;max_air=max(max_air,air)
        bad|=bool(contact_frame.unexpected)
        if extension is not None:extension.sample(model,data,contact_frame,qids,dids)
        up=float(data.xmat[model.body('base').id].reshape(3,3)[2,2]);minimum_up=min(minimum_up,up);max_height=max(max_height,float(data.qpos[2]))
        if data.time>=next_trace:
            trace.append(dict(time=float(data.time),position=data.qpos[:3].tolist(),quaternion=data.qpos[3:7].tolist(),
                joints=data.qpos[qids].tolist(),upright=up,speed=float(np.linalg.norm(data.qvel[:2])),angular_speed=float(np.linalg.norm(data.qvel[3:6]))))
            from skills.fixstand import POSE
            trace[-1].update(standard_pose_error=float(np.max(abs(data.qpos[qids]-[POSE[n] for n in module.JOINT_NAMES]))),
                max_joint_speed=float(np.max(abs(data.qvel[dids]))),
                all_feet_ground=all(value>4. for value in foot_forces.values()),foot_forces_N=foot_forces)
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
    from rob2nav.fixstand_validation import rotation_matches
    direction_pass=rotation_matches(rotation,a.direction);success=bool(success and direction_pass)
    metrics=dict(body_y_rotation_rad=rotation,axis_dominance=dominance,airtime_s=max_air,max_height=max_height,
        terminal_upright=trace[-1]['upright'],unexpected_body_ground_contact=bad,max_joint_speed_ratio=max_speed,max_joint_limit_excess_rad=max_excess)
    metrics.update(expected_direction=a.direction,direction_pass=direction_pass)
    metrics['ground_contact_rule']='foot spheres and contact points inside their own anatomical foot region'
    metrics['ground_contact_rule_sha256']=ground_reader.rule_sha256
    if extension is not None:
        extra=extension.report()
        metrics.update(extra)
        success=bool(extra['rebound_motion']['motion_pass'] and extra['target_continuity']['passed'] and extra['entry_quiet']['passed'] and max_speed<=1. and max_excess<=.001 and not bad)
    if a.fixstand_roundtrip:
        from rob2nav.fixstand_validation import assess
        metrics['action_completed']=bool(success);metrics['fixstand']=assess(trace,initial_end=.8 if delayed or extension is not None else .35)
        success=bool(success and metrics['fixstand']['passed'])
    if entry_policy_trace:(a.output/'entry_policy_trace.json').write_text(json.dumps(entry_policy_trace,indent=2)+'\n')
    summary=dict(args=dict(kind='backflip_rebound' if extension is not None else a.direction+'flip',seed=a.seed,physics_hz=a.physics_hz,initial_joint_noise=a.initial_joint_noise,initial_yaw_deg=a.initial_yaw_deg,
        initial_state_sha256=hashlib.sha256(a.initial_state.read_bytes()).hexdigest() if a.initial_state else None),backend='MuJoCo',
        role='locally trained Go2 '+a.direction+'flip; pinned native observation/controller',successes=int(success),total=1,episodes=[dict(success=success,**metrics)],metrics=metrics,
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),source_commit=PIN,source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        entry_hold_diagnostic=a.entry_hold_diagnostic,trained_execution_contract=a.entry_hold_diagnostic is None,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policy_hz=50,physics_hz=a.physics_hz,
        joint_names=list(module.JOINT_NAMES),physical_resets=resets,root_state_writes='initialization only; no in-flight reset/stabilizing force',
        hardware_constraints_pass=max_speed<=1. and max_excess<=.001,hardware_connected=False,
        control_contract=controller,control_parameter_source='exported_training_config' if controller else 'legacy_pinned_native_defaults')
    (a.output/'trace.json').write_text(json.dumps(trace))
    if diagnostics is not None:(a.output/'diagnostics.json').write_text(json.dumps(diagnostics.report(),indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
