"""Real free-base MuJoCo hot-switch integration, not kinematic state replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from .types import Snapshot,SDK_JOINTS
from .core import EFFORT
from .runner import UnifiedRuntime


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wire-mode',choices=['torque','hybrid_pd'],default='hybrid_pd');p.add_argument('--asset',choices=['robogauge','source'],default='robogauge');p.add_argument('--atomic-cycle',action='store_true')
    p.add_argument('--seed',type=int,default=9201);p.add_argument('--video',action='store_true')
    p.add_argument('--sequence',type=Path,help='Use the same state-driven sequence component as the device service')
    p.add_argument('--source-model',type=Path,help='Exact Go2 MuJoCo source XML; defaults to packaged deployment/models/go2_isaac.xml')
    p.add_argument('--initial-yaw-deg',type=float,default=0.)
    p.add_argument('--max-seconds',type=float,default=80.,help='Whole sequence budget; individual entry/exit deadlines are unchanged')
    a=p.parse_args()
    if not np.isfinite(a.initial_yaw_deg) or abs(a.initial_yaw_deg)>180:p.error('Invalid initial yaw')
    if not np.isfinite(a.max_seconds) or not 1<=a.max_seconds<=300:p.error('Invalid simulation duration')
    if a.initial_yaw_deg and a.asset!='source':p.error('Initial yaw variation currently requires the source asset')
    root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from go2_nav.env import Go2MaplessEnv,load_config
    cfg=load_config(root/'skills/navigation/config.json');cfg.update(max_seconds=1e9)
    env=Go2MaplessEnv(cfg,scene='empty',sensor_mode='lidar');env.reset(seed=a.seed)
    if a.asset=='source':
        source_model=(a.source_model or root/'deployment/models/go2_isaac.xml').resolve()
        if not source_model.is_file():raise FileNotFoundError(f'Missing exact Go2 source model: {source_model}')
        env.model=mujoco.MjModel.from_xml_path(str(source_model))
        env.model.opt.timestep=.001;env.data=mujoco.MjData(env.model);env.base_id=env.model.body('base').id
        yaw=np.radians(a.initial_yaw_deg)
        env.data.qpos[:3]=[0,0,.32];env.data.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        from .core import NEUTRAL
        for k,n in enumerate(SDK_JOINTS):env.data.qpos[env.model.jnt_qposadr[env.model.joint(n).id]]=NEUTRAL[k]
        mujoco.mj_forward(env.model,env.data)
        for _ in range(1000):
            for k,n in enumerate(SDK_JOINTS):
                joint=env.model.joint(n).id;aid=int(np.where(env.model.actuator_trnid[:,0]==joint)[0][0]);q=env.model.jnt_qposadr[joint];dq=env.model.jnt_dofadr[joint]
                env.data.ctrl[aid]=np.clip(40*(NEUTRAL[k]-env.data.qpos[q])-env.data.qvel[dq],-EFFORT[k],EFFORT[k])
            mujoco.mj_step(env.model,env.data)
    model,data=env.model,env.data
    ids=np.array([model.joint(n).id for n in SDK_JOINTS]);qids=model.jnt_qposadr[ids];dids=model.jnt_dofadr[ids]
    aids=np.array([int(np.where(model.actuator_trnid[:,0]==i)[0][0]) for i in ids])
    from .observer import KinematicObserver
    observer=KinematicObserver(model);foot_ids=observer.feet;floor=model.geom('floor').id
    start_time=float(data.time);runtime=UnifiedRuntime(root,allow_research_stunts=a.atomic_cycle);samples=[];events=[];requests=[];video=None;next_render=0.
    sequence=['stand','walk_moe','sideflip69','walk_moe','stand'] if a.atomic_cycle else ['stand_soft','stand_stiff','walk_mlp','walk_moe','stand'];requested=0;last_active=None;active_since=0.
    choreography=None
    if a.sequence:
        from .sequence import SkillSequence,load_steps
        choreography=SkillSequence(runtime,load_steps(a.sequence))
        sequence=[step.skill for step in choreography.steps]
    native_peak=0.;flip_rotation=0.;maximum_speed_ratio=0.
    old_tau=None;old_kp=None;last_ref=None;naive=[];max_torque_step=0.;max_gain_step=0.;max_ref_step=0.;max_tilt=0.;min_height=1e9
    wall=time.monotonic();loop_gaps=[];last_wall=wall;passed=False;reason='time budget'
    def snapshot(t,i):
        forces=np.zeros(4);contact_force=np.zeros(6)
        for index in range(data.ncon):
            c=data.contact[index];other=int(c.geom2) if int(c.geom1)==floor else int(c.geom1) if int(c.geom2)==floor else -1
            if other in foot_ids:
                mujoco.mj_contactForce(model,data,index,contact_force);forces[foot_ids.index(other)]+=abs(contact_force[0])
        rot=env.rotation;velocity=np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,env.base_id,velocity,0)
        return observer.enrich(Snapshot(t,i,data.qpos[qids].copy(),data.qvel[dids].copy(),data.actuator_force[aids].copy(),
            rot.T@velocity[:3],rot.T@np.array([0.,0.,-1.]),forces,rot.T@velocity[3:],data.qpos[3:7].copy()))
    try:
        initial=snapshot(0.,0);runtime.arm(initial,0.);runtime.stage_async(sequence[0],initial);requested=1
        if a.video:
            import imageio.v2 as imageio
            video=imageio.get_writer(a.output/'switches.mp4',fps=10,macro_block_size=1)
        for i in range(1,round(a.max_seconds/.002)+1):
            t=i*.002;before=time.monotonic();loop_gaps.append(before-last_wall);last_wall=before
            s=snapshot(t,i)
            c=runtime.controller
            if choreography is not None:choreography.tick(s,t)
            current_id=c.active.id if c.active else None
            if choreography is None and c.phase=='ARMED' and t-c.phase_at>.5 and not c.trigger_requested:c.trigger()
            if c.phase=='ACTIVE' and current_id!=last_active:
                last_active=current_id;active_since=t;events.append(dict(time=t,skill=current_id,position=data.qpos[:3].tolist()))
            if choreography is None and c.phase=='ACTIVE' and t-active_since>=(.35 if current_id=='sideflip69' else 2.5) and requested<len(sequence) and runtime.future is None and runtime.next is None:
                target=sequence[requested];profile=runtime.profile(target)[1]
                raw=profile.to_sdk(profile.kp)*(profile.to_sdk(profile.default)-s.q)-profile.to_sdk(profile.kd)*s.dq
                naive.append(dict(from_skill=current_id,to_skill=target,analytical_naive_torque_jump_Nm=float(np.max(abs(raw-c.servo.tau)))))
                runtime.stage_async(target,s);requests.append(dict(time=t,to_skill=target));requested+=1
            if choreography is None:c.set_velocity([.18,0,0] if c.phase=='ACTIVE' and current_id in ('walk_mlp','walk_moe') and t-active_since>.4 and runtime.next is None and runtime.future is None else [0,0,0])
            complete=choreography.done if choreography is not None else requested==len(sequence) and t-active_since>2
            if c.phase=='ACTIVE' and complete and current_id==sequence[-1]:
                passed=True;reason='all requested profiles committed and settled';break
            command=runtime.step(s,t)
            native=c.phase=='ACTIVE' and c.active and c.active.actuation=='native_pd'
            if native and current_id=='sideflip69':flip_rotation+=s.gyro[0]*.002
            from .core import FREE_SPEED
            maximum_speed_ratio=max(maximum_speed_ratio,float(np.max(abs(s.dq)/FREE_SPEED)))
            if old_tau is not None:
                change=float(np.max(abs(command.tau-old_tau)))
                if native and (c.active.atomic or t-c.phase_at>=.1):native_peak=max(native_peak,change)
                else:max_torque_step=max(max_torque_step,change)
            if old_kp is not None:max_gain_step=max(max_gain_step,float(np.max(abs(command.kp-old_kp))))
            if last_ref is not None:max_ref_step=max(max_ref_step,float(np.max(abs(command.qref-last_ref))))
            old_tau=command.tau.copy();old_kp=command.kp.copy();last_ref=command.qref.copy()
            data.ctrl[aids]=command.tau
            for _ in range(round(.002/model.opt.timestep)):
                if a.wire_mode=='hybrid_pd':
                    from .core import FREE_SPEED
                    speed=np.clip(data.qvel[dids],-2*FREE_SPEED,2*FREE_SPEED)
                    lower=np.maximum(EFFORT*(-1-speed/FREE_SPEED),-EFFORT);upper=np.minimum(EFFORT*(1-speed/FREE_SPEED),EFFORT)
                    raw=command.kp*(command.qref-data.qpos[qids])+command.kd*(command.dqref-data.qvel[dids])+command.feedforward
                    data.ctrl[aids]=np.clip(raw,lower,upper)
                mujoco.mj_step(model,data)
            if abs(float(data.time)-start_time-t)>1e-7:raise RuntimeError('Physics time discontinuity/reset')
            tilt=float(np.arccos(np.clip(env.rotation[2,2],-1,1)));max_tilt=max(max_tilt,tilt);min_height=min(min_height,float(data.qpos[2]))
            samples.append(np.r_[t,command.tau,command.qref,command.kp,command.kd,s.q,s.dq,s.feet,tilt,data.qpos[:3]])
            if c.phase=='FAULT':reason=c.fault;break
            if not native and (data.qpos[2]<(0.16 if a.asset=='source' else .22) or tilt>.35):reason='posture envelope exceeded';break
            if runtime.stage_error:reason='staging failure:'+runtime.stage_error;break
            if choreography is not None and choreography.error:reason='sequence failure:'+choreography.error;break
            if video is not None and t>=next_render:video.append_data(env.render());next_render=t+.1
            delay=.002-(time.monotonic()-before)
            if delay>0:time.sleep(delay)
        np.savez_compressed(a.output/'control_trace.npz',values=np.asarray(samples),
            columns=np.array(['time']+[f'{k}:{j}' for k in ['tau','qref','kp','kd','q','dq'] for j in SDK_JOINTS]+[f'force:{l}' for l in ['FR','FL','RR','RL']]+['tilt','x','y','z']))
        report=dict(seed=a.seed,asset=a.asset,wire_mode=a.wire_mode,sequence=sequence,shared_sequencer=choreography is not None,initial_yaw_deg=a.initial_yaw_deg,atomic_cycle=a.atomic_cycle,flip_rotation_rad=flip_rotation,native_motion_peak_torque_step_Nm=native_peak,maximum_joint_speed_ratio=maximum_speed_ratio,passed=bool(passed and (not a.atomic_cycle or abs(flip_rotation)>5.5) and max_torque_step<=.300001 and max_gain_step<=.030001),reason=reason,
            backend='MuJoCo real free-base hot-switch harness; commits/model hashes below define which actors were actually exercised',
            commits=events,requests=requests,controller_events=runtime.controller.events,
            max_torque_step_Nm=max_torque_step,max_kp_step=max_gain_step,max_reference_step_rad=max_ref_step,
            max_tilt_deg=float(np.degrees(max_tilt)),min_base_height_m=min_height,steps=len(samples),
            analytical_naive_switches=naive,loop_wall_gap_ms_p99=float(np.percentile(loop_gaps,99)*1000),
            loop_wall_gap_ms_max=max(loop_gaps)*1000,
            timing_scope='Simulation clock drives the controller; pacing/rendering IPC timing is recorded but is not a hard-real-time hardware certificate',
            model_sha256=dict(runtime.model_history),retained_prepared_entries=len(runtime.prepared),hardware_connected=False,
            root_writes='initialization only; actuator rollout in reported wire_mode, no stabilizing root force or weld',final_position=data.qpos[:3].tolist())
        report['source_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'rob2nav/deployment').glob('*.py')}
        report['profile_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'deployment/skills').glob('*.json')}
        (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:
        if video is not None:video.close()
        runtime.close();env.close()


if __name__=='__main__':main()
