"""Free-base Go2 policy evaluation. Root state is set only at episode initialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['sideflip','upright_walk'],required=True)
    p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--backflip-source',required=True,type=Path)
    p.add_argument('--motion',type=Path)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--command',type=float,nargs=3,default=[.2,0.,0.])
    p.add_argument('--video',action='store_true')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if not 0<args.seconds<=60:
        raise ValueError('Expected a finite duration in (0,60] seconds')
    meta=json.loads(args.policy.with_suffix('.json').read_text())
    if hashlib.sha256(args.policy.read_bytes()).hexdigest()!=meta['policy_sha256']:
        raise ValueError('Exported policy hash mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    actor=torch.jit.load(str(args.policy),map_location='cpu').eval()
    model=mujoco.MjModel.from_xml_path(str(args.backflip_source/'mujoco/go2/go2_isaac.xml'))
    model.opt.timestep=.005
    data=mujoco.MjData(model)
    names=meta['joint_names']
    jids=np.array([model.joint(n).id for n in names])
    qids=model.jnt_qposadr[jids]
    vids=model.jnt_dofadr[jids]
    aids=np.array([int(np.where(model.actuator_trnid[:,0]==j)[0][0]) for j in jids])
    default=np.asarray(meta['default_joint_pos'],dtype=float)
    urdf=ET.parse(args.backflip_source/'resources/robots/go2/urdf/go2_description.urdf').getroot()
    limits={j.attrib['name']:j.find('limit').attrib for j in urdf.findall('joint') if j.attrib['name'] in names}
    effort=np.array([float(limits[n]['effort']) for n in names])
    velocity_limits=np.array([float(limits[n]['velocity']) for n in names])
    base=model.body('base').id
    reference=None
    if args.kind=='sideflip':
        if args.motion is None:
            raise ValueError('--motion reference NPZ is required')
        if hashlib.sha256(args.motion.read_bytes()).hexdigest()!=meta['reference_sha256']:
            raise ValueError('Motion reference hash mismatch')
        reference=np.load(args.motion)
        anchor=meta['body_names'].index(meta['anchor'])
        data.qpos[:3]=reference['body_pos_w'][0,anchor]
        data.qpos[3:7]=reference['body_quat_w'][0,anchor]
        data.qpos[qids]=reference['joint_pos'][0]
    else:
        data.qpos[:3]=[0,0,.32]
        data.qpos[3:7]=[1,0,0,0]
        data.qpos[qids]=default
    mujoco.mj_forward(model,data)
    rng=np.random.default_rng(args.seed)
    # Small initial joint uncertainty, never an in-flight state reset.
    data.qpos[qids]+=rng.uniform(-.002,.002,12)
    target=data.qpos[qids].copy()
    for _ in range(100):
        data.ctrl[aids]=np.clip(40*(target-data.qpos[qids])-data.qvel[vids],-effort,effort)
        mujoco.mj_step(model,data)
    start=float(data.time)
    last=np.zeros(12,dtype=np.float32)
    trace=[]
    raw_roll=[]
    body_velocity=np.zeros(6)
    command=np.asarray(args.command,dtype=np.float32)
    writer=None
    renderer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(args.output/'episode_0.mp4',fps=25,macro_block_size=1)
        renderer=mujoco.Renderer(model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.distance=2.3;camera.azimuth=130;camera.elevation=-15
    next_frame=0.
    max_speed_ratio=0.;max_limit_excess=0.;max_airtime=0.;airtime=0.;bad_floor_contact=False
    signed_x_rotation=0.;angular_travel=0.;velocity_buffer=np.zeros(6)
    floor_ids={i for i in range(model.ngeom) if model.geom_bodyid[i]==0}
    try:
        while float(data.time)-start<args.seconds-1e-6:
            now=float(data.time)-start
            rot=data.xmat[base].reshape(3,3)
            gravity=rot.T@np.array([0.,0.,-1.])
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,body_velocity,1)
            q=data.qpos[qids]-default
            dq=data.qvel[vids]
            if reference is not None:
                index=min(int(round(now*float(reference['fps'][0]))),len(reference['joint_pos'])-1)
                refrot=np.zeros(9)
                mujoco.mju_quat2Mat(refrot,reference['body_quat_w'][index,anchor].astype(float))
                anchor_ori=(rot.T@refrot.reshape(3,3))[:,:2].reshape(-1)
                obs=np.r_[reference['joint_pos'][index],reference['joint_vel'][index],anchor_ori,body_velocity[:3],q,dq,last]
            else:
                terms={'base_ang_vel':body_velocity[:3], 'projected_gravity':gravity,
                    'velocity_commands':command,'joint_pos':q,'joint_vel':dq,'actions':last,
                    'base_lin_vel':body_velocity[3:]}
                values=[]
                for term in meta['actor_observation_terms']:
                    value=np.asarray(terms[term['name']]).copy()
                    if term['scale'] is not None:
                        value*=term['scale']
                    if term['clip'] is not None:
                        value=np.clip(value,*term['clip'])
                    values.append(value)
                obs=np.concatenate(values)
            if len(obs)!=meta['actor_input_dim'] or not np.isfinite(obs).all():
                raise ValueError('Observation contract or numerical error')
            with torch.inference_mode():
                action=actor(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)).numpy()[0]
            if not np.isfinite(action).all():
                raise FloatingPointError('Nonfinite actor output')
            last=np.clip(action,-100,100)
            target=default+float(meta['action_scale'])*last
            for _ in range(4):
                data.ctrl[aids]=np.clip(40*(target-data.qpos[qids])-data.qvel[vids],-effort,effort)
                mujoco.mj_step(model,data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('Nonfinite MuJoCo state')
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,velocity_buffer,1)
                signed_x_rotation+=float(velocity_buffer[0])*model.opt.timestep
                angular_travel+=float(np.linalg.norm(velocity_buffer[:3]))*model.opt.timestep
                max_speed_ratio=max(max_speed_ratio,float(np.max(np.abs(data.qvel[vids])/velocity_limits)))
                excess=np.maximum(data.qpos[qids]-model.jnt_range[jids,1],model.jnt_range[jids,0]-data.qpos[qids])
                max_limit_excess=max(max_limit_excess,float(max(0,np.max(excess))))
                ground_bodies=[]
                for contact in data.contact:
                    a,b=int(contact.geom1),int(contact.geom2)
                    if a in floor_ids and b not in floor_ids:ground_bodies.append(model.body(int(model.geom_bodyid[b])).name)
                    elif b in floor_ids and a not in floor_ids:ground_bodies.append(model.body(int(model.geom_bodyid[a])).name)
                airtime=0. if ground_bodies else airtime+model.opt.timestep
                max_airtime=max(max_airtime,airtime)
                bad_floor_contact|=any(not ('foot' in name.lower() or 'calf' in name.lower()) for name in ground_bodies)
            rot=data.xmat[base].reshape(3,3)
            raw_roll.append(float(np.arctan2(-rot[1,2],rot[2,2])))
            up_error=float(np.linalg.norm(rot.T@np.array([0.,0.,-1.])-[-1,0,0]))
            over=np.maximum(data.qpos[qids]-model.jnt_range[jids,1],model.jnt_range[jids,0]-data.qpos[qids])
            trace.append(dict(time=float(data.time)-start,position=data.qpos[:3].tolist(),quaternion=data.qpos[3:7].tolist(),
                joints=data.qpos[qids].tolist(),upright=float(rot[2,2]),upright_walk_error=up_error,
                front_ground=any(n.startswith(('FL_','FR_')) for n in ground_bodies),
                rear_ground=any(n.startswith(('RL_','RR_')) for n in ground_bodies),
                speed=float(np.linalg.norm(data.qvel[:2])),joint_speed_ratio=float(np.max(np.abs(data.qvel[vids])/velocity_limits)),
                angular_speed=float(np.linalg.norm(velocity_buffer[:3])),
                joint_limit_excess=float(max(0,np.max(over)))))
            if renderer is not None and now>=next_frame:
                camera.lookat[:]=data.qpos[:3]
                renderer.update_scene(data,camera=camera);writer.append_data(renderer.render());next_frame+=.04
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    roll=np.unwrap(raw_roll)
    last_window=trace[-min(len(trace),100):]
    if args.kind=='sideflip':
        turn=float(roll[-1]-roll[0])
        axis_dominance=abs(signed_x_rotation)/max(angular_travel,1e-8)
        success=bool(5.5<abs(turn)<7.1 and abs(signed_x_rotation)>4.5 and axis_dominance>.75 and
            max_airtime>=.08 and not bad_floor_contact and
            max(x['position'][2] for x in trace)>.45 and min(x['upright'] for x in trace)<-.8 and
            all(x['upright']>.90 and x['position'][2]>.22 and x['speed']<.25 and x['angular_speed']<1. for x in last_window))
        metrics=dict(unwrapped_roll_rad=turn,body_x_rotation_rad=signed_x_rotation,axis_dominance=axis_dominance,
            terminal_upright=trace[-1]['upright'],airtime_s=max_airtime)
    else:
        duration=sum(x['upright_walk_error']<.45 and x['position'][2]>.42 for x in trace)*.02
        distance=float(np.linalg.norm(np.array(trace[-1]['position'][:2])-trace[0]['position'][:2]))
        upright_travel=float(np.linalg.norm(np.array(last_window[-1]['position'][:2])-last_window[0]['position'][:2]))
        support=sum(x['rear_ground'] and not x['front_ground'] for x in last_window)/len(last_window)
        success=bool(duration>=2. and upright_travel>.20 and support>=.8 and not bad_floor_contact and
            all(x['upright_walk_error']<.45 for x in last_window))
        metrics=dict(upright_seconds=duration,travel_m=distance,upright_travel_m=upright_travel,
            rear_support_fraction=support,started_on_four_feet=True)
    metrics.update(max_joint_speed_ratio=max_speed_ratio,max_joint_limit_excess_rad=max_limit_excess,
        unexpected_body_ground_contact=bad_floor_contact)
    (args.output/'trace.json').write_text(json.dumps(trace))
    summary=dict(args=dict(kind=args.kind,seed=args.seed),backend='MuJoCo',successes=int(success),total=1,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        episodes=[dict(success=success,**metrics)],metrics=metrics,policy_sha256=meta['policy_sha256'],
        root_state_writes='initialization only; no root stabilization/weld/teleport during evaluation',hardware_connected=False,
        hardware_constraints_pass=metrics['max_joint_speed_ratio']<=1. and metrics['max_joint_limit_excess_rad']<=.001)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
