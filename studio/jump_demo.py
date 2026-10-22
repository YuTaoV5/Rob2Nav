"""Bounded actuator-only vertical-jump primitive; no base forces or pose replay."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from go2_nav.skills import JumpSkill
from rob2nav.deployment.types import SDK_JOINTS
from rob2nav.deployment.core import EFFORT,FREE_SPEED,NEUTRAL
from rob2nav.deployment.observer import KinematicObserver
TARGET_RATE=10.  # rad/s, reference slew across crouch/flight/landing phase boundaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(a.root/'deployment/models/go2_isaac.xml'));m.opt.timestep=.001
    d=mujoco.MjData(m);ids=np.array([m.joint(n).id for n in SDK_JOINTS]);qids=m.jnt_qposadr[ids];dids=m.jnt_dofadr[ids]
    aids=np.array([int(np.where(m.actuator_trnid[:,0]==j)[0][0]) for j in ids]);floor=m.geom('floor').id
    feet=set(KinematicObserver(m).feet);d.qpos[:3]=[0,0,.32];d.qpos[3:7]=[1,0,0,0];d.qpos[qids]=NEUTRAL
    mujoco.mj_forward(m,d);jump=JumpSkill({'pushback':0.,'crouch_height':.25,'crouch_time':.3,'extend_height':.37,'flight_time':.12,'kd':2.});triggered=False
    # Existing primitive order FL FR RL RR -> SDK FR FL RR RL.
    reorder=np.array([1,0,3,2]);trace=[];air=0.;max_air=0.;peak=0.;base_height=None;bad_contact=False;max_speed=0.;base=m.body('base').id
    smooth_target=NEUTRAL.copy();max_limit_excess=0.
    for i in range(3500):
        t=i*.002
        if t>=1.5 and not triggered:jump.trigger(t);triggered=True
        contact_feet=set();nonfoot=False
        for contact in d.contact:
            other=int(contact.geom2) if contact.geom1==floor else int(contact.geom1) if contact.geom2==floor else -1
            if other in feet:contact_feet.add(other)
            elif other>=0:nonfoot=True
        if t>1.45 and base_height is None:base_height=float(d.qpos[2])
        if t>1.5:
            air=air+.002 if not contact_feet and not nonfoot and d.qpos[2]>.23 else 0.;max_air=max(max_air,air)
            peak=max(peak,float(d.qpos[2]));bad_contact|=nonfoot
        target,kp,kd=jump.targets(t,NEUTRAL.reshape(4,3)[reorder].ravel(),np.full(12,40.),np.full(12,2.),allow_release=len(contact_feet)>=3)
        if jump.active:
            legs=target.reshape(4,3);legs[:,0]=0.;legs[2:,1]+=.2
        target=target.reshape(4,3)[reorder].ravel();kp=kp.reshape(4,3)[reorder].ravel();kd=kd.reshape(4,3)[reorder].ravel()
        target=np.clip(target,m.jnt_range[ids,0],m.jnt_range[ids,1])
        smooth_target+=np.clip(target-smooth_target,-TARGET_RATE*.002,TARGET_RATE*.002)
        target=smooth_target.copy()
        for _ in range(2):
            speed=np.clip(d.qvel[dids],-2*FREE_SPEED,2*FREE_SPEED)
            lower=np.maximum(EFFORT*(-1-speed/FREE_SPEED),-EFFORT);upper=np.minimum(EFFORT*(1-speed/FREE_SPEED),EFFORT)
            d.ctrl[aids]=np.clip(kp*(target-d.qpos[qids])-kd*d.qvel[dids],lower,upper)
            mujoco.mj_step(m,d)
            max_speed=max(max_speed,float(np.max(abs(d.qvel[dids])/FREE_SPEED)))
            max_limit_excess=max(max_limit_excess,float(np.max(np.maximum(m.jnt_range[ids,0]-d.qpos[qids],d.qpos[qids]-m.jnt_range[ids,1]))))
        if i%10==0:trace.append(dict(time=float(d.time),height=float(d.qpos[2]),contacts=len(contact_feet),position=d.qpos[:3].tolist(),upright=float(d.xmat[base].reshape(3,3)[2,2]),joint_speed=float(np.max(abs(d.qvel[dids])))))
    # Score only this modest primitive; no trained high-jump qualification is implied.
    terminal=np.mean([x['contacts']>=3 and x['height']>.26 and x['upright']>.97 and x['joint_speed']<1. for x in trace[-25:]])>.9
    passed=bool(max_air>.04 and peak-(base_height or .32)>.05 and terminal and not bad_contact and max_speed<=1. and max_limit_excess<=.001)
    report=dict(backend='MuJoCo',successes=int(passed),total=1,controller='bounded joint-space JumpSkill; no learned weights',
        hardware_constraints_pass=bool(max_speed<=1. and max_limit_excess<=.001 and not bad_contact),
        metrics=dict(airtime_s=max_air,base_height_gain_m=peak-(base_height or .32),peak_base_height_m=peak,terminal_stance=bool(terminal),nonfoot_contact=bad_contact,max_joint_speed_ratio=max_speed,max_joint_limit_excess_rad=max_limit_excess,target_rate_rad_s=TARGET_RATE),
        parameters=jump.p,hardware_enabled=False,root_writes='initialization only; no applied base force')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));(a.output/'trace.json').write_text(json.dumps(trace))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
