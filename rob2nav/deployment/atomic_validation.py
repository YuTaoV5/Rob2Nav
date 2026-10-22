"""Read-only, per-physics-step backflip acceptance inside a switch sequence."""
import numpy as np
from .types import SDK_JOINTS
from rob2nav.fixstand_validation import assess


class BackflipValidation:
    def __init__(self,model,skill,start):
        self.skill=skill;self.start=start;self.trace=[];self.next_sample=0.
        self.rotation=0.;self.travel=0.;self.air=0.;self.max_air=0.;self.height=0.
        self.minimum_up=1.;self.bad=False;self.speed=0.;self.excess=0.;self.elapsed=0.
        self.ids=np.array([model.joint(n).id for n in SDK_JOINTS])
        self.qids=model.jnt_qposadr[self.ids];self.dids=model.jnt_dofadr[self.ids]
        self.base=model.body('base').id

    def sample(self,model,data,now):
        import mujoco
        t=now-self.start
        if t>self.skill.atomic_duration_s+1e-7:return
        dt=model.opt.timestep;self.elapsed=t
        self.rotation+=float(data.qvel[4])*dt;self.travel+=float(np.linalg.norm(data.qvel[3:6]))*dt
        from .core import FREE_SPEED,NEUTRAL
        self.speed=max(self.speed,float(np.max(abs(data.qvel[self.dids])/FREE_SPEED)))
        q=data.qpos[self.qids];ranges=model.jnt_range[self.ids]
        self.excess=max(self.excess,float(max(0.,np.max(np.maximum(q-ranges[:,1],ranges[:,0]-q)))))
        forces={leg:0. for leg in ('FR','FL','RR','RL')};contacts=[];force=np.zeros(6)
        for ci in range(data.ncon):
            c=data.contact[ci];x,y=int(c.geom1),int(c.geom2)
            bx,by=int(model.geom_bodyid[x]),int(model.geom_bodyid[y])
            other=by if bx==0 and by else bx if by==0 and bx else None
            if other is None:continue
            name=model.body(other).name;contacts.append(name)
            for leg in forces:
                if name.startswith(leg+'_'):
                    mujoco.mj_contactForce(model,data,ci,force);forces[leg]+=abs(float(force[0]))
        self.bad|=any('foot' not in n and 'calf' not in n for n in contacts)
        self.air=0. if contacts else self.air+dt;self.max_air=max(self.max_air,self.air)
        up=float(data.xmat[self.base].reshape(3,3)[2,2]);self.minimum_up=min(self.minimum_up,up)
        self.height=max(self.height,float(data.qpos[2]))
        if t+1e-7>=self.next_sample:
            self.trace.append(dict(time=t,position=data.qpos[:3].tolist(),upright=up,
                speed=float(np.linalg.norm(data.qvel[:2])),angular_speed=float(np.linalg.norm(data.qvel[3:6])),
                standard_pose_error=float(np.max(abs(q-NEUTRAL))),max_joint_speed=float(np.max(abs(data.qvel[self.dids]))),
                all_feet_ground=all(v>4. for v in forces.values()),foot_forces_N=forces))
            self.next_sample+=.02

    def report(self):
        dominance=abs(self.rotation)/max(self.travel,1e-8);tail=self.trace[-100:]
        complete=self.elapsed>=self.skill.atomic_duration_s-.003
        motion=bool(complete and 5.5<abs(self.rotation)<7.1 and dominance>.75 and self.max_air>=.08 and
                    self.height>.45 and self.minimum_up<-.8 and not self.bad and tail and
                    all(x['upright']>.9 and x['position'][2]>.22 and x['speed']<.25 and x['angular_speed']<1. for x in tail))
        endpoints=assess(self.trace,initial_end=.8);constraints=self.speed<=1. and self.excess<=.001
        contract=self.skill.motor_control is not None
        return dict(skill=self.skill.id,start_s=self.start,coverage_complete=complete,action_completed=motion,
            fixstand=endpoints,body_y_rotation_rad=self.rotation,axis_dominance=dominance,
            airtime_s=self.max_air,max_height_m=self.height,unexpected_body_ground_contact=self.bad,
            max_joint_speed_ratio=self.speed,max_joint_limit_excess_rad=self.excess,
            hardware_constraints_pass=constraints,motor_contract=self.skill.motor_control,
            motor_contract_present=contract,passed=bool(motion and endpoints['passed'] and constraints and contract),hardware_qualified=False)
