"""Read-only, per-physics-step backflip acceptance inside a switch sequence."""
import numpy as np
from .types import SDK_JOINTS
from rob2nav.fixstand_validation import assess,rotation_matches


class JointConstraintValidation:
    """All phases, including entry/exit: peaks cannot hide between servo ticks."""
    def __init__(self,model,names,velocity_limits):
        ids=np.array([model.joint(n).id for n in names])
        self.qids=model.jnt_qposadr[ids];self.dids=model.jnt_dofadr[ids]
        self.ranges=model.jnt_range[ids];self.velocity=np.asarray(velocity_limits)
        self.hz=1./model.opt.timestep;self.samples=0;self.speed=0.;self.excess=0.

    def sample(self,data):
        self.speed=max(self.speed,float(np.max(abs(data.qvel[self.dids])/self.velocity)))
        q=data.qpos[self.qids]
        self.excess=max(self.excess,float(max(0.,np.max(np.maximum(q-self.ranges[:,1],self.ranges[:,0]-q)))))
        self.samples+=1

    def report(self,expected_samples):
        covered=self.samples==expected_samples and self.samples>0
        return dict(scope='every physical substep throughout the executed sequence',physics_hz=self.hz,
            samples=self.samples,expected_samples=expected_samples,coverage_complete=covered,
            max_joint_speed_ratio=self.speed,max_joint_limit_excess_rad=self.excess,
            passed=bool(covered and self.speed<=1. and self.excess<=.001))


class EntryHoldStatistics:
    """Read-only substep peaks during the trained one-second entry hold."""
    def __init__(self, dt, duration=1.):
        self.dt=float(dt);self.duration=float(duration);self.samples=0
        self.first=None;self.last=None;self.joint_speed=0.;self.angular_speed=0.;self.pose_error=0.

    def sample(self, elapsed, q, dq, omega, neutral):
        if not 1e-7<elapsed<=self.duration+1e-7:return
        self.samples+=1
        if self.first is None:self.first=float(elapsed)
        self.last=float(elapsed)
        self.joint_speed=max(self.joint_speed,float(np.max(np.abs(dq))))
        self.angular_speed=max(self.angular_speed,float(np.linalg.norm(omega)))
        self.pose_error=max(self.pose_error,float(np.max(np.abs(np.asarray(q)-neutral))))

    def report(self):
        expected=round(self.duration/self.dt)
        covered=(self.samples==expected and self.first is not None and self.first<=self.dt+1e-7
                 and self.last>=self.duration-self.dt-1e-7)
        return dict(scope='trained entry hold only; every physical substep',duration_s=self.duration,
                    physics_hz=1./self.dt,samples=self.samples,expected_samples=expected,
                    coverage_complete=bool(covered),first_sample_s=self.first,last_sample_s=self.last,
                    max_joint_speed_rad_s=self.joint_speed,max_body_angular_speed_rad_s=self.angular_speed,
                    max_pose_error_rad=self.pose_error,visual_twitch_free_verified=False)


class BackflipValidation:
    def __init__(self,model,skill,start,data=None):
        self.skill=skill;self.start=start;self.trace=[];self.next_sample=0.
        self.rotation=0.;self.travel=0.;self.air=0.;self.max_air=0.;self.height=0.
        self.entry_hold=EntryHoldStatistics(model.opt.timestep)
        self.minimum_up=1.;self.bad=False;self.speed=0.;self.excess=0.;self.elapsed=0.
        self.ids=np.array([model.joint(n).id for n in SDK_JOINTS])
        self.qids=model.jnt_qposadr[self.ids];self.dids=model.jnt_dofadr[self.ids]
        self.base=model.body('base').id
        from rob2nav.ground_contacts import GroundContactReader
        self.ground_reader=GroundContactReader(model)
        self.initial_state=None if data is None else dict(schema_version=1,joint_names=list(SDK_JOINTS),
            joint_position=data.qpos[self.qids].tolist(),joint_velocity=data.qvel[self.dids].tolist(),
            root_position=data.qpos[:3].tolist(),root_quaternion=data.qpos[3:7].tolist(),
            root_linear_velocity_world=data.qvel[:3].tolist(),root_angular_velocity_body=data.qvel[3:6].tolist())

    def sample(self,model,data,now):
        import mujoco
        t=now-self.start
        if t>self.skill.atomic_duration_s+1e-7:return
        dt=model.opt.timestep;self.elapsed=t
        self.rotation+=float(data.qvel[4])*dt;self.travel+=float(np.linalg.norm(data.qvel[3:6]))*dt
        from .core import FREE_SPEED,NEUTRAL
        self.speed=max(self.speed,float(np.max(abs(data.qvel[self.dids])/FREE_SPEED)))
        q=data.qpos[self.qids];ranges=model.jnt_range[self.ids]
        self.entry_hold.sample(t,q,data.qvel[self.dids],data.qvel[3:6],NEUTRAL)
        self.excess=max(self.excess,float(max(0.,np.max(np.maximum(q-ranges[:,1],ranges[:,0]-q)))))
        frame=self.ground_reader.sample(data);forces=frame.forces;contacts=frame.bodies
        self.bad|=bool(frame.unexpected)
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
        direction=rotation_matches(self.rotation,'back')
        motion=bool(direction and complete and 5.5<abs(self.rotation)<7.1 and dominance>.75 and self.max_air>=.08 and
                    self.height>.45 and self.minimum_up<-.8 and not self.bad and tail and
                    all(x['upright']>.9 and x['position'][2]>.22 and x['speed']<.25 and x['angular_speed']<1. for x in tail))
        endpoints=assess(self.trace,initial_end=.8);constraints=self.speed<=1. and self.excess<=.001
        contract=self.skill.motor_control is not None
        return dict(skill=self.skill.id,start_s=self.start,coverage_complete=complete,action_completed=motion,
            initial_state=getattr(self,'initial_state',None),
            entry_hold_physics=self.entry_hold.report() if self.skill.action_parameterization in ('backflip_entry_v2','backflip_entry_v3') else None,
            expected_direction='back',direction_pass=direction,
            fixstand=endpoints,body_y_rotation_rad=self.rotation,axis_dominance=dominance,
            airtime_s=self.max_air,max_height_m=self.height,unexpected_body_ground_contact=self.bad,
            ground_contact_rule='foot spheres and contact points inside their own anatomical foot region',
            ground_contact_rule_sha256=self.ground_reader.rule_sha256,
            max_joint_speed_ratio=self.speed,max_joint_limit_excess_rad=self.excess,
            hardware_constraints_pass=constraints,motor_contract=self.skill.motor_control,
            motor_contract_present=contract,passed=bool(motion and endpoints['passed'] and constraints and contract),hardware_qualified=False)
