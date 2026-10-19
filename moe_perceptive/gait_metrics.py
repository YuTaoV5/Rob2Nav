"""First-episode diagnostics, independent of Isaac imports; no success labels."""
import math


class GaitMetrics:
    def __init__(self,count,dt,contact_threshold=5.):
        self.dt=dt;self.threshold=contact_threshold
        self.rows=[dict(samples=0,seconds=0.,stance_foot_seconds=0.,slip_distance_m=0.,
            absolute_joint_work_J=0.,speed_integral_m=0.,upright_integral_s=0.,minimum_upright=1.) for _ in range(count)]

    def update(self,active,foot_velocity,contact_force,joint_velocity,torque,root_velocity,upright):
        for i,on in enumerate(active):
            if not on:continue
            r=self.rows[i];r['samples']+=1;r['seconds']+=self.dt
            for velocity,force in zip(foot_velocity[i],contact_force[i]):
                if force[2]>self.threshold:
                    r['stance_foot_seconds']+=self.dt
                    r['slip_distance_m']+=math.hypot(velocity[0],velocity[1])*self.dt
            r['absolute_joint_work_J']+=sum(abs(t*v) for t,v in zip(torque[i],joint_velocity[i]))*self.dt
            r['speed_integral_m']+=math.hypot(*root_velocity[i][:2])*self.dt
            r['upright_integral_s']+=upright[i]*self.dt
            r['minimum_upright']=min(r['minimum_upright'],upright[i])

    def episode(self,index):
        r=self.rows[index];seconds=r['seconds'];contact=r['stance_foot_seconds']
        return {**r,'mean_stance_foot_speed_m_s':r['slip_distance_m']/contact if contact else None,
            'mean_planar_speed_m_s':r['speed_integral_m']/seconds if seconds else None,
            'mean_upright':r['upright_integral_s']/seconds if seconds else None}

    @staticmethod
    def definition():
        return {'contact_threshold_N':5.,'stance_slip':'World horizontal link velocity while vertical net contact force >5N; not sole contact-point slip',
            'joint_work':'Integral sum |applied_torque * joint_velocity| dt; mechanical diagnostic, not battery energy or COT',
            'scope':'Each first episode only, including failures; do not compare different termination horizons without stratifying success/time',
            'not_measured':['foot clearance collisions','command tracking error','gait aesthetics','battery energy']}
