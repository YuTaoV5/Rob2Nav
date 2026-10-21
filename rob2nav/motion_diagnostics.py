"""Read-only substep diagnostics; never modifies simulation or acceptance metrics."""
import numpy as np


class MotionDiagnostics:
    def __init__(self, joint_names, limits, primary_axis, bin_seconds=.25):
        self.names=list(joint_names)
        self.limits=np.asarray(limits,dtype=float)
        self.axis=primary_axis
        self.bin_seconds=bin_seconds
        self.joints={name:dict(max_ratio=0.,peak_time_s=0.,peak_velocity_rad_s=0.,overspeed_seconds=0.) for name in self.names}
        self.contacts={}
        self.bins={}

    def record(self, time, dt, dq, omega, contacts, height):
        dq=np.asarray(dq);omega=np.asarray(omega)
        ratios=np.abs(dq)/self.limits
        key=int(max(0.,time-1e-8)//self.bin_seconds)
        row=self.bins.setdefault(key,dict(start_s=key*self.bin_seconds,end_s=(key+1)*self.bin_seconds,
            signed_axis_rad=0.,angular_travel_rad=0.,off_axis_travel_rad=0.,max_speed_ratio=0.,min_height_m=float(height)))
        row['signed_axis_rad']+=float(omega[self.axis])*dt
        row['angular_travel_rad']+=float(np.linalg.norm(omega))*dt
        row['off_axis_travel_rad']+=float(np.linalg.norm(np.delete(omega,self.axis)))*dt
        row['max_speed_ratio']=max(row['max_speed_ratio'],float(max(ratios)))
        row['min_height_m']=min(row['min_height_m'],float(height))
        for i,name in enumerate(self.names):
            j=self.joints[name]
            if ratios[i]>j['max_ratio']:
                j.update(max_ratio=float(ratios[i]),peak_time_s=float(time),peak_velocity_rad_s=float(dq[i]))
            if ratios[i]>1.:j['overspeed_seconds']+=dt
        seen=set()
        for body,force,distance in contacts:
            if 'foot' in body.lower() or 'calf' in body.lower():continue
            c=self.contacts.setdefault(body,dict(first_time_s=float(time),last_time_s=float(time),duration_s=0.,max_normal_force_N=0.,max_penetration_m=0.))
            c['last_time_s']=float(time)
            if body not in seen:c['duration_s']+=dt
            seen.add(body)
            c['max_normal_force_N']=max(c['max_normal_force_N'],abs(float(force)))
            c['max_penetration_m']=max(c['max_penetration_m'],max(0.,-float(distance)))

    def sample(self, model, data, time, dof_ids, omega):
        import mujoco
        contacts=[];force=np.zeros(6)
        for i,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            other=b if model.geom_bodyid[a]==0 and model.geom_bodyid[b]!=0 else a if model.geom_bodyid[b]==0 and model.geom_bodyid[a]!=0 else None
            if other is not None:
                mujoco.mj_contactForce(model,data,i,force)
                contacts.append((model.body(int(model.geom_bodyid[other])).name,force[0],c.dist))
        self.record(time,model.opt.timestep,data.qvel[dof_ids],omega,contacts,data.qpos[2])

    def report(self):
        return dict(joint_speed_peaks=self.joints,nonfoot_ground_contacts=self.contacts,
            time_bins=[self.bins[k] for k in sorted(self.bins)],primary_axis=self.axis,
            angular_frame='body',state_writes=False,acceptance_thresholds_changed=False)
