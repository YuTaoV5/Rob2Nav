"""Read-only substep diagnostics; never modifies simulation or acceptance metrics."""
import numpy as np


class MotionDiagnostics:
    CONTEXT_FIELDS=('peak_joint_position_rad','peak_target_position_rad','peak_actuator_generalized_torque_Nm',
                    'peak_constraint_generalized_torque_Nm','peak_bias_generalized_torque_Nm')
    def __init__(self, joint_names, limits, primary_axis, bin_seconds=.25):
        self.names=list(joint_names)
        self.limits=np.asarray(limits,dtype=float)
        self.axis=primary_axis
        self.bin_seconds=bin_seconds
        self.joints={name:dict(max_ratio=0.,peak_time_s=0.,peak_velocity_rad_s=0.,overspeed_seconds=0.) for name in self.names}
        self.contacts={}
        self.bins={}

    def record(self, time, dt, dq, omega, contacts, height, context=None):
        dq=np.asarray(dq);omega=np.asarray(omega)
        context={} if context is None else context
        if set(context)-set(self.CONTEXT_FIELDS):raise ValueError('Unknown peak context field')
        if any(np.asarray(x).shape!=dq.shape or not np.isfinite(x).all() for x in context.values()):raise ValueError('Invalid peak context shape or values')
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
                for field in self.CONTEXT_FIELDS:j.pop(field,None)
                j.update(max_ratio=float(ratios[i]),peak_time_s=float(time),peak_velocity_rad_s=float(dq[i]))
                j.update({key:float(value[i]) for key,value in context.items()})
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

    def sample(self, model, data, time, dof_ids, omega, target=None):
        import mujoco
        contacts=[];force=np.zeros(6)
        for i,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            other=b if model.geom_bodyid[a]==0 and model.geom_bodyid[b]!=0 else a if model.geom_bodyid[b]==0 and model.geom_bodyid[a]!=0 else None
            if other is not None:
                mujoco.mj_contactForce(model,data,i,force)
                contacts.append((model.body(int(model.geom_bodyid[other])).name,force[0],c.dist))
        if not hasattr(self,'_model'):
            ids=np.array([model.joint(n).id for n in self.names])
            if not np.array_equal(model.jnt_dofadr[ids],dof_ids):raise ValueError('Diagnostic joint/DOF order mismatch')
            self._model=model;self._qids=model.jnt_qposadr[ids]
        elif model is not self._model:raise ValueError('Diagnostic model changed')
        context=dict(peak_joint_position_rad=data.qpos[self._qids],
            peak_actuator_generalized_torque_Nm=data.qfrc_actuator[dof_ids],
            peak_constraint_generalized_torque_Nm=data.qfrc_constraint[dof_ids],
            peak_bias_generalized_torque_Nm=data.qfrc_bias[dof_ids])
        if target is not None:context['peak_target_position_rad']=np.asarray(target)
        self.record(time,model.opt.timestep,data.qvel[dof_ids],omega,contacts,data.qpos[2],context)

    def report(self):
        return dict(joint_speed_peaks=self.joints,nonfoot_ground_contacts=self.contacts,
            time_bins=[self.bins[k] for k in sorted(self.bins)],primary_axis=self.axis,
            angular_frame='body',state_writes=False,acceptance_thresholds_changed=False,
            peak_context_scope='Post-step q/dq and forces used in the just-completed integration step; generalized joint torques, not isolated causal attribution')
