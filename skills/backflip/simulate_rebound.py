"""Dedicated continuous rebound rollout using the pinned native motor loop."""
import hashlib
import json
import numpy as np
from .rebound_playback import ReboundPlaybackState
from rob2nav.rebound_validation import ContinuousFlipAudit


class ReboundEvaluation:
    def configure(self, module, exported, args):
        contract=exported.get('rebound_contract')
        if not contract:raise ValueError('Audited rebound export required')
        if hashlib.sha256(args.onnx.read_bytes()).hexdigest()!=exported.get('policy_sha256'):
            raise ValueError('Policy hash mismatch')
        if args.direction!='back' or args.initial_state is not None or args.entry_hold_diagnostic is not None:
            raise ValueError('Unsupported rebound diagnostic override')
        if args.seconds<contract['episode_seconds']:
            raise ValueError('Rebound rollout must cover its full trained episode')
        args.fixstand_roundtrip=True
        self.module=module;self.args=args;self.contract=contract
        self.runtime=ReboundPlaybackState(contract,module.SIM_DT)
        c=contract['settings']
        self.audit=ContinuousFlipAudit(module.SIM_DT,cycles=c['cycles'],max_rebound_seconds=c['max_rebound_seconds'],
            takeoff_confirm_seconds=c['takeoff_confirm_seconds'],min_airtime_seconds=c['min_airtime_seconds'],
            min_rotation=c['min_rotation'],max_rotation=c['max_rotation'],min_axis_dominance=c['min_axis_dominance'])
        self.velocity=np.asarray(exported['control_contract']['joint_velocity_limits'])
        self.previous=np.zeros(12);self.applied=np.zeros(12);self.slew=np.zeros(12)
        self.last_target=module.DEFAULT_DOF_POS.copy()
        self.target_ratio=0.;self.issued_ratio=0.;self.entry_target_step=0.;self.calls=0
        self.entry_dq=0.;self.entry_omega=0.;self.last_time=0.;self.samples=[]
        module.phase_features=lambda _:self.runtime.phase_features()
        original_load=module.load_onnx
        outer=self
        class Session:
            def __init__(self,session):self.session=session
            def run(self,*args,**kwargs):
                from skills.fixstand import backflip_entry_weight
                result=self.session.run(*args,**kwargs)
                time=outer.calls*.02;w=float(backflip_entry_weight(time))
                action=np.clip((1-w)*result[0]+w*contract['entry_hold_scale']*np.tanh(result[0]),
                               -module.ACTION_CLIP,module.ACTION_CLIP).astype(np.float32)
                delta=np.abs(action.reshape(12)-outer.previous)*module.ACTION_SCALE
                outer.target_ratio=max(outer.target_ratio,float(np.max(delta/(outer.velocity*.02))))
                if time<=1.+1e-7:outer.entry_target_step=max(outer.entry_target_step,float(delta.max()))
                outer.applied=outer.previous.copy();outer.previous=action.reshape(12).copy()
                outer.calls+=1
                return [action,*result[1:]]
        def load(path):
            session,inp,out,name=original_load(path)
            return Session(session),inp,out,name
        module.load_onnx=load

    def before_step(self):
        m=self.module
        limit=np.asarray(m.TARGET_VELOCITY_LIMIT)*m.SIM_DT/m.ACTION_SCALE
        self.slew+=np.clip(self.applied-self.slew,-limit,limit)
        target=m.DEFAULT_DOF_POS+m.ACTION_SCALE*self.slew
        self.issued_ratio=max(self.issued_ratio,float(np.max(np.abs(target-self.last_target)/(self.velocity*m.SIM_DT))))
        self.last_target=target.copy()

    def sample(self,model,data,frame,qids,dids):
        t=float(data.time);omega=data.qvel[3:6].copy()
        thresholds=self.contract['execution_thresholds']
        contact=max(frame.forces.values())>thresholds['recovery_contact_force']
        body=any(c['force_N']>thresholds['unsafe_body_contact_force'] for c in frame.unexpected)
        self.runtime.sample(t,bool(contact),omega,bool(body))
        self.audit.sample(t,bool(contact),omega.tolist(),bool(frame.unexpected))
        if t<=1.+1e-7:
            self.entry_dq=max(self.entry_dq,float(np.abs(data.qvel[dids]).max()))
            self.entry_omega=max(self.entry_omega,float(np.linalg.norm(omega)))
        self.last_time=t
        self.samples.append(dict(time=t,foot_contact=bool(contact),body_contact=bool(frame.unexpected),
            omega_body=omega.tolist(),phase=int(self.runtime.state.phase[0]),
            execution_completed=int(self.runtime.state.completed[0]),phase_clock=float(self.runtime.clock[0])))

    def report(self):
        expected=round(self.last_time/self.module.SIM_DT)
        motion=self.audit.report(expected)
        duration_ok=self.args.seconds-self.module.SIM_DT<=self.last_time<=self.args.seconds+.02+self.module.SIM_DT
        motion['duration_covered']=duration_ok
        motion['motion_pass'] &= duration_ok
        # Issued target statistics are a read-only reconstruction of the pinned
        # motor loop; raw desired commands are checked separately at actor rate.
        targets=dict(passed=bool(self.target_ratio<=1.+1e-6 and self.issued_ratio<=1.+1e-6),
            raw_target_slew_urdf_ratio=self.target_ratio,issued_target_slew_urdf_ratio=self.issued_ratio,
            policy_calls=self.calls,first_target_checked_against_reset_pose=True)
        entry=dict(passed=bool(self.entry_dq<=3. and self.entry_omega<=1. and self.entry_target_step<=.08),
            duration_s=1.,max_joint_speed_rad_s=self.entry_dq,max_body_angular_speed_rad_s=self.entry_omega,
            max_target_step_rad=self.entry_target_step,scope='initial trained hold only, not full visual acceptance')
        (self.args.output/'rebound_physics_trace.json').write_text(json.dumps(self.samples))
        return dict(rebound_motion=motion,target_continuity=targets,entry_quiet=entry,
                    execution_contract=self.contract,visual_review_complete=False,hardware_qualified=False)


if __name__=='__main__':
    from .simulate import main
    main(extension=ReboundEvaluation())
