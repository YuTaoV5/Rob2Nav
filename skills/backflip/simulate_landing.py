"""Landing-v2 rollout with independent raw-physics acceptance."""
from .simulate_rebound import ReboundEvaluation
from .landing_playback import LandingPlaybackState
from rob2nav.landing_validation import audit_landing_cycles


class RawLandingAudit:
    def __init__(self,dt,settings):
        self.dt=dt;self.settings=settings;self.rows=[];self.upright=None

    def sample(self,time,foot_contact,omega_body,body_contact=False):
        if self.upright is None:raise ValueError('Every physics sample needs measured orientation')
        self.rows.append(dict(time=time,foot_contact=foot_contact,omega_body=list(omega_body),
            body_contact=body_contact,upright=self.upright))
        self.upright=None

    def report(self,expected_samples):
        return audit_landing_cycles(self.rows,self.dt,self.settings,expected_samples)


class LandingEvaluation(ReboundEvaluation):
    runtime_type=LandingPlaybackState

    def create_audit(self,dt,settings):return RawLandingAudit(dt,settings)

    def sample(self,model,data,frame,qids,dids):
        q=data.qpos[3:7]
        upright=float(1.-2.*(q[1]*q[1]+q[2]*q[2]))
        self.runtime.upright=upright;self.audit.upright=upright
        super().sample(model,data,frame,qids,dids)
        self.samples[-1]['upright']=upright


if __name__=='__main__':
    from .simulate import main
    main(extension=LandingEvaluation())
