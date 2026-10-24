"""Explicit frozen-parent intervention: candidate landing phases drive the clock.

Never changes model metadata or certifies capability. No robot state writes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from .diagnose_rebound_parent import FrozenParentDiagnostic
from .simulate_rebound import ReboundEvaluation
from .rebound_landing import LandingConfig, LandingPhases


class DiagnosticLandingClock:
    def __init__(self, settings, dt, takeoff_start):
        self.state=LandingPhases(1,dt,settings)
        self.dt=dt; self.samples=0; self.takeoff_start=takeoff_start
        self.clock=torch.tensor([-settings.entry_seconds])
        self.upright=None
        self.events=[]

    def sample(self, elapsed, foot_contact, omega_body, body_contact=False):
        if not math.isfinite(elapsed) or abs(elapsed-(self.samples+1)*self.dt)>max(1e-7,self.dt*1e-4):
            raise ValueError('Missing physics sample')
        if self.upright is None:
            raise ValueError('Measured orientation required before sampling')
        r=self.state; old=int(r.phase[0])
        r.step(torch.tensor([foot_contact]),torch.tensor([list(omega_body)],dtype=torch.float32),
               torch.tensor([body_contact]),torch.tensor([self.upright]))
        self.samples+=1; self.clock+=self.dt
        before=float(self.clock[0])
        if bool(r.just_landed[0]) and int(r.phase[0])==r.REBOUND:
            self.clock[0]=self.takeoff_start
        if int(r.phase[0])==r.RECOVERY:
            self.clock[0]=2.
        if old!=int(r.phase[0]):
            self.events.append(dict(time_s=elapsed,phase=int(r.phase[0]),completed=int(r.completed[0]),
                upright=self.upright,rebound_elapsed_s=float(r.rebound_time[0]),
                clock_before=before,clock_after=float(self.clock[0])))
        self.upright=None

    def phase_features(self):
        phase=torch.pi*self.clock.clamp(max=2.)/2.
        return torch.stack(tuple(f(phase/d) for d in (1.,2.,4.) for f in (torch.sin,torch.cos)),dim=-1)[0].numpy()


class LandingDriveDiagnostic(FrozenParentDiagnostic):
    def configure(self,module,exported,args):
        super().configure(module,exported,args)
        settings=LandingConfig(**self.contract['settings'])
        self.runtime=DiagnosticLandingClock(settings,module.SIM_DT,self.contract['execution_thresholds']['takeoff_start'])
        self.provenance['intervention']='candidate landing-aware phase clock drives frozen single-flip actor'
        self.policy_trace=[]
        original_load=module.load_onnx
        outer=self
        class TracedSession:
            def __init__(self,inner):self.inner=inner
            def run(self,*a,**kw):
                result=self.inner.run(*a,**kw)
                outer.policy_trace.append(dict(time_s=(outer.calls-1)*.02,
                    phase_clock=float(outer.runtime.clock[0]),
                    mapped_action=result[0].reshape(-1).tolist()))
                return result
        def load(path):
            session,inp,out,name=original_load(path)
            return TracedSession(session),inp,out,name
        module.load_onnx=load
        self.contract['diagnostic_execution_override']=dict(version='landing_clock_candidate_v2',
            settings=asdict(settings),controls_actor=True,training_contract_verified=False,
            source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('rebound_landing.py','diagnose_landing_drive.py')})

    def sample(self,model,data,frame,qids,dids):
        q=data.qpos[3:7]
        self.runtime.upright=float(1.-2.*(q[1]*q[1]+q[2]*q[2]))
        ReboundEvaluation.sample(self,model,data,frame,qids,dids)

    def report(self):
        result=ReboundEvaluation.report(self)
        (self.args.output/'landing_policy_trace.json').write_text(json.dumps(self.policy_trace))
        result['counterfactual_diagnostic']=self.provenance
        result['landing_clock_diagnostic']=dict(events=self.runtime.events,
            completed=int(self.runtime.state.completed[0]),controls_actor=True,
            capability_verified=False,independent_v2_acceptance_implemented=False)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rebound-env',required=True,type=Path)
    args,rest=parser.parse_known_args()
    sys.argv=[sys.argv[0],*rest]
    from .simulate import main
    main(extension=LandingDriveDiagnostic(args.rebound_env))
