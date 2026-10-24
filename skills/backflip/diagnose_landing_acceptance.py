"""Counterfactual frozen actor; exercise independent v2 physics audit only."""
import argparse
from pathlib import Path
import sys
from .diagnose_landing_drive import LandingDriveDiagnostic
from .simulate_landing import RawLandingAudit
from .rebound_landing import LandingConfig
from dataclasses import asdict


class LandingAcceptanceDiagnostic(LandingDriveDiagnostic):
    def create_audit(self,dt,settings):
        return RawLandingAudit(dt,asdict(LandingConfig(**settings)))

    def sample(self,model,data,frame,qids,dids):
        q=data.qpos[3:7]
        upright=float(1.-2.*(q[1]*q[1]+q[2]*q[2]))
        self.audit.upright=upright
        super().sample(model,data,frame,qids,dids)
        self.samples[-1]['upright']=upright

    def report(self):
        result=super().report()
        result['landing_clock_diagnostic']['independent_v2_acceptance_implemented']=True
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--rebound-env',required=True,type=Path)
    a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
    from .simulate import main
    main(extension=LandingAcceptanceDiagnostic(a.rebound_env))
