"""Frozen single-flip actor under proposed rebound phases: never qualification.

No export metadata is rewritten. The intended experiment uses an explicit
actual rebound env.yaml as a counterfactual execution specification only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import yaml
from rob2nav.export_actor import ConfigLoader
from .rebound import ReboundConfig
from .rebound_contract import VERSION,implementation_hashes
from .simulate_rebound import ReboundEvaluation


class FrozenParentDiagnostic(ReboundEvaluation):
    counterfactual=True
    def __init__(self, env_file):
        self.env_file=Path(env_file)

    def configure(self,module,exported,args):
        parent=exported.get('fixstand_roundtrip') or {}
        if parent.get('action_parameterization') not in ('backflip_entry_v2','backflip_entry_v3'):
            raise ValueError('This diagnostic requires a known single-flip parent')
        env=yaml.load(self.env_file.read_text(),Loader=ConfigLoader)
        settings=ReboundConfig(**env['rebound'])
        if env['entry_delay_seconds']!=1. or env['entry_hold_scale']!=parent['entry_hold_scale']:
            raise ValueError('Diagnostic must preserve parent entry control')
        from .control import control_contract
        if control_contract(env['control'])!=exported['control_contract']:
            raise ValueError('Diagnostic must preserve the parent motor contract')
        hypothetical=dict(version=VERSION,settings=asdict(settings),episode_seconds=3.+2.*settings.cycles,
            phase_contract='contact_cycle_v1',entry_hold_scale=env['entry_hold_scale'],
            implementation_sha256=implementation_hashes(),capability_verified=False,
            counterfactual=True,execution_thresholds={k:env['rewards'][k] for k in
                ('takeoff_start','recovery_contact_force','unsafe_body_contact_force')})
        self.provenance=dict(parent_policy_sha256=exported['policy_sha256'],
            specification_yaml_sha256=hashlib.sha256(self.env_file.read_bytes()).hexdigest(),
            intervention='current rebound phase clock instead of the trained single-flip clock',
            weights_changed=False,training_contract_verified=False,hardware_qualified=False)
        super().configure(module,{**exported,'rebound_contract':hypothetical},args)
        from .rebound_landing import LandingConfig, LandingPhases
        self.landing_observer=LandingPhases(1,module.SIM_DT,LandingConfig(**asdict(settings)))
        self.landing_events=[]

    def sample(self,model,data,frame,qids,dids):
        # Observation only: the original v1 runtime still drives actor phase features.
        super().sample(model,data,frame,qids,dids)
        import torch
        q=data.qpos[3:7]
        upright=float(1.-2.*(q[1]*q[1]+q[2]*q[2]))
        r=self.landing_observer
        old=int(r.phase[0])
        contact=max(frame.forces.values())>self.contract['execution_thresholds']['recovery_contact_force']
        r.step(torch.tensor([bool(contact)]),torch.tensor([data.qvel[3:6].tolist()]),
               torch.tensor([bool(frame.unexpected)]),torch.tensor([upright]))
        if old!=int(r.phase[0]):
            self.landing_events.append(dict(time_s=float(data.time),phase=int(r.phase[0]),
                completed=int(r.completed[0]),upright=upright,rebound_elapsed_s=float(r.rebound_time[0])))

    def report(self):
        result=super().report()
        result['counterfactual_diagnostic']=self.provenance
        result['landing_counter_observer']=dict(events=self.landing_events,
            completed=int(self.landing_observer.completed[0]),controls_actor=False,
            capability_verified=False,scope='candidate v2 bookkeeping on unchanged v1 physical rollout')
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rebound-env',required=True,type=Path)
    args,rest=parser.parse_known_args()
    sys.argv=[sys.argv[0],*rest]
    from .simulate import main
    main(extension=FrozenParentDiagnostic(args.rebound_env))
