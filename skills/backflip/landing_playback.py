"""Execution of an audited landing-v2 actor; no physical state writes."""
import math
import torch
from .rebound_playback import ReboundPlaybackState
from .rebound_landing import LandingConfig,LandingPhases
from .landing_contract import VERSION,implementation_hashes


class LandingPlaybackState(ReboundPlaybackState):
    def __init__(self,contract,dt):
        if contract.get('version')!=VERSION or contract.get('phase_contract')!='landing_cycle_v2':
            raise ValueError('Landing-v2 export required')
        if contract.get('implementation_sha256')!=implementation_hashes():
            raise ValueError('Landing training source mismatch')
        self.settings=LandingConfig(**contract['settings'])
        self.thresholds=contract['execution_thresholds']
        for k in ('takeoff_start','recovery_contact_force','unsafe_body_contact_force'):
            v=self.thresholds[k]
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('Invalid execution threshold')
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid physics step')
        self.dt=dt;self.samples=0;self.upright=None
        self.state=LandingPhases(1,dt,self.settings)
        self.clock=torch.tensor([-self.settings.entry_seconds])

    def sample(self,elapsed,foot_contact,omega_body,body_contact=False):
        if not math.isfinite(elapsed) or abs(elapsed-(self.samples+1)*self.dt)>max(1e-7,self.dt*1e-4):
            raise ValueError('Missing or duplicated physics sample')
        if self.upright is None:raise ValueError('Measured orientation required for each sample')
        r=self.state
        r.step(torch.tensor([foot_contact]),torch.tensor([list(omega_body)],dtype=torch.float32),
               torch.tensor([body_contact]),torch.tensor([self.upright]))
        self.upright=None;self.samples+=1;self.clock+=self.dt
        if bool(r.just_landed[0]) and int(r.phase[0])==r.REBOUND:self.clock[0]=self.thresholds['takeoff_start']
        if int(r.phase[0])==r.RECOVERY:self.clock[0]=2.
