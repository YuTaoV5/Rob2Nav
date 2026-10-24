"""Policy phase state for MuJoCo; physical state is never written here.

Mirrors the exported training phase semantics, with independent acceptance in
rob2nav.rebound_validation. Call after each physics step, before the next actor call.
"""
import math
import torch
from .rebound import ReboundConfig, ReboundPhases
from .rebound_contract import VERSION, implementation_hashes


class ReboundPlaybackState:
    def __init__(self, contract, dt):
        if contract.get('version') != VERSION or contract.get('phase_contract') != 'contact_cycle_v1':
            raise ValueError('Unsupported rebound execution semantics')
        if contract.get('implementation_sha256') != implementation_hashes():
            raise ValueError('Playback training implementation hash mismatch')
        self.settings = ReboundConfig(**contract['settings'])
        self.thresholds = contract['execution_thresholds']
        for key in ('takeoff_start','recovery_contact_force','unsafe_body_contact_force'):
            value=self.thresholds[key]
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Invalid exported phase/contact threshold')
        self.dt=dt;self.samples=0
        self.state=ReboundPhases(1,dt,self.settings)
        self.clock=torch.full((1,),-self.settings.entry_seconds)

    def sample(self, elapsed, foot_contact, omega_body, body_contact=False):
        if not math.isfinite(elapsed) or abs(elapsed-(self.samples+1)*self.dt)>max(1e-7,self.dt*1e-4):
            raise ValueError('Missing or duplicate rebound physics step')
        self.state.step(torch.tensor([foot_contact],dtype=torch.bool),
                        torch.tensor([list(omega_body)],dtype=torch.float32),
                        torch.tensor([body_contact],dtype=torch.bool))
        self.samples+=1
        self.clock+=self.dt
        r=self.state
        if bool(r.just_landed[0]) and int(r.phase[0])==r.REBOUND:
            self.clock[0]=self.thresholds['takeoff_start']
        if int(r.phase[0])==r.RECOVERY:
            self.clock[0]=2.

    def phase_features(self):
        phase=torch.pi*self.clock.clamp(max=2.)/2.
        return torch.stack(tuple(f(phase/d) for d in (1.,2.,4.) for f in (torch.sin,torch.cos)),dim=-1)[0].numpy()
