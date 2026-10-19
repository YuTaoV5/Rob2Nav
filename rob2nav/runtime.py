"""Transport-neutral control lifecycle and command validation; no motor bridge."""
from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol


class Mode(str,Enum):
    STANDBY='standby'
    ARMED='armed'
    ACTIVE='active'
    FAULT='fault'
    STOPPED='stopped'


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float
    wz: float
    stamp: float
    frame: str='base_link'

    def checked(self,now,limits=(1.0,.5,1.0),ttl=.25):
        if self.frame!='base_link':raise ValueError('Command must be transformed into base_link')
        if not all(math.isfinite(x) for x in (self.vx,self.vy,self.wz,self.stamp,now)):raise ValueError('Nonfinite command')
        if self.stamp>now+.01 or now-self.stamp>ttl:raise ValueError('Stale/future command')
        return VelocityCommand(*(max(-l,min(l,x)) for x,l in zip((self.vx,self.vy,self.wz),limits)),self.stamp)


class Transport(Protocol):
    def read(self): ...
    def write(self,command): ...
    def stop(self): ...


class DeploymentRuntime:
    """Owns the command gate. Adapters own observations, inference and joint PD."""
    def __init__(self,transport,clock):
        self.transport=transport;self.clock=clock;self.mode=Mode.STANDBY;self.skill=None;self.events=[]

    def _event(self,name):self.events.append({'time':self.clock(),'event':name,'mode':self.mode.value,'skill':self.skill})

    def arm(self,contract_validated=False):
        if self.mode!=Mode.STANDBY or not contract_validated:raise ValueError('Validate backend and policy contract before arming')
        self.mode=Mode.ARMED;self._event('arm')

    def activate(self,skill):
        if self.mode!=Mode.ARMED or not skill:raise ValueError('Activate only from armed state')
        self.skill=skill;self.mode=Mode.ACTIVE;self._event('activate')

    def command(self,command):
        if self.mode!=Mode.ACTIVE:raise ValueError('Control is not active')
        try:
            safe=command.checked(self.clock());self.transport.write(safe);return safe
        except Exception:
            self.mode=Mode.FAULT;self.transport.stop();self._event('command_fault');raise

    def hold(self):
        if self.mode!=Mode.ACTIVE:raise ValueError('Hold requires active state')
        self.transport.stop();self.mode=Mode.ARMED;self._event('hold')

    def stop(self):
        self.transport.stop();self.mode=Mode.STOPPED;self._event('stop')
