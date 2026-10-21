"""State-driven gait/atomic/gait choreography; it never owns a motor publisher."""
from dataclasses import dataclass
import numpy as np
from .types import vector


@dataclass(frozen=True)
class Step:
    skill:str
    seconds:float=2.
    velocity:tuple[float,...]=(0.,0.,0.)
    trigger:bool=False
    def checked(self):
        if not self.skill or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in self.skill):raise ValueError('Invalid sequence skill ID')
        if not np.isfinite(self.seconds) or not 0<self.seconds<=120:raise ValueError('Invalid sequence dwell time')
        vector(self.velocity,3);return self


class SkillSequence:
    def __init__(self,runtime,steps):
        self.runtime=runtime;self.steps=[s.checked() for s in steps]
        if not self.steps:raise ValueError('Empty skill sequence')
        if self.steps[-1].skill!='stand':self.steps.append(Step('stand'))
        self.index=0;self.requested=False;self.started=None;self.done=False;self.error=None
    def tick(self,snapshot,now):
        r=self.runtime;c=r.controller
        if self.done or self.error:c.set_velocity([0,0,0],now);return
        if c.phase=='FAULT' or r.stage_error:self.error=c.fault or r.stage_error;c.set_velocity([0,0,0],now);return
        step=self.steps[self.index]
        if not self.requested:
            if r.future is None and r.next is None and r.queued is None:r.stage_async(step.skill,snapshot);self.requested=True
        current=c.active.id if c.active is not None else None
        if current!=step.skill or c.phase not in ('ACTIVE','ARMED'):c.set_velocity([0,0,0],now);return
        if c.phase=='ARMED':
            if step.trigger and not c.trigger_requested:c.trigger()
            return
        if self.started is None:self.started=now
        c.set_velocity(step.velocity,now)
        # Queue the return gait as soon as the atomic action starts. The core,
        # not this sequencer's timer, enforces completion and landing stability.
        if c.active.atomic or now-self.started>=step.seconds:
            if self.index==len(self.steps)-1:self.done=True;c.set_velocity([0,0,0],now)
            else:self.index+=1;self.requested=False;self.started=None
