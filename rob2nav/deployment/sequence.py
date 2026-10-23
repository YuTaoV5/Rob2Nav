"""State-driven gait/atomic/gait choreography; it never owns a motor publisher."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .types import vector


@dataclass(frozen=True)
class Step:
    skill:str
    seconds:float=2.
    velocity:tuple[float,...]=(0.,0.,0.)
    trigger:bool=False
    handover_extension_s:float=60.
    def checked(self):
        if not self.skill or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in self.skill):raise ValueError('Invalid sequence skill ID')
        if not np.isfinite(self.seconds) or not 0<self.seconds<=120:raise ValueError('Invalid sequence dwell time')
        command=vector(self.velocity,3)
        if np.any(abs(command)>np.array([.5,.25,.7])):raise ValueError(f'速度 {tuple(self.velocity)} 超出指令范围：前后 ±0.5 m/s、左右 ±0.25 m/s、转向 ±0.7 rad/s')
        if type(self.trigger) is not bool:raise ValueError('Sequence trigger must be boolean')
        if not np.isfinite(self.handover_extension_s) or not 0<=self.handover_extension_s<=300:raise ValueError('Invalid terrain handover extension')
        return self


def load_steps(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict) or set(data)!={'steps'} or not isinstance(data['steps'],list) or not 1<=len(data['steps'])<=32:
        raise ValueError('Expected 1..32 explicit sequence steps')
    return [Step(**item).checked() for item in data['steps']]


class SkillSequence:
    def __init__(self,runtime,steps):
        self.runtime=runtime;self.steps=[s.checked() for s in steps]
        if not self.steps:raise ValueError('Empty skill sequence')
        if self.steps[-1].skill!='stand':self.steps.append(Step('stand'))
        for step in self.steps:
            _,contract=runtime.profile(step.skill)
            if contract.atomic!=step.trigger:raise ValueError('Atomic sequence steps require an explicit trigger; gait steps must not trigger')
        self.index=0;self.requested=False;self.started=None;self.done=False;self.error=None
        self.completed_dwell=0.;self.atomic_index=None;self.consumed_atomic_epoch=None
    def progress(self,now):
        """Read-only telemetry; ACTIVE dwell excludes preparation and handover waits."""
        c=self.runtime.controller;step=self.steps[self.index]
        elapsed=max(0.,now-self.started) if self.started is not None else 0.
        dwell=min(elapsed,step.seconds) if not step.trigger else 0.
        total=sum(s.seconds for s in self.steps if not s.trigger)
        credited=min(total,self.completed_dwell+dwell)
        atomic=bool(c.active is not None and c.active.atomic and c.phase=='ACTIVE')
        status=('complete' if self.done else 'failed' if self.error or c.phase=='FAULT' else
            'atomic' if atomic else 'transition' if self.started is None else
            'handover' if elapsed>=step.seconds else 'dwell')
        index=self.atomic_index if atomic and self.atomic_index is not None else self.index
        return dict(step_index=index,step_count=len(self.steps),skill=self.steps[index].skill,
            current_skill=c.active.id if c.active is not None else None,controller_phase=c.phase,
            status=status,step_dwell_s=0. if self.done or atomic else dwell,
            step_target_s=None if self.steps[index].trigger else self.steps[index].seconds,
            planned_dwell_s=total,dwell_elapsed_s=credited,execution_elapsed_s=max(0.,now),
            extra_elapsed_s=max(0.,now-credited),atomic_steps=sum(s.trigger for s in self.steps))
    @staticmethod
    def terrain_handover_ready(snapshot):
        # A terrain policy may remain pitched on a stair. Keep it active until
        # the body is back in a level transfer corridor; the core still demands
        # a slower, fully settled stance before committing the next policy.
        return bool(np.linalg.norm(snapshot.gravity-np.array([0.,0.,-1.]))<.18 and
            np.linalg.norm(snapshot.gyro)<.35 and np.count_nonzero(snapshot.contacts())>=3)
    def tick(self,snapshot,now):
        r=self.runtime;c=r.controller
        if self.done or self.error:c.set_velocity([0,0,0],now);return
        if c.phase=='FAULT' or r.stage_error:self.error=c.fault or r.stage_error;c.set_velocity([0,0,0],now);return
        step=self.steps[self.index]
        repeat_consumed=(step.trigger and c.active is not None and c.active.id==step.skill
                         and self.consumed_atomic_epoch is not None and c.epoch==self.consumed_atomic_epoch)
        if not self.requested:
            if c.phase=='ACTIVE' and c.active is not None and c.active.id==step.skill and not repeat_consumed:
                self.requested=True
            elif r.future is None and r.next is None and r.queued is None:
                r.stage_async(step.skill,snapshot);self.requested=True
        if repeat_consumed:c.set_velocity([0,0,0],now);return
        current=c.active.id if c.active is not None else None
        if current!=step.skill or c.phase not in ('ACTIVE','ARMED'):c.set_velocity([0,0,0],now);return
        if c.phase=='ARMED':
            if step.trigger and not c.trigger_requested:c.trigger()
            return
        if self.started is None:self.started=now
        c.set_velocity(step.velocity,now)
        # Queue the return gait as soon as the atomic action starts. The core,
        # not this sequencer's timer, enforces completion and landing stability.
        elapsed=now-self.started
        if not c.active.atomic and elapsed>=step.seconds and self.index<len(self.steps)-1:
            next_step=self.steps[self.index+1]
            if next_step.skill!=step.skill and not self.terrain_handover_ready(snapshot):
                if elapsed>step.seconds+step.handover_extension_s:
                    self.error='terrain handover window timed out';c.set_velocity([0,0,0],now)
                return
        if c.active.atomic or elapsed>=step.seconds:
            if c.active.atomic:self.atomic_index=self.index;self.consumed_atomic_epoch=c.epoch
            else:self.completed_dwell+=step.seconds
            if self.index==len(self.steps)-1:self.done=True;c.set_velocity([0,0,0],now)
            else:self.index+=1;self.requested=False;self.started=None
