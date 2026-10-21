"""500 Hz pure control core. No model loading, Torch, DDS, disk I/O or sleeps."""
import numpy as np
from dataclasses import replace
from .types import SDK_JOINTS,Snapshot,Skill,Intent,MotorCommand,vector

LOW=np.array([-1.0472,-1.5708,-2.7227]*2+[-1.0472,-.5236,-2.7227]*2)+.08
HIGH=np.array([1.0472,3.4907,-.83776]*2+[1.0472,4.5379,-.83776]*2)-.08
EFFORT=np.array([23.7,23.7,45.43]*4)
FREE_SPEED=np.array([30.1,30.1,15.7]*4)
NEUTRAL=np.array([0.,.8,-1.5,0.,.8,-1.5,0.,1.,-1.5,0.,1.,-1.5])


def quintic(x):
    x=float(np.clip(x,0,1));return 10*x**3-15*x**4+6*x**5


class Servo:
    def __init__(self,s):
        self.q=s.q.copy();self.dq=s.dq.copy();self.kp=np.zeros(12);self.kd=np.zeros(12)
        self.tau=np.clip(s.tau,-EFFORT,EFFORT);self.bias=self.tau.copy();self.envelope_conflicts=0
    def match_bias(self,s,capture=False):
        if capture:self.q=s.q.copy();self.dq=np.zeros(12)
        self.bias=np.clip(self.tau-self.kp*(self.q-s.q)-self.kd*(self.dq-s.dq),-.5*EFFORT,.5*EFFORT)
    def native(self,s,target,kp,kd,phase,epoch):
        self.q=vector(target);self.dq=np.zeros(12);self.kp=vector(kp);self.kd=vector(kd);self.bias=np.zeros(12)
        speed=np.clip(s.dq,-2*FREE_SPEED,2*FREE_SPEED)
        low=np.maximum(EFFORT*(-1-speed/FREE_SPEED),-EFFORT);high=np.minimum(EFFORT*(1-speed/FREE_SPEED),EFFORT)
        self.tau=np.clip(self.kp*(self.q-s.q)-self.kd*s.dq,low,high)
        return MotorCommand(self.tau.copy(),self.q.copy(),self.dq.copy(),self.kp.copy(),self.kd.copy(),phase,epoch)
    def step(self,s,target,kp,kd,dt,phase,epoch,fault='',integrate=False,bias_scale=1.):
        target=np.clip(vector(target),LOW,HIGH)
        acceleration=np.clip(100*(target-self.q)-20*self.dq,-20,20)
        self.dq=np.clip(self.dq+dt*acceleration,-2.,2.)
        self.q=np.clip(self.q+dt*self.dq,LOW,HIGH)
        self.kp+=np.clip(vector(kp)-self.kp,-15*dt,15*dt)
        self.kd+=np.clip(vector(kd)-self.kd,-2*dt,2*dt)
        if integrate and s.stable():self.bias=np.clip(self.bias+20*(self.q-s.q)*dt,-.5*EFFORT,.5*EFFORT)
        raw=self.kp*(self.q-s.q)+self.kd*(self.dq-s.dq)+self.bias*bias_scale
        speed=np.clip(s.dq,-2*FREE_SPEED,2*FREE_SPEED)
        low=np.maximum(EFFORT*(-1-speed/FREE_SPEED),-EFFORT)
        high=np.minimum(EFFORT*(1-speed/FREE_SPEED),EFFORT)
        lo=np.maximum(low,self.tau-150*dt);hi=np.minimum(high,self.tau+150*dt)
        if np.any(lo>hi):
            self.envelope_conflicts+=1
            fault=fault or 'torque envelope/rate intersection empty'
            tau=np.clip(raw,low,high)
        else:tau=np.clip(raw,lo,hi)
        self.tau=tau
        return MotorCommand(tau.copy(),self.q.copy(),self.dq.copy(),self.kp.copy(),self.kd.copy(),phase,epoch,fault)


class SwitchController:
    """Two-phase skill activation. Atomic skills cannot be interrupted by a switch."""
    def __init__(self,hardware=False):
        self.hardware=hardware;self.phase='OFF';self.active=None;self.pending=None;self.epoch=0
        self.servo=None;self.last_tick=None;self.last_sequence=None;self.sequence_at=0.;self.stable_since=None
        self.phase_at=0.;self.fault='';self.events=[];self.start_pose=None;self.exit_ready=False;self.warm_since=None
        self.command=np.zeros(3);self.requested_command=np.zeros(3)
    def set_velocity(self,command):self.requested_command=np.clip(vector(command,3),[-.5,-.25,-.7],[.5,.25,.7])
    def event(self,now,name):self.events.append(dict(time=now,event=name,phase=self.phase,epoch=self.epoch))
    def arm(self,s,now):
        if self.phase!='OFF':raise ValueError('Already armed; fault reset needs a fresh runtime')
        s.checked(now)
        if not s.stable():raise ValueError('Acquire control only in a measured stable four-foot stance')
        if np.any(s.q<LOW) or np.any(s.q>HIGH):raise ValueError('Measured joints outside transition workspace')
        self.servo=Servo(s);self.start_pose=s.q.copy();self.phase='CAPTURE';self.phase_at=now;self.last_tick=now
        self.last_sequence=s.sequence;self.sequence_at=now;self.event(now,'capture_measured_state')
    def request(self,skill,now):
        skill.checked()
        if self.phase in ('OFF','FAULT'):raise ValueError('Controller is not ready')
        if self.pending is not None or self.phase not in ('ACTIVE','READY'):raise ValueError('Switch transaction already active')
        if self.hardware and not skill.hardware_qualified:raise ValueError('Unqualified hardware skill: '+skill.blocker)
        if not self.hardware and not skill.simulation_enabled:raise ValueError('Quarantined skill: '+skill.blocker)
        default=skill.entry_sdk()
        if np.any(default<LOW) or np.any(default>HIGH):raise ValueError('Entry pose outside joint workspace')
        if skill.entry!='four_feet':raise ValueError('No validated stand/entry or exit controller for this skill')
        self.pending=skill;self.stable_since=None;self.event(now,'request:'+skill.id)
    def latch(self,now,reason):
        if self.phase!='FAULT':self.phase='FAULT';self.fault=reason;self.event(now,'fault:'+reason)
    def _phase(self,name,now):self.phase=name;self.phase_at=now;self.stable_since=None;self.event(now,name)
    def tick(self,s,now,intent=None):
        if self.phase=='OFF':return None
        dt=now-self.last_tick
        if not np.isfinite(dt) or dt<=0:raise ValueError('Control clock must advance monotonically')
        if dt>.01:self.latch(now,'control deadline exceeded')
        dt=min(dt,.01);self.last_tick=now
        velocity_goal=self.requested_command if self.phase=='ACTIVE' and self.pending is None else np.zeros(3)
        self.command+=np.clip(velocity_goal-self.command,-np.array([.5,.5,.7])*dt,np.array([.5,.5,.7])*dt)
        try:
            s.checked(now)
            if s.sequence!=self.last_sequence:self.last_sequence=s.sequence;self.sequence_at=now
            elif now-self.sequence_at>.03:raise ValueError('state sequence stopped')
            if np.any(s.q<LOW-.12) or np.any(s.q>HIGH+.12):raise ValueError('joint position envelope')
            research_native=not self.hardware and self.active is not None and self.active.actuation=='native_pd' and self.phase=='ACTIVE'
            if np.any(abs(s.dq)>FREE_SPEED*(1.2 if research_native else 1.05)):raise ValueError('joint speed envelope')
        except ValueError as exc:self.latch(now,str(exc))
        if not all(np.isfinite(x).all() for x in [s.q,s.dq]):raise ValueError('Cannot compute torque from nonfinite state; transport must enter independently qualified damping')
        elapsed=now-self.phase_at
        target=self.servo.q.copy();kp=np.full(12,30.);kd=np.full(12,1.5)
        integrate=True;bias_scale=1.
        stable=s.stable()
        if stable:
            if self.stable_since is None:self.stable_since=now
        else:self.stable_since=None
        settled=self.stable_since is not None and now-self.stable_since>=.3
        if self.phase=='FAULT':target=s.q.copy();kp*=0;kd[:]=2.
        elif self.phase=='CAPTURE':
            target=self.start_pose
            if elapsed>=1. and settled:self.start_pose=self.servo.q.copy();self._phase('NEUTRAL',now)
            elif elapsed>4:self.latch(now,'measured-state capture did not stabilize')
        elif self.phase=='NEUTRAL':
            target=self.start_pose+quintic(elapsed/2.)*(NEUTRAL-self.start_pose)
            if elapsed>=2. and settled and np.max(abs(s.q-NEUTRAL))<.10:self._phase('READY',now)
            elif elapsed>6:self.latch(now,'neutral acquisition timeout')
        elif self.phase=='READY':
            target=NEUTRAL
            if self.pending and settled:
                self.start_pose=self.servo.q.copy();self._phase('ENTRY',now)
        elif self.phase=='ACTIVE':
            integrate=self.active.observation=='stand';bias_scale=1. if integrate else 0.
            if intent is None:self.latch(now,'missing policy output')
            else:
                try:intent.checked(self.epoch,now)
                except ValueError as exc:self.latch(now,str(exc))
                if self.phase!='FAULT':
                    target=intent.target;kp=self.active.to_sdk(self.active.kp);kd=self.active.to_sdk(self.active.kd)
                    quiescent_tripod=not self.active.atomic and s.safe_tripod() and np.linalg.norm(self.command)<.01 and elapsed>2
                    if self.pending and (settled or quiescent_tripod) and (not self.active.atomic or intent.finished):
                        self.servo.match_bias(s,capture=self.active.actuation=='native_pd');integrate=True;bias_scale=1.
                        self.start_pose=self.servo.q.copy();self._phase('EXIT',now)
        elif self.phase=='EXIT':
            target=self.start_pose+quintic(elapsed/2.)*(NEUTRAL-self.start_pose)
            if elapsed>=2. and settled and np.max(abs(s.q-NEUTRAL))<.10:
                self.start_pose=self.servo.q.copy();self._phase('ENTRY',now)
            elif elapsed>6:self.latch(now,'exit did not recover stable neutral stance')
        elif self.phase=='ENTRY':
            goal=self.pending.entry_sdk()
            target=self.start_pose+quintic(elapsed/2.)*(goal-self.start_pose)
            kp=self.pending.to_sdk(self.pending.kp);kd=self.pending.to_sdk(self.pending.kd)
            if elapsed>=2. and settled and np.max(abs(s.q-goal))<.10:
                self.epoch+=1;self.warm_since=None;self._phase('WARMUP',now)
            elif elapsed>6:self.latch(now,'entry pose/gain transition timeout')
        elif self.phase in ('WARMUP','BLEND'):
            target=self.pending.entry_sdk();kp=self.pending.to_sdk(self.pending.kp);kd=self.pending.to_sdk(self.pending.kd)
            valid=False
            invalid_reason='no candidate output'
            if intent is not None:
                try:
                    intent.checked(self.epoch,now)
                    baseline=target if self.phase=='WARMUP' else self.servo.q
                    delta=float(np.max(abs(intent.target-baseline)))
                    valid=delta<.40;invalid_reason=f'candidate target step {delta:.4f} rad'
                except ValueError as exc:invalid_reason=str(exc)
            if self.phase=='WARMUP':
                if valid and stable:
                    if self.warm_since is None:self.warm_since=now
                    if now-self.warm_since>self.pending.warmup_s:self._phase('BLEND',now)
                else:self.warm_since=None
                if elapsed>self.pending.warmup_s+3:self.latch(now,'shadow policy warmup failed')
            elif not valid:self.latch(now,'candidate during blend: '+invalid_reason)
            else:
                target=target+quintic(elapsed)*(intent.target-target)
                if self.pending.observation!='stand':bias_scale=1-quintic(elapsed);integrate=False
                if elapsed>=1.:
                    self.active=self.pending;self.pending=None;self._phase('ACTIVE',now);self.event(now,'commit:'+self.active.id)
        if self.phase=='FAULT':target=s.q.copy();kp=np.zeros(12);kd=np.full(12,2.);integrate=False;bias_scale=0.
        if self.phase=='ACTIVE' and self.active.actuation=='native_pd':
            command=self.servo.native(s,target,kp,kd,self.phase,self.epoch)
        else:command=self.servo.step(s,target,kp,kd,dt,self.phase,self.epoch,self.fault,integrate=integrate,bias_scale=bias_scale)
        if command.fault and self.phase!='FAULT':self.latch(now,command.fault)
        return replace(command,stamp=now)
