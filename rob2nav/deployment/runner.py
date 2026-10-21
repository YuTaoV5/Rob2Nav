"""Async weight staging and one control owner; only canonical commands cross the boundary."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,replace
import hashlib
from pathlib import Path
import time
import numpy as np
from .core import SwitchController
from .types import Skill,Intent
from .policies import PolicyProcess,observation


@dataclass
class Prepared:
    skill:Skill
    worker:PolicyProcess|None
    last:np.ndarray
    previous:np.ndarray
    intent:Intent|None=None
    next_inference:float=0.
    reference:object=None
    reference_yaw:float=0.


class UnifiedRuntime:
    def __init__(self,root,hardware=False,allow_research_stunts=False):
        self.root=Path(root).resolve();self.controller=SwitchController(hardware)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='policy-staging')
        self.future=None;self.active=None;self.next=None;self.queued=None;self.stage_error=None;self.started=0.;self.prepared=[]
        self.allow_research_stunts=allow_research_stunts and not hardware
    def profile(self,id):
        if not id or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in id):raise ValueError('Invalid profile ID')
        path=self.root/'deployment/skills'/(id+'.json')
        return path,Skill.load(path,self.root)
    def stage_async(self,profile_id,snapshot):
        if self.future is not None or self.next is not None or self.queued is not None:raise ValueError('A switch transaction is already pending')
        self.stage_error=None
        def prepare():
            path,skill=self.profile(profile_id)
            if self.controller.hardware and not skill.hardware_qualified:raise ValueError('Hardware skill quarantined: '+skill.blocker)
            if not self.controller.hardware and not skill.simulation_enabled:
                if self.allow_research_stunts and skill.atomic:skill=replace(skill,simulation_enabled=True)
                else:raise ValueError('Simulation switch quarantined: '+skill.blocker)
            if skill.entry!='four_feet':raise ValueError('Entry/exit controller has not been validated')
            worker=None
            try:
                reference=None;yaw=0.
                if skill.observation=='mimic69':
                    file=(self.root/skill.reference).resolve()
                    if not file.is_relative_to(self.root) or hashlib.sha256(file.read_bytes()).hexdigest()!=skill.reference_sha256:raise ValueError('Motion reference hash mismatch')
                    reference=np.load(file,allow_pickle=False)
                    from .policies import rotation
                    live=rotation(snapshot.quaternion);ref=rotation(reference['body_quat_w'][0,0]);yaw=float(np.arctan2(live[1,0],live[0,0])-np.arctan2(ref[1,0],ref[0,0]))
                action=(snapshot.q-skill.to_sdk(skill.default))/skill.scale
                if skill.policy:
                    cached=next((p for p in reversed(self.prepared) if p.skill==skill and p.worker is not None and p.worker.process.is_alive() and p is not self.active),None)
                    worker=cached.worker if cached is not None else PolicyProcess(path,self.root);deadline=time.monotonic()+15
                    while time.monotonic()<deadline and not worker.ready and not worker.error:
                        worker.poll();time.sleep(.005)
                    if not worker.ready or worker.error:raise ValueError('Candidate load failure: '+str(worker.error))
                    while worker.inflight and time.monotonic()<deadline:worker.poll();time.sleep(.002)
                    if worker.inflight:raise ValueError('Cached worker did not drain its previous output')
                    obs=observation(skill,snapshot,[0,0,0],action,action,reference=reference,reference_yaw=yaw)
                    if not worker.submit(obs,-1,0.):raise ValueError('Candidate probe could not be submitted')
                    deadline=time.monotonic()+3;result=None
                    while time.monotonic()<deadline:
                        result=worker.poll()
                        if result is not None:break
                        time.sleep(.002)
                    if result is None or result.get('epoch')!=-1 or 'target' not in result or not np.isfinite(result['target']).all():raise ValueError('Candidate dry inference failed')
                    worker.pipe.send(dict(reset=True));deadline=time.monotonic()+2;reset=None
                    while time.monotonic()<deadline:
                        reset=worker.poll()
                        if reset is not None:break
                        time.sleep(.002)
                    if not reset or not reset.get('reset_done'):raise ValueError('Candidate recurrent-state reset failed')
                    # The initial probe is discarded. Actual state/history observations warm
                    # the candidate before commit; no model output is sent to motors here.
                prepared=Prepared(skill,worker,action.copy(),action.copy(),reference=reference,reference_yaw=yaw);self.prepared.append(prepared)
                keep={id(worker),id(self.active.worker) if self.active else None};inactive=[]
                for old in reversed(self.prepared):
                    if old.worker is not None and id(old.worker) not in keep and old.worker.process.is_alive():
                        if len(inactive)<1:inactive.append(old.worker);keep.add(id(old.worker))
                        else:old.worker.close()
                return prepared
            except Exception:
                if worker:worker.close()
                raise
        self.future=self.executor.submit(prepare)
    def arm(self,snapshot,now):self.controller.arm(snapshot,now)
    def step(self,snapshot,now):
        c=self.controller
        if self.future is not None and self.future.done():
            try:self.queued=self.future.result()
            except Exception as exc:self.stage_error=repr(exc);c.event(now,'staging_rejected:'+self.stage_error)
            self.future=None
        if self.queued is not None and c.phase in ('READY','ACTIVE') and c.pending is None:
            try:c.request(self.queued.skill,now);self.next=self.queued
            except ValueError as exc:self.stage_error=str(exc);c.event(now,'switch_rejected:'+str(exc))
            self.queued=None
        candidate=self.next if c.phase in ('WARMUP','BLEND') else self.active if c.phase=='ACTIVE' else None
        intent=None
        if candidate:
            skill=candidate.skill
            if candidate.worker is None:intent=Intent(c.epoch,now,skill.to_sdk(skill.default))
            else:
                try:result=candidate.worker.poll()
                except (EOFError,OSError) as exc:result={'error':repr(exc)}
                if result and 'error' in result:c.latch(now,'policy process failed:'+result['error'])
                elif result and 'target' in result:
                    candidate.intent=Intent(result['epoch'],result['stamp'],np.asarray(result['target']),finished=now-self.started>skill.atomic_duration_s+skill.recovery_s)
                if now>=candidate.next_inference and c.phase!='FAULT':
                    effective=(c.servo.q-skill.to_sdk(skill.default))/skill.scale
                    elapsed=max(0.,now-self.started) if c.phase=='ACTIVE' else 0.
                    try:
                        obs=observation(skill,snapshot,c.command,effective,candidate.last,phase=elapsed,reference=candidate.reference,reference_yaw=candidate.reference_yaw)
                        if candidate.worker.submit(obs,c.epoch,now):candidate.previous=candidate.last.copy();candidate.last=effective.copy();candidate.next_inference=now+1/skill.policy_hz
                    except ValueError as exc:c.latch(now,'observation adapter:'+str(exc))
                intent=candidate.intent
        command=c.tick(snapshot,now,intent)
        if c.phase=='ACTIVE' and self.next is not None and c.active is self.next.skill:
            previous=self.active;self.active=self.next;self.next=None;self.started=now
            # Retain at most one inactive model for fast return after an atomic skill.
            # Preparation evicts extra inactive workers off the control thread.
        return command
    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        for prepared in self.prepared:
            if prepared.worker:prepared.worker.close()
            if prepared.reference is not None:prepared.reference.close()
