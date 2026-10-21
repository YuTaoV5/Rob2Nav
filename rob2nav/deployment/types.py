"""Canonical SDK-order state and strict per-policy deployment contracts."""
from dataclasses import dataclass,asdict,field
import hashlib
import json
from pathlib import Path
import numpy as np

SDK_JOINTS=tuple(f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf'))
SDK_COMMIT='814556d15970dd2ecf1c9984e845ca02ab07e206'


def vector(x,n=12):
    a=np.asarray(x,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'Expected {n} finite values')
    return a.copy()


@dataclass(frozen=True)
class Snapshot:
    stamp:float
    sequence:int
    q:np.ndarray
    dq:np.ndarray
    tau:np.ndarray
    gyro:np.ndarray
    gravity:np.ndarray
    feet:np.ndarray
    velocity:np.ndarray|None=None
    quaternion:np.ndarray|None=None
    support_margin:float|None=None
    support_torque:np.ndarray|None=None
    contact_mask:np.ndarray|None=None

    def checked(self,now,ttl=.03):
        if not np.isfinite([self.stamp,now]).all() or not -.002<=now-self.stamp<=ttl:raise ValueError('stale/future state')
        for x,n in [(self.q,12),(self.dq,12),(self.tau,12),(self.gyro,3),(self.gravity,3),(self.feet,4)]:vector(x,n)
        if not .9<=np.linalg.norm(self.gravity)<=1.1:raise ValueError('invalid gravity normalization')
        if self.velocity is not None:vector(self.velocity,3)
        if self.quaternion is not None and abs(np.linalg.norm(vector(self.quaternion,4))-1)>.01:raise ValueError('invalid quaternion')
        if self.support_torque is not None:vector(self.support_torque)
        if self.contact_mask is not None and not np.isin(vector(self.contact_mask,4),[0,1]).all():raise ValueError('invalid contact mask')
        return self
    def contacts(self):return np.asarray(self.contact_mask,dtype=bool) if self.contact_mask is not None else self.feet>8
    def entry_stable(self,skill):
        goal=np.array(skill.entry_gravity)
        return bool(np.all(self.contacts()) and np.linalg.norm(self.gyro)<.25 and np.max(abs(self.dq))<.8 and
            np.linalg.norm(self.gravity-goal)<skill.entry_gravity_tolerance and
            (self.velocity is None and not skill.atomic or self.velocity is not None and np.linalg.norm(self.velocity[:2])<.12))
    def safe_tripod(self):
        return bool(self.support_margin is not None and np.isfinite(self.support_margin) and self.support_margin>.015 and
            np.count_nonzero(self.contacts())>=3 and self.velocity is not None and np.linalg.norm(self.velocity[:2])<.08 and
            np.linalg.norm(self.gyro)<.18 and np.max(abs(self.dq))<2. and self.gravity[2]<-.99)
    def controllable_support(self):
        return bool(self.support_margin is not None and self.support_margin>.015 and np.count_nonzero(self.contacts())>=3 and
            self.velocity is not None and np.linalg.norm(self.velocity[:2])<.12 and np.linalg.norm(self.gyro)<.35 and
            np.max(abs(self.dq))<2. and self.gravity[2]<-.94)

    def stable(self):
        return bool(np.all(self.contacts()) and np.linalg.norm(self.gyro)<.25 and np.max(abs(self.dq))<.8 and
            self.gravity[2]<-.98 and (self.velocity is None or np.linalg.norm(self.velocity[:2])<.12))


@dataclass(frozen=True)
class Skill:
    id:str
    joints:tuple[str,...]
    default:tuple[float,...]
    kp:tuple[float,...]
    kd:tuple[float,...]
    observation:str
    policy_hz:float
    scale:float
    policy:str=''
    sha256:str=''
    atomic:bool=False
    entry:str='four_feet'
    hardware_qualified:bool=False
    blocker:str='No physical qualification'
    simulation_enabled:bool=True
    reference:str=''
    reference_sha256:str=''
    warmup_s:float=1.
    actuation:str='bounded_pd'
    atomic_duration_s:float=0.
    recovery_s:float=2.
    entry_pose:tuple[float,...]|None=None
    entry_tolerance:float=.1
    entry_gravity:tuple[float,...]=(0.,0.,-1.)
    entry_gravity_tolerance:float=.12

    def checked(self):
        if len(self.joints)!=12 or set(self.joints)!=set(SDK_JOINTS):raise ValueError('Joint-name mapping must be a complete Go2 bijection')
        for x in [self.default,self.kp,self.kd]:vector(x)
        if min(self.kp)<0 or max(self.kp)>80 or min(self.kd)<0 or max(self.kd)>8:raise ValueError('PD outside configured bound')
        if not np.isfinite([self.policy_hz,self.scale]).all() or self.policy_hz not in (25,50,100) or self.scale<=0:raise ValueError('Invalid policy timing/scale')
        if self.observation not in ('stand','robogauge45','backflip60','upright48','mimic69'):raise ValueError('Unsupported observation adapter')
        if self.entry not in ('four_feet','requires_exit_controller'):raise ValueError('Unsupported entry condition')
        if not np.isfinite(self.warmup_s) or not .3<=self.warmup_s<=3:raise ValueError('Invalid history warmup duration')
        if self.actuation not in ('bounded_pd','native_pd'):raise ValueError('Unknown motor execution contract')
        if not np.isfinite([self.atomic_duration_s,self.recovery_s]).all() or self.atomic_duration_s<0 or self.recovery_s<1:raise ValueError('Invalid atomic phase duration')
        if self.entry_pose is not None:vector(self.entry_pose)
        if not np.isfinite(self.entry_tolerance) or not .005<=self.entry_tolerance<=.15:raise ValueError('Invalid entry state tolerance')
        if abs(np.linalg.norm(vector(self.entry_gravity,3))-1)>.01 or not .03<=self.entry_gravity_tolerance<=.2:raise ValueError('Invalid gravity entry contract')
        return self

    def to_sdk(self,values):
        values=vector(values)
        return np.array([values[self.joints.index(n)] for n in SDK_JOINTS])

    def from_sdk(self,values):
        values=vector(values)
        return np.array([values[SDK_JOINTS.index(n)] for n in self.joints])
    def entry_sdk(self):return self.to_sdk(self.entry_pose if self.entry_pose is not None else self.default)
    def digest(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    @classmethod
    def load(cls,path,root):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        for key in ('joints','default','kp','kd'):d[key]=tuple(d[key])
        s=cls(**d).checked()
        if s.policy:
            file=(Path(root)/s.policy).resolve()
            if not file.is_relative_to(Path(root).resolve()) or hashlib.sha256(file.read_bytes()).hexdigest()!=s.sha256:raise ValueError('Policy path/hash mismatch')
        return s


@dataclass(frozen=True)
class Intent:
    epoch:int
    stamp:float
    target:np.ndarray
    finished:bool=False

    def checked(self,epoch,now):
        if self.epoch!=epoch:raise ValueError('obsolete policy generation')
        if not np.isfinite(self.stamp) or not -.002<=now-self.stamp<=.06:raise ValueError('stale/future policy output')
        vector(self.target)
        return self


@dataclass(frozen=True)
class MotorCommand:
    tau:np.ndarray
    qref:np.ndarray
    dqref:np.ndarray
    kp:np.ndarray
    kd:np.ndarray
    phase:str
    epoch:int
    fault:str=''
    stamp:float=0.
    feedforward:np.ndarray=field(default_factory=lambda:np.zeros(12))
