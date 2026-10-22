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
    support_model_gain:float=0.
    action_parameterization:str='joint_offset_v1'
    residual_limit_rad:float=0.
    bridge_residual_limit_rad:float=0.
    entry_hold_scale:float=.4
    endpoint_residual_limit_rad:float=.2
    terminal_hip_taper_frames:int=0
    boundary_blend_frames:int=0
    motor_control:dict|None=None
    entry_torque_blend_s:float=0.

    def checked(self):
        if len(self.joints)!=12 or set(self.joints)!=set(SDK_JOINTS):raise ValueError('Joint-name mapping must be a complete Go2 bijection')
        for x in [self.default,self.kp,self.kd]:vector(x)
        if min(self.kp)<0 or max(self.kp)>80 or min(self.kd)<0 or max(self.kd)>8:raise ValueError('PD outside configured bound')
        if not np.isfinite([self.policy_hz,self.scale]).all() or self.policy_hz not in (25,50,100) or self.scale<=0:raise ValueError('Invalid policy timing/scale')
        if self.observation not in ('stand','robogauge45','backflip60','upright48','upright52','mimic69'):raise ValueError('Unsupported observation adapter')
        if self.motor_control is not None:
            from skills.backflip.control import control_contract
            c=control_contract(self.motor_control)
            if self.observation!='backflip60' or self.actuation!='native_pd':raise ValueError('Backflip motor contract requires native backflip control')
            if self.scale!=c['action_scale'] or c['action_clip']!=8. or any(k!=c['stiffness'] for k in self.kp) or any(k!=c['damping'] for k in self.kd):raise ValueError('Backflip action/PD and motor contracts disagree')
        if not np.isfinite(self.entry_torque_blend_s) or not 0<=self.entry_torque_blend_s<=.4 or (self.entry_torque_blend_s and self.motor_control is None):raise ValueError('Entry torque blending requires a validated motor contract and at most 0.4 seconds')
        compatible={'joint_offset_v1':None,'reference_residual_v2':'mimic69','reference_residual_v3':'mimic69','reference_residual_v4':'mimic69','backflip_entry_v2':'backflip60','backflip_entry_v3':'backflip60','upright_endpoint_v2':'upright52','upright_endpoint_v3':'upright52'}
        if self.action_parameterization not in compatible or (compatible[self.action_parameterization] is not None and self.observation!=compatible[self.action_parameterization]):raise ValueError('Action and observation contracts disagree')
        if self.motor_control is not None and self.action_parameterization not in ('backflip_entry_v2','backflip_entry_v3'):raise ValueError('Motor contract requires the delayed-entry action mapping')
        if self.action_parameterization in ('reference_residual_v2','reference_residual_v3','reference_residual_v4'):
            if not self.reference or not .01<=self.bridge_residual_limit_rad<=self.residual_limit_rad<=.3:raise ValueError('Invalid sideflip residual contract')
        elif self.residual_limit_rad or self.bridge_residual_limit_rad:raise ValueError('Residual limits require reference residual action')
        if self.action_parameterization in ('reference_residual_v3','reference_residual_v4'):
            taper=self.terminal_hip_taper_frames
            if type(taper) is not int or not (2<=taper<=50 or self.action_parameterization=='reference_residual_v4' and taper==0):raise ValueError('Invalid terminal hip taper')
        elif self.terminal_hip_taper_frames:raise ValueError('Terminal hip taper requires reference_residual_v3')
        if self.action_parameterization=='reference_residual_v4':
            if type(self.boundary_blend_frames) is not int or not 2<=self.boundary_blend_frames<=20:raise ValueError('Invalid continuous boundary width')
        elif self.boundary_blend_frames:raise ValueError('Boundary blending requires reference_residual_v4')
        if not np.isfinite(self.entry_hold_scale) or not 0<=self.entry_hold_scale<=.4:raise ValueError('Invalid backflip entry hold scale')
        if self.action_parameterization=='backflip_entry_v2' and self.entry_hold_scale!=.4:raise ValueError('Backflip v2 hold scale is 0.4')
        if self.action_parameterization not in ('backflip_entry_v2','backflip_entry_v3') and self.entry_hold_scale!=.4:raise ValueError('Entry hold scale requires backflip contract')
        if not np.isfinite(self.endpoint_residual_limit_rad) or not .01<=self.endpoint_residual_limit_rad<=.2:raise ValueError('Invalid upright endpoint limit')
        if self.action_parameterization=='upright_endpoint_v2' and self.endpoint_residual_limit_rad!=.2:raise ValueError('Upright v2 endpoint limit is 0.2')
        if self.action_parameterization!='upright_endpoint_v3' and self.endpoint_residual_limit_rad!=.2:raise ValueError('Upright limit requires v3 contract')
        if self.entry not in ('four_feet','requires_exit_controller'):raise ValueError('Unsupported entry condition')
        if not np.isfinite(self.warmup_s) or not .3<=self.warmup_s<=3:raise ValueError('Invalid history warmup duration')
        if self.actuation not in ('bounded_pd','native_pd'):raise ValueError('Unknown motor execution contract')
        if not np.isfinite([self.atomic_duration_s,self.recovery_s]).all() or self.atomic_duration_s<0 or self.recovery_s<1:raise ValueError('Invalid atomic phase duration')
        if self.entry_pose is not None:vector(self.entry_pose)
        if not np.isfinite(self.entry_tolerance) or not .005<=self.entry_tolerance<=.15:raise ValueError('Invalid entry state tolerance')
        if abs(np.linalg.norm(vector(self.entry_gravity,3))-1)>.01 or not .03<=self.entry_gravity_tolerance<=.2:raise ValueError('Invalid gravity entry contract')
        if not np.isfinite(self.support_model_gain) or not 0<=self.support_model_gain<=4:raise ValueError('Invalid entry load-model gain')
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
