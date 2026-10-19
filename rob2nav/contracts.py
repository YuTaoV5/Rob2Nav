"""Explicit policy contracts. Equal action dimensions do not imply compatibility."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class PolicyContract:
    id: str
    robot: str
    joints: tuple[str, ...]
    observation: str
    action: str
    policy_hz: float
    physics_hz: float
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    backend: str
    source: str
    validation: str = 'inference only; independent skill evaluation pending'

    def validate(self):
        n=len(self.joints)
        if not self.id or not self.observation or not self.action:raise ValueError('Incomplete policy contract')
        if not n or len(set(self.joints))!=n:raise ValueError('Joint names must be unique')
        if len(self.kp)!=n or len(self.kd)!=n:raise ValueError('PD gains must match joint order')
        values=(*self.kp,*self.kd,self.policy_hz,self.physics_hz)
        if not all(math.isfinite(x) and x>=0 for x in values) or self.policy_hz<=0:raise ValueError('Invalid gains/rates')
        ratio=self.physics_hz/self.policy_hz
        if ratio<1 or not math.isclose(ratio,round(ratio)):raise ValueError('Physics/policy rates require integer decimation')
        return self

    def manifest(self):return asdict(self.validate())


def profiles():
    physical=tuple(f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf'))
    mapped=tuple(physical[i] for i in (3,0,9,6,4,1,10,7,5,2,11,8))
    pie_joints=tuple(f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf'))
    contracts=[PolicyContract('pie','go2',pie_joints,'45 proprio + 450 history + 2x60x86 depth + recurrent state',
        'source ONNX joint targets; source default pose and scale .25',50,200,(20,20,40)*4,(1,1,2)*4,
        'pinned_source_mujoco','WwWv0v/go2_PIE@fb0db87924a9731913ad1011381c01db6945a9c8')]
    for name in ('lsd_jump','lsd_backflip','lsd_sideflip'):
        # Tracking gains are read from each pinned source config by the backend;
        # do not advertise a guessed shared PD contract here.
        contracts.append(dict(id=name,robot='go2',joints=mapped,observation='69 motion-reference conditioned values',
            action='source reference tracking joint targets',policy_hz=50,physics_hz=200,
            gains='resolved from pinned per-skill source config at runtime',backend='pinned_source_mujoco',
            source='haozhang04/LeggedSkillDeploy@9126dad84708fc2714c6a1158897c86844d0cc1c',
            validation='inference recorded; sideflip landing failed; no autonomous obstacle decision'))
    return [x.manifest() if isinstance(x,PolicyContract) else x for x in contracts]
