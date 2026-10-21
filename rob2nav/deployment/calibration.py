"""Device-specific sign/zero calibration; all shipped examples are unverified."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .types import SDK_JOINTS,vector


@dataclass(frozen=True)
class Calibration:
    signs:tuple[float,...]=(1.,)*12
    offsets:tuple[float,...]=(0.,)*12
    verified:bool=False
    imu_frame:str='native_go2_imu_body'
    foot_scales:tuple[float,...]=(1.,)*4
    foot_offsets:tuple[float,...]=(0.,)*4
    foot_indices:tuple[int,...]=(0,1,2,3)

    def checked(self):
        if not np.isin(vector(self.signs),[-1.,1.]).all() or np.max(abs(vector(self.offsets)))>.3:raise ValueError('Invalid device sign/zero calibration')
        if self.imu_frame!='native_go2_imu_body':raise ValueError('IMU extrinsic adapter not qualified for this frame')
        if np.any(vector(self.foot_scales,4)<=0) or np.any(vector(self.foot_scales,4)>100) or set(self.foot_indices)!={0,1,2,3}:raise ValueError('Invalid force unit/order calibration')
        vector(self.foot_offsets,4)
        return self
    def digest(self):
        data=dict(joints=SDK_JOINTS,signs=self.signs,offsets=self.offsets,verified=self.verified,imu_frame=self.imu_frame,foot_scales=self.foot_scales,foot_offsets=self.foot_offsets,foot_indices=self.foot_indices)
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def position_from_sdk(self,q):return vector(self.signs)*(vector(q)-vector(self.offsets))
    def vector_from_sdk(self,x):return vector(self.signs)*vector(x)
    def torque_to_sdk(self,x):return vector(self.signs)*vector(x)
    def position_to_sdk(self,q):return vector(self.signs)*vector(q)+vector(self.offsets)
    def forces_from_sdk(self,x):return np.maximum(0.,(vector(x,4)[list(self.foot_indices)]-vector(self.foot_offsets,4))*vector(self.foot_scales,4))

    @classmethod
    def load(cls,path):
        data=json.loads(path.read_text(encoding='utf-8'))
        if tuple(data.pop('joints'))!=SDK_JOINTS:raise ValueError('Calibration joint order differs')
        for k in ['signs','offsets','foot_scales','foot_offsets','foot_indices']:
            if k in data:data[k]=tuple(data[k])
        return cls(**data).checked()
