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

    def checked(self):
        if not np.isin(vector(self.signs),[-1.,1.]).all() or np.max(abs(vector(self.offsets)))>.3:raise ValueError('Invalid device sign/zero calibration')
        if self.imu_frame!='native_go2_imu_body':raise ValueError('IMU extrinsic adapter not qualified for this frame')
        return self
    def digest(self):
        data=dict(joints=SDK_JOINTS,signs=self.signs,offsets=self.offsets,verified=self.verified,imu_frame=self.imu_frame)
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def position_from_sdk(self,q):return vector(self.signs)*(vector(q)-vector(self.offsets))
    def vector_from_sdk(self,x):return vector(self.signs)*vector(x)
    def torque_to_sdk(self,x):return vector(self.signs)*vector(x)

    @classmethod
    def load(cls,path):
        data=json.loads(path.read_text(encoding='utf-8'))
        if tuple(data.pop('joints'))!=SDK_JOINTS:raise ValueError('Calibration joint order differs')
        for k in ['signs','offsets']:data[k]=tuple(data[k])
        return cls(**data).checked()
