"""Pad a motion reference with standard stance and smooth bridges; not physical validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .fixstand import POSE


def padded_motion(raw,fps=50.,hold=1.,bridge=.8):
    if raw.ndim!=2 or raw.shape[1]!=19 or not np.isfinite(raw).all():raise ValueError('Expected finite 19-column motion CSV')
    names=[f'{leg}_{joint}_joint' for leg in ('FR','FL','RR','RL') for joint in ('hip','thigh','calf')]
    q=np.array([POSE[name] for name in names])
    def stand(frame):
        result=frame.copy();result[2]=.32;result[7:]=q
        yaw=Rotation.from_quat(frame[3:7]).as_euler('xyz')[2]
        result[3:7]=Rotation.from_euler('z',yaw).as_quat()
        return result
    start,end=stand(raw[0]),stand(raw[-1])
    def transition(a,b):
        t=np.linspace(0,1,round(bridge*fps)+2)[1:-1];w=10*t**3-15*t**4+6*t**5
        frames=a[None,:]+w[:,None]*(b-a)[None,:]
        frames[:,3:7]=Slerp([0,1],Rotation.from_quat(np.stack([a[3:7],b[3:7]])))(w).as_quat()
        return frames
    return np.concatenate([np.tile(start,(round(hold*fps),1)),transition(start,raw[0]),raw,
                           transition(raw[-1],end),np.tile(end,(round(hold*fps),1))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    raw=np.loadtxt(args.csv,delimiter=',');result=padded_motion(raw)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savetxt(args.output,result,delimiter=',',fmt='%.9f')
    args.output.with_suffix('.padding.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        frames=len(result),fps=50,hold_s=1.,bridge_s=.8,standard_pose=POSE,
        role='reference preparation only; not evidence of learned endpoints'),indent=2)+'\n')


if __name__=='__main__':main()
