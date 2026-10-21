"""Compare all reference-frame forward kinematics across native Isaac data and MuJoCo."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from .common import pinned,BACKFLIP_COMMIT


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--motion',required=True,type=Path)
 p.add_argument('--backflip-source',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 a=p.parse_args()
 source=pinned(a.backflip_source,BACKFLIP_COMMIT)
 contract=json.loads(a.motion.with_suffix('.json').read_text())
 digest=hashlib.sha256(a.motion.read_bytes()).hexdigest()
 if contract['npz_sha256']!=digest:raise ValueError('Reference hash mismatch')
 ref=np.load(a.motion,allow_pickle=False)
 xml=source/'mujoco/go2/go2_isaac.xml'
 m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
 qids=[m.jnt_qposadr[m.joint(n).id] for n in contract['joint_names']]
 ids=[m.body(n).id for n in contract['body_names']]
 base=contract['body_names'].index('base')
 position_error=0.;orientation_error=0.
 for frame in range(len(ref['joint_pos'])):
  d.qpos[:3]=ref['body_pos_w'][frame,base]
  d.qpos[3:7]=ref['body_quat_w'][frame,base]
  d.qpos[qids]=ref['joint_pos'][frame]
  mujoco.mj_forward(m,d)  # FK only, intentionally no dynamics step.
  error=np.linalg.norm(d.xpos[ids]-ref['body_pos_w'][frame],axis=1)
  q=np.asarray(ref['body_quat_w'][frame],dtype=float)
  q/=np.linalg.norm(q,axis=1,keepdims=True)
  cosine=np.abs(np.sum(d.xquat[ids]*q,axis=1))
  angle=2*np.arccos(np.clip(cosine,-1,1))
  position_error=max(position_error,float(error.max()))
  orientation_error=max(orientation_error,float(angle.max()))
 passed=position_error<=1e-5 and orientation_error<=1e-4
 result=dict(role='cross-engine FK contract check only; not dynamics/skill acceptance',
  frames=len(ref['joint_pos']),bodies=len(ids),max_position_error_m=position_error,max_orientation_error_rad=orientation_error,
  limits=dict(position_m=1e-5,orientation_rad=1e-4),passed=passed,
  reference_sha256=digest,source_commit=BACKFLIP_COMMIT,mjcf_sha256=hashlib.sha256(xml.read_bytes()).hexdigest())
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
 if not passed:raise SystemExit(1)

if __name__=='__main__':main()
