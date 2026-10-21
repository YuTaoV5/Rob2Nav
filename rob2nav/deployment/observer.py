"""Contact-constrained proprioceptive velocity and static support geometry.

Uses measured joints/IMU/forces with a fixed Go2 model, never physics integration.
Common-mode foot slip remains unobservable; no absolute position is invented.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
from .types import SDK_JOINTS


def model_fingerprint(model,source):
    import mujoco
    blob=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,buffer=blob)
    h=hashlib.sha256();h.update(mujoco.__version__.encode());h.update(source);h.update(blob.tobytes())
    return h.hexdigest()


def support_margin(points,com):
    points=sorted(set(tuple(map(float,p)) for p in np.asarray(points)[:,:2]))
    if len(points)<3:return None
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=lower[:-1]+upper[:-1]
    if len(hull)<3:return None
    return float(min(cross(a,b,com)/max(np.linalg.norm(np.subtract(b,a)),1e-10) for a,b in zip(hull,hull[1:]+hull[:1])))


class KinematicObserver:
    def __init__(self,model):
        import mujoco
        self.mj=mujoco;self.model=model;self.data=mujoco.MjData(model)
        self.base=next((model.body(n).id for n in ['base_link','base'] if any(model.body(i).name==n for i in range(model.nbody))),None)
        ids=[model.joint(n).id for n in SDK_JOINTS];self.qids=model.jnt_qposadr[ids];self.dids=model.jnt_dofadr[ids]
        self.feet=[]
        for leg in ['FR','FL','RR','RL']:
            named=[i for i in range(model.ngeom) if model.geom(i).name==leg]
            if named:self.feet.append(named[0]);continue
            body=model.body(leg+'_foot').id
            self.feet.append(next(i for i in range(model.ngeom) if model.geom_bodyid[i]==body and model.geom_contype[i]))
        self.source_sha256=''
        self.contact_on=np.ones(4)*8.;self.contact_off=np.ones(4)*5.;self.contacts=np.zeros(4,dtype=bool)
    @classmethod
    def from_xml(cls,path):
        import mujoco
        path=Path(path);observer=cls(mujoco.MjModel.from_xml_path(str(path)))
        # Hash the compiled model too: hashing only the root XML would miss
        # changed included masses, COMs, collision assets or actuator parameters.
        observer.source_sha256=model_fingerprint(observer.model,path.read_bytes());return observer
    def enrich(self,s):
        if s.quaternion is None:return s
        m,d,mj=self.model,self.data,self.mj
        # This separate MjData is FK only. Lift its arbitrary origin above the
        # scene floor to avoid spending time solving irrelevant model contacts.
        d.qpos[:3]=[0.,0.,3.];d.qpos[3:7]=s.quaternion;d.qpos[self.qids]=s.q
        d.qvel[:3]=0.;d.qvel[3:6]=s.gyro;d.qvel[self.dids]=s.dq;mj.mj_forward(m,d)
        self.contacts=np.where(self.contacts,s.feet>self.contact_off,s.feet>self.contact_on)
        stance=np.flatnonzero(self.contacts)
        if len(stance)<2:return replace(s,velocity=None,support_margin=None,contact_mask=self.contacts.copy())
        velocities=[];positions=[];jac=np.zeros((3,m.nv));jacr=np.zeros_like(jac);support=d.qfrc_bias[self.dids].copy()
        for index in stance:
            geom=self.feet[index];pos=d.geom_xpos[geom];mj.mj_jac(m,d,jac,jacr,pos,int(m.geom_bodyid[geom]))
            positions.append(pos.copy());velocities.append(jac@d.qvel)
        # A balanced nominal load, rather than copying an already unloaded foot,
        # helps the fourth foot regain contact during a slow neutral transition.
        all_positions=np.asarray([d.geom_xpos[g].copy() for g in self.feet]);com=d.subtree_com[self.base]
        a=np.vstack([np.ones(4),all_positions[:,0]-com[0],all_positions[:,1]-com[1]])
        desired=np.array([float(m.body_subtreemass[self.base])*9.81,0.,0.]);available=list(range(4));loads=np.zeros(4)
        while available:
            solved=np.linalg.lstsq(a[:,available],desired,rcond=None)[0]
            if np.min(solved)>=0:loads[available]=solved;break
            available.pop(int(np.argmin(solved)))
        for index,geom in enumerate(self.feet):
            mj.mj_jac(m,d,jac,jacr,d.geom_xpos[geom],int(m.geom_bodyid[geom]))
            support-=jac[:,self.dids].T@np.array([0.,0.,loads[index]])
        velocities=np.asarray(velocities);vbase=-np.average(velocities,axis=0,weights=s.feet[stance])
        residual=float(np.max(np.linalg.norm(velocities+vbase,axis=1)))
        if residual>.05:return replace(s,velocity=None,support_margin=None,contact_mask=self.contacts.copy())
        mj.mj_jacSubtreeCom(m,d,jac,self.base)
        velocity=d.xmat[self.base].reshape(3,3).T@(vbase+jac@d.qvel)
        margin=support_margin(positions,d.subtree_com[self.base,:2])
        return replace(s,velocity=velocity,support_margin=margin,support_torque=support,contact_mask=self.contacts.copy())
