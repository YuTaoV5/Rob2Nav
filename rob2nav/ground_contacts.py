"""Foot-region ground contacts, including calf endcaps inside the actual foot pad."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

LEGS=('FR','FL','RR','RL')
RULE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def foot_collision_geometries(model):
    import mujoco
    result={}
    for leg in LEGS:
        found=[gid for gid in range(model.ngeom)
               if model.body(int(model.geom_bodyid[gid])).name.startswith(leg+'_')
               and model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_SPHERE
               and model.geom_contype[gid] and model.geom_size[gid,0]>0]
        if len(found)!=1:raise ValueError(f'Expected one unambiguous {leg} foot sphere, found {found}')
        result[leg]=found[0]
    return result


def foot_region_leg(body,point,centers,radii):
    """Non-sphere contacts qualify only within their OWN anatomical foot region."""
    for leg in LEGS:
        if body in (leg+'_foot',leg+'_calf') and np.linalg.norm(np.asarray(point)-centers[leg])<=radii[leg]+1.e-6:
            return leg
    return None


@dataclass
class GroundFrame:
    forces:dict
    bodies:list
    unexpected:list


class GroundContactReader:
    def __init__(self,model):
        self.model=model;self.feet=foot_collision_geometries(model)
        self.rule_sha256=RULE_SHA256
        self.direct={gid:leg for leg,gid in self.feet.items()}
        self.floors={i for i in range(model.ngeom) if int(model.geom_bodyid[i])==0}
        self.radii={leg:float(model.geom_size[gid,0]) for leg,gid in self.feet.items()}

    def sample(self,data):
        import mujoco
        centers={leg:data.geom_xpos[gid] for leg,gid in self.feet.items()}
        forces={leg:0. for leg in LEGS};bodies=[];unexpected=[];force=np.zeros(6)
        for ci,c in enumerate(data.contact):
            a,b=int(c.geom1),int(c.geom2)
            other=b if a in self.floors and b not in self.floors else a if b in self.floors and a not in self.floors else None
            if other is None:continue
            body=self.model.body(int(self.model.geom_bodyid[other])).name;bodies.append(body)
            leg=self.direct.get(other) or foot_region_leg(body,c.pos,centers,self.radii)
            mujoco.mj_contactForce(self.model,data,ci,force)
            normal=abs(float(force[0]))
            if leg is not None:forces[leg]+=normal
            else:
                own=next((x for x in LEGS if body.startswith(x+'_')),None)
                unexpected.append(dict(geom=other,body=body,force_N=normal,position=c.pos.tolist(),
                    own_foot_center_distance_m=float(np.linalg.norm(c.pos-centers[own])) if own else None))
        return GroundFrame(forces,bodies,unexpected)


class GroundContactValidation:
    def __init__(self,model):
        self.reader=GroundContactReader(model);self.hz=1./model.opt.timestep
        self.samples=0;self.bad_samples=0;self.violations={}

    def sample(self,data):
        frame=self.reader.sample(data);self.samples+=1;self.bad_samples+=bool(frame.unexpected)
        for row in frame.unexpected:
            key=str(row['geom'])
            record=self.violations.setdefault(key,dict(body=row['body'],first_time_s=float(data.time),
                last_time_s=float(data.time),max_force_N=0.,max_own_foot_center_distance_m=0.))
            record['last_time_s']=float(data.time);record['max_force_N']=max(record['max_force_N'],row['force_N'])
            if row['own_foot_center_distance_m'] is not None:
                record['max_own_foot_center_distance_m']=max(record['max_own_foot_center_distance_m'],row['own_foot_center_distance_m'])
        return frame

    def report(self,expected_samples):
        covered=self.samples==expected_samples and self.samples>0
        return dict(scope='every physical substep; foot spheres plus calf/foot contact points inside their own asset-defined foot region',
            physics_hz=self.hz,samples=self.samples,expected_samples=expected_samples,coverage_complete=covered,
            foot_radius_m=self.reader.radii,unexpected_samples=self.bad_samples,violations=self.violations,
            rule_sha256=self.reader.rule_sha256,
            passed=bool(covered and not self.bad_samples))
