"""Generate physically supported non-inverted recovery pose diagnostics."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from train_stair_low_speed_course import OmniEnvironment


def quaternion(roll, pitch):
    sr,cr=np.sin(roll/2),np.cos(roll/2)
    sp,cp=np.sin(pitch/2),np.cos(pitch/2)
    return np.array((cr*cp,sr*cp,cr*sp,-sr*sp))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--seed",type=int,default=292)
    sides = p.add_mutually_exclusive_group()
    sides.add_argument("--lateral-bank",action="store_true")
    sides.add_argument("--shallow-side-bank",action="store_true")
    a=p.parse_args()
    e=OmniEnvironment(a.assets,1,stop_training=True)
    e.reset(seed=29201,options={"scene_seed":29201,"target":(0.,0.,0.)})
    rng=np.random.default_rng(a.seed)
    low=e.defaults[:12]+.90*(e.leg_low-e.defaults[:12])
    high=e.defaults[:12]+.90*(e.leg_high-e.defaults[:12])
    data=e.data
    output={}
    for family in ("prone","left","right"):
        rows=[]
        for _ in range(3200 if a.lateral_bank and family != "prone" else 1600):
            roll=rng.uniform(-.10,.10) if family=="prone" else (
                rng.uniform(.48,1.15)*(1 if family=="left" else -1))
            if a.lateral_bank and family != "prone":
                roll=rng.uniform(.95,1.53)*(-1 if family=="left" else 1)
            if a.shallow_side_bank and family != "prone":
                roll=rng.uniform(.38,.96)*(-1 if family=="left" else 1)
            pitch=rng.uniform(-.25,.25)
            q=np.r_[rng.uniform(-.16,.16,4),rng.uniform(2.25,2.90,4),
                    rng.uniform(-2.05,-1.35,4)]
            if a.lateral_bank and family != "prone":
                q[:4] = rng.uniform(-.26,.26,4)
                q[4:8] = rng.uniform(1.85,3.15,4)
                q[8:12] = rng.uniform(-2.45,-1.15,4)
            if a.shallow_side_bank and family != "prone":
                q=e.defaults[:12]+rng.uniform(-.09,.09,12)
                q[4:8] += rng.uniform(.34,.46)
                q[8:12] -= rng.uniform(.58,.72)
            q=np.clip(q,low+.08,high-.08)
            data.qpos[e.qa[:12]]=q
            data.qpos[3:7]=quaternion(roll,pitch)
            lo,hi=.01,.60
            for _ in range(15):
                mid=(lo+hi)/2
                data.qpos[2]=mid
                mujoco.mj_forward(e.model,data)
                if any(e.ground in c.geom and c.dist < -.0005 for c in data.contact):
                    lo=mid
                else:
                    hi=mid
            data.qpos[2]=lo
            mujoco.mj_forward(e.model,data)
            grounded = {e.model.geom_bodyid[c.geom[1] if c.geom[0]==e.ground else c.geom[0]]
                        for c in data.contact if e.ground in c.geom and c.dist < 0.}
            lateral = a.lateral_bank and family != "prone"
            supports = grounded-set(e.wheel_index)
            shallow = a.shallow_side_bank and family != "prone"
            upper = .46 if shallow else .42 if lateral else .30
            supported = bool(grounded) if shallow else bool(supports) if lateral else e.base in supports
            if not supported or not .04 < hi < upper:
                continue
            self_collision=any(e.ground not in c.geom and c.dist < -.004 for c in data.contact)
            if self_collision:
                continue
            rows.append(dict(leg_q=q.tolist(),height_m=hi+.001,root_wxyz=data.qpos[3:7].tolist(),
                             roll_rad=float(roll),pitch_rad=float(pitch),
                             supporting_body="base" if e.base in supports else e.model.body(min(supports or grounded)).name))
            if len(rows)>=32:
                break
        output[family]=rows
        print(family,len(rows),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()