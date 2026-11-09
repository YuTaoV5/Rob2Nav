"""Recheck the complete flat command suites with front-foot geometry and self contacts."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import torch

from export_r152_onnx import OnnxActor, session_for
from as2w_contacts import exclude_nominal_overlaps
import evaluate_omni_stairs as ev
from evaluate_omni_joint import make_controller
from train_omni_stop_transition import load


@lru_cache(maxsize=2)
def runtime(checkpoint):
    return load(assets,checkpoint)


@lru_cache(maxsize=1)
def ort_session():
    return session_for(onnx_path)


@lru_cache(maxsize=2)
def flat_model(filtered):
    path = assets/"local-step-diagnostic/as2w_flat.xml"
    if filtered:
        model = exclude_nominal_overlaps(ev.sim.mj.MjSpec.from_file(str(path))).compile()
    else:
        model = ev.sim.load_flat_model(True,True,seed=0)
    ground = model.geom("ground").id
    model.geom_priority[ground] = int(model.geom_priority.max())+1
    return model


video=None


def evaluate(job):
    checkpoint, split, name, target, seed, filtered = job
    policy, contract, limits, base, controller = runtime(source if checkpoint.suffix==".onnx" else checkpoint)
    actor = (OnnxActor(ort_session(), np.asarray(contract["default_joint_pos"]))
             if checkpoint.suffix==".onnx" else make_controller(policy,contract,controller,base()))
    model = flat_model(filtered)
    body = model.body("base_link").id
    wheels = [model.body(n).id for n in contract["joint_names"][12:]]
    force = np.zeros(6)
    front_sides = []
    rear_sides = []
    peak_self = peak_nonwheel = 0.
    first_ground_hit = None
    defaults = np.asarray(contract["default_joint_pos"])

    def plan(unused_model, unused_data, obs, command, tick, advance):
        frame = obs.copy()
        frame[3:9] = frame[3:9].astype(np.float32)
        frame[10:26] = (frame[10:26]+defaults).astype(np.float32).astype(np.float64)-defaults
        frame[26:42] = frame[26:42].astype(np.float32)
        return actor.plan(None,None,frame,command,tick,None)

    def capture(model,data,command):
        nonlocal peak_self,peak_nonwheel,first_ground_hit
        if video is not None:
            video.capture(model,data,command)
        rotation = data.xmat[body].reshape(3,3)
        wheel_centers = (data.xpos[wheels]-data.xpos[body])@rotation
        front_sides.append(float(min(wheel_centers[0,1],-wheel_centers[1,1])))
        rear_sides.append(float(min(wheel_centers[2,1],-wheel_centers[3,1])))
        for i,c in enumerate(data.contact):
            a,b = (int(model.geom_bodyid[g]) for g in c.geom)
            if a and b:
                ev.sim.mj.mj_contactForce(model,data,i,force)
                peak_self = max(peak_self,force[0])
            elif (a or b) not in wheels:
                ev.sim.mj.mj_contactForce(model,data,i,force)
                peak_nonwheel = max(peak_nonwheel,force[0])
                if first_ground_hit is None and force[0]>=50.:
                    first_ground_hit = dict(tick=len(front_sides)-1,body=model.body(a or b).name,
                                            force_n=float(force[0]))

    row = ev.flat_episode(None,contract,limits,np.asarray(target),seed,ev.gait_clock,plan,capture,model=model)
    qualified = row["passed"] and min(front_sides)>0. and peak_self<1. and peak_nonwheel<50.
    return dict(row,split=split,command=name,seed=seed,target=list(target),filtered=filtered,
                front_min=min(front_sides),rear_min=min(rear_sides),
                front_cross_ticks=sum(v<=0. for v in front_sides),qualified=qualified,
                peak_self=float(peak_self),peak_nonwheel=float(peak_nonwheel),first_ground_hit=first_ground_hit)


if __name__ == "__main__":
    assets, source, onnx_path, output = (Path(a).resolve() for a in sys.argv[1:5])
    ev.sim.ROOT=assets
    torch.set_num_threads(1)
    fixed = [("fixed",name,target,seed) for name,target in ev.FLAT_COMMANDS.items()
             for seed in range(81201,81205)]
    rng = np.random.default_rng(91291)
    held = [("independent",f"mixed_{i:02d}",target,91101+i)
            for i,target in enumerate(rng.uniform((-.6,-.35,-.8),(.6,.35,.8),size=(48,3)))]
    held.extend(("independent","stop",np.zeros(3),seed) for seed in range(91201,91205))
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(8) as pool:
        for checkpoint in (source,onnx_path):
            for filtered in (True,):
                tasks = [(checkpoint,*case,filtered) for case in fixed+held]
                rows = list(pool.map(evaluate,tasks,chunksize=2))
                label = "onnx-flat" if checkpoint.suffix==".onnx" else "python-flat"
                (output/f"{label}.json").write_text(json.dumps(rows,indent=2,default=lambda x:x.item()))
                for split in ("fixed","independent"):
                    batch = [r for r in rows if r["split"]==split]
                    print(label,split,len(batch),"tracking",sum(r["passed"] for r in batch),
                          "qualified",sum(r["qualified"] for r in batch),"crossing",
                          Counter(r["command"] for r in batch if r["front_cross_ticks"]),
                          "falls",sum(r["fallen"] for r in batch),flush=True)
