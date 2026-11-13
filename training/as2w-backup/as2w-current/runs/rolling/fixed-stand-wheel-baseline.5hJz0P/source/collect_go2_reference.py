"""Collect proprioceptive teacher gait on the original Go2 MuJoCo robot."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from as2w_go2_retarget import GO2_NEUTRAL
from as2w_policy_provenance import checkpoint_record


def collect(xml, actor, scenario, command):
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir",str((xml.parent/"assets").resolve()))
    root.find("option").set("timestep",".002")
    world = root.find("worldbody")
    ET.SubElement(world,"geom",name="ground",type="plane",size="0 0 .1",friction=".8 .02 .001")
    if scenario in ("up","down"):
        for index in range(4):
            height = .05*(index+1)
            width = .25 if index<3 else 2.
            x = .8+.25*index+width/2.
            ET.SubElement(world,"geom",name=f"tread_{index}",type="box",
                          pos=f"{x} 0 {height/2}",size=f"{width/2} .7 {height/2}",
                          friction=".8 .02 .001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data = mujoco.MjData(model)
    neutral = GO2_NEUTRAL.ravel()
    data.qpos[2] = .36
    if scenario=="down":
        data.qpos[:3] = (2.15,0.,.56)
        data.qpos[3:7] = (0.,0.,0.,1.)
    data.qpos[7:] = neutral
    mujoco.mj_forward(model,data)
    actor.reset()
    base = model.body("base_link").id
    foot_bodies = [model.body(f"{leg}_foot").id for leg in ("FL","FR","RL","RR")]
    foot_geoms = [model.geom(leg).id for leg in ("FL","FR","RL","RR")]
    terrain_geoms = {model.geom("ground").id}
    if scenario in ("up","down"):
        terrain_geoms.update(model.geom(f"tread_{i}").id for i in range(4))
    q_prev = np.zeros(12,dtype=np.float32)
    twist = np.empty(6)
    samples = {k:[] for k in ("observation","action","position_target","foot_body_m","contact","base_xyz_m","up_z")}
    for tick in range(550):
        cmd = command if 50<=tick<450 else np.zeros(3)
        rotation = data.xmat[base].reshape(3,3)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,twist,1)
        obs = np.r_[twist[:3]*.25,-rotation[2],cmd*[2.,2.,.25],
                    data.qpos[7:]-neutral,data.qvel[6:]*.05,q_prev].astype(np.float32)
        result = actor(torch.from_numpy(obs[None]))
        q_prev = (result[0] if isinstance(result,tuple) else result)[0].numpy()
        target = neutral+.25*q_prev
        contacts = np.zeros(4,dtype=bool)
        for contact in data.contact:
            a,b = contact.geom
            if a in terrain_geoms and b in foot_geoms:
                contacts[foot_geoms.index(b)] = True
            elif b in terrain_geoms and a in foot_geoms:
                contacts[foot_geoms.index(a)] = True
        values = (obs,q_prev.copy(),target,(data.xpos[foot_bodies]-data.xpos[base])@rotation,
                  contacts,data.qpos[:3].copy(),float(rotation[2,2]))
        for key,value in zip(samples,values):
            samples[key].append(value)
        if rotation[2,2]<.75 or not np.isfinite(obs).all():
            break
        for _ in range(10):
            torque = 20.*(target-data.qpos[7:])-.5*data.qvel[6:]
            data.ctrl[:] = np.clip(torque,-23.7,23.7)
            mujoco.mj_step(model,data)
    return {k:np.asarray(v) for k,v in samples.items()}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    actor=torch.jit.load(str(args.checkpoint),map_location="cpu").eval()
    cases={"forward":(.25,0.,0.),"back":(-.20,0.,0.),"left":(0.,.20,0.),
           "right":(0.,-.20,0.),"turn_left":(0.,0.,.30),"turn_right":(0.,0.,-.30),
           "up":(.25,0.,0.),"down":(.25,0.,0.)}
    report={}
    for scenario,cmd in cases.items():
        record=collect(args.xml,actor,scenario,np.array(cmd))
        np.savez_compressed(args.out/f"{scenario}.npz",**record)
        xyz=record["base_xyz_m"]
        report[scenario]={"frames":len(xyz),"upright":bool(len(xyz)==550 and record["up_z"][-1]>=.75),
                          "delta_xyz_m":(xyz[-1]-xyz[0]).tolist(),
                          "contact_switches":np.count_nonzero(np.diff(record["contact"],axis=0),axis=0).tolist()}
        print(scenario,report[scenario],flush=True)
    (args.out/"summary.json").write_text(json.dumps(
        {"source":checkpoint_record(args.checkpoint),"control_hz":50,"cases":report},indent=2)+"\n")


if __name__=="__main__":
    main()
