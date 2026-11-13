"""Prepare motion-only Go2 traces and flag AS2-W targets outside the stance envelope."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from as2w_go2_retarget import GO2_NEUTRAL, Go2WheelRetarget, endpoint, project_wheel_targets

AXES = [f"{leg}_{axis}" for leg in ("FL","FR","RL","RR") for axis in ("hip","thigh","calf")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("traces",type=Path,nargs="+")
    parser.add_argument("--project",action="store_true")
    args = parser.parse_args()
    joints = {j.attrib["name"]:j.find("limit") for j in ET.parse(args.urdf).getroot().findall("joint")}
    bounds = np.array([[float(joints[name].attrib[k]) for k in ("lower","upper")] for name in AXES])
    neutral = np.tile([0.,.8,-1.6],(4,1))
    bounds = neutral.ravel()[:,None]+.9*(bounds-neutral.ravel()[:,None])
    mapper = Go2WheelRetarget(neutral)
    args.output.mkdir(exist_ok=True,parents=True)
    for trace in args.traces:
        with np.load(trace) as data:
            motion = np.linalg.norm(data["observation"][:,6:9],axis=1)>.05
            xyz = data["base_position_m"][motion]
            contacts = data["contact"][motion]
            if (len(xyz)<100 or np.linalg.norm(xyz[-1,:2]-xyz[0,:2])<.8
                    or not np.all(np.count_nonzero(np.diff(contacts,axis=0),axis=0)>0)):
                raise ValueError(f"{trace} does not contain a complete moving gait with foot contacts")
            go2_targets = data["joint_target"][motion].reshape(-1,4,3)
            q = np.stack([mapper.target((target-GO2_NEUTRAL)/.25) for target in go2_targets])
            flat_q = q.reshape(-1,12)
            bounded = ((flat_q >= bounds[:,0]) & (flat_q <= bounds[:,1])).all(axis=1)
            rear_ok = ((q[:,2:,1]>=.18) & (q[:,2:,1]<=1.40)).all(axis=1)
            desired_wheel = (mapper.as2w_neutral_xyz+endpoint(go2_targets,.0955,.213)
                             -mapper.go2_neutral_xyz)
            if args.project:
                constrained = bounds.reshape(4,3,2).copy()
                constrained[2:,1,0] = np.maximum(constrained[2:,1,0],.23)
                constrained[2:,1,1] = np.minimum(constrained[2:,1,1],1.35)
                q = project_wheel_targets(q,desired_wheel,constrained)
            wheel_target = endpoint(q,.1054,.212)
            wheel_error = np.linalg.norm(wheel_target-desired_wheel,axis=-1)
            output_q = q.reshape(-1,12)
            output_ok = ((output_q>=bounds[:,0]) & (output_q<=bounds[:,1])).all(axis=1)
            output_ok &= ((q[:,2:,1]>=.18) & (q[:,2:,1]<=1.40)).all(axis=1)
            np.savez_compressed(args.output/f"{trace.stem}.npz",as2w_joint_target_rad=q,
                as2w_wheel_hip_m=wheel_target, go2_observation=data["observation"][motion],
                go2_contact=contacts,base_position_m=xyz,time_s=data["time_s"][motion],
                as2w_joint_and_rear_stance_ok=output_ok, wheel_error_m=wheel_error, joint_names=AXES)
            summary = dict(source=str(trace),frames=len(xyz),elevation_change_m=float(xyz[-1,2]-xyz[0,2]),
                           within_joint_bounds=int(bounded.sum()), within_rear_stance=int(rear_ok.sum()),
                           both=int((bounded & rear_ok).sum()), projected=args.project, output_in_bounds=int(output_ok.sum()),
                           wheel_error_p95_m=float(np.quantile(wheel_error,.95)))
            (args.output/f"{trace.stem}.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(summary,flush=True)


if __name__=="__main__":
    main()
