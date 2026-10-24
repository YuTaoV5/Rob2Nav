"""Read-only phase/rotation diagnostic; never a skill qualification report."""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path


def audit(rows, trace, dt=.001):
    if not rows or not trace:
        raise ValueError("Both raw physics and pose traces are required")
    times = [r["time"] for r in trace]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Pose trace must be chronological")
    angle = 0.
    events = []
    segments = []
    previous_phase = None
    start = None
    first_landing = None
    probes = {}
    for index, row in enumerate(rows):
        if abs(row["time"] - (index+1)*dt) > 1e-6:
            raise ValueError("Missing or irregular physical samples")
        omega = row["omega_body"][1]
        if not math.isfinite(omega):
            raise ValueError("Nonfinite pitch velocity")
        angle += omega*dt
        key = bool(row["foot_contact"])
        if start is None or key != start["foot_contact"]:
            if start is not None:
                start["end_s"] = row["time"]-dt
                start["pitch_rad"] = angle-omega*dt-start.pop("angle_before")
                segments.append(start)
                if (not start["foot_contact"] and first_landing is None
                        and row["time"]-start["start_s"] >= .02-1e-7):
                    first_landing = row["time"]
            start = dict(start_s=row["time"],foot_contact=key,angle_before=angle-omega*dt)
        if row["phase"] != previous_phase:
            events.append(dict(time_s=row["time"],phase=row["phase"],pitch_integral_rad=angle))
            previous_phase = row["phase"]
        if first_landing is not None:
            for name, threshold in (("total_5p5_rad",5.5),("total_two_pi_rad",2*math.pi)):
                if name not in probes and -angle >= threshold:
                    j=bisect.bisect_left(times,row["time"])
                    j=min(j,len(trace)-1)
                    pose=trace[j]
                    probes[name]=dict(time_s=row["time"],seconds_after_first_contact=row["time"]-first_landing,
                        pose_sample_offset_s=pose["time"]-row["time"],upright=pose["upright"],
                        pitch_integral_rad=angle,phase=row["phase"])
    start["end_s"]=rows[-1]["time"]
    start["pitch_rad"]=angle-start.pop("angle_before")
    segments.append(start)
    return dict(scope="read-only rotation allocation; body-axis integral is not orientation certification",
        physics_samples=len(rows),dt=dt,segments=segments,phase_events=events,landing_probes=probes,
        total_pitch_rad=angle,acceptance_unchanged=True,capability_verified=False,hardware_qualified=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    raw=a.run/"rebound_physics_trace.json"
    pose=a.run/"trace.json"
    result=audit(json.loads(raw.read_text()),json.loads(pose.read_text()))
    result["inputs_sha256"]={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (raw,pose)}
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))
