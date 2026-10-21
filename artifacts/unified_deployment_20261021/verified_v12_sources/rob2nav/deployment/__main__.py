"""Offline inspection by default; hardware requires explicit device qualification."""
import argparse
import json
from pathlib import Path
import numpy as np
from .types import Skill


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('profiles')
    plan=sub.add_parser('plan');plan.add_argument('--from',dest='old',required=True);plan.add_argument('--to',required=True)
    run=sub.add_parser('run');run.add_argument('--profile',required=True);run.add_argument('--interface',required=True)
    run.add_argument('--sdk-root',type=Path,required=True);run.add_argument('--calibration',type=Path,required=True)
    run.add_argument('--qualification',type=Path,required=True);run.add_argument('--model',type=Path,required=True)
    run.add_argument('--output',type=Path,required=True);run.add_argument('--enable-motors',action='store_true')
    run.add_argument('--sequence',type=Path,help='Explicit choreography; every contract still needs device qualification')
    a=p.parse_args();profiles={x.stem:Skill.load(x,a.root) for x in (a.root/'deployment/skills').glob('*.json')}
    if a.command=='profiles':
        print(json.dumps([dict(id=x.id,observation=x.observation,sha256=x.sha256,simulation_enabled=x.simulation_enabled,
            hardware_qualified=x.hardware_qualified,blocker=x.blocker) for x in profiles.values()],indent=2));return
    if a.command=='run':
        if a.profile not in profiles:p.error('Unknown profile')
        if not a.enable_motors:
            print(json.dumps(dict(hardware_connected=False,motor_write_enabled=False,note='No DDS initialized. Explicit motor enable and complete device qualification are required.')));return
        if not profiles[a.profile].hardware_qualified:p.error('Profile is not device qualified: '+profiles[a.profile].blocker)
        from .service import run
        run(a.root,a.profile,a.interface,a.sdk_root,a.calibration,a.qualification,a.model,a.output,sequence_path=a.sequence);return
    if a.old not in profiles or a.to not in profiles:p.error('Unknown profile ID')
    old,new=profiles[a.old],profiles[a.to]
    print(json.dumps(dict(from_skill=old.id,to_skill=new.id,max_pose_difference_rad=float(np.max(abs(old.to_sdk(old.default)-new.to_sdk(new.default)))),
        max_kp_difference=float(np.max(abs(old.to_sdk(old.kp)-new.to_sdk(new.kp)))),
        route=['quiesce gait or wait for complete atomic action','measured stable stance; gait-only validated low-speed support-polygon fallback','neutral bridge','entry pose/gain slew','fresh-history warmup','blend','commit generation'],
        simulation_enabled=new.simulation_enabled,hardware_qualified=new.hardware_qualified,blocker=new.blocker,hardware_connected=False),indent=2))


if __name__=='__main__':main()
