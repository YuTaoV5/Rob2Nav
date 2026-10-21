"""Offline deployment inspection only; no implicit hardware entry point."""
import argparse
import json
from pathlib import Path
import numpy as np
from .types import Skill


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('profiles')
    plan=sub.add_parser('plan');plan.add_argument('--from',dest='old',required=True);plan.add_argument('--to',required=True)
    a=p.parse_args();profiles={x.stem:Skill.load(x,a.root) for x in (a.root/'deployment/skills').glob('*.json')}
    if a.command=='profiles':
        print(json.dumps([dict(id=x.id,observation=x.observation,sha256=x.sha256,simulation_enabled=x.simulation_enabled,
            hardware_qualified=x.hardware_qualified,blocker=x.blocker) for x in profiles.values()],indent=2));return
    old,new=profiles[a.old],profiles[a.to]
    print(json.dumps(dict(from_skill=old.id,to_skill=new.id,max_pose_difference_rad=float(np.max(abs(old.to_sdk(old.default)-new.to_sdk(new.default)))),
        max_kp_difference=float(np.max(abs(old.to_sdk(old.kp)-new.to_sdk(new.kp)))),
        route=['quiesce current policy','wait for stable measured four-foot stance','neutral bridge','entry pose/gain slew','fresh-history warmup','blend','commit generation'],
        simulation_enabled=new.simulation_enabled,hardware_qualified=new.hardware_qualified,blocker=new.blocker,hardware_connected=False),indent=2))


if __name__=='__main__':main()
