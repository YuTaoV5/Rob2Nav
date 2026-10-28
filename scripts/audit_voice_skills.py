#!/usr/bin/env python3
"""Inspect manifests/weights without loading torch, ROS, SDK or actuators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_agent.intents import SKILLS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/voice/skills-audit.json'));a=p.parse_args()
    result=[]
    for entry in list(SKILLS.values())+[dict(name='站立',profile='stand'),dict(name='MoE 行走',profile='walk_moe')]:
        path=a.root/'deployment/skills'/(entry['profile']+'.json');row=dict(entry)
        if not path.exists():row.update(available=False,reason='manifest missing')
        else:
            d=json.loads(path.read_text(encoding='utf-8'));row.update(manifest=str(path),hardware_qualified=d.get('hardware_qualified',False),
                simulation_enabled=d.get('simulation_enabled',False),blocker=d.get('blocker'),entry=d.get('entry'),
                atomic=d.get('atomic',False),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            policy=d.get('policy','');weight=a.root/policy
            valid_path=weight.resolve().is_relative_to(a.root.resolve())
            row['available']=not policy or valid_path and weight.is_file() and hashlib.sha256(weight.read_bytes()).hexdigest()==d.get('sha256')
            row['expected_policy_sha256']=d.get('sha256')
        result.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
