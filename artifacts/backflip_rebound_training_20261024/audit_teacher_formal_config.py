"""Read actual formal training YAML and retention record; no action qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from rob2nav.export_actor import ConfigLoader
from skills.backflip.landing_contract import validate_landing_contract


def differences(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,baseline=a,current=b)]
    if isinstance(a,dict):
        return [d for k in sorted(set(a)|set(b)) for d in differences(a.get(k),b.get(k),path+'.'+k)]
    return [] if a==b else [dict(path=path,baseline=a,current=b)]


def audit(run,baseline,output):
    run=Path(run);baseline=Path(baseline)
    env=yaml.load((run/'params/env.yaml').read_text(),Loader=ConfigLoader)
    base=yaml.load((baseline/'params/env.yaml').read_text(),Loader=ConfigLoader)
    env_diff=differences(base,env)
    if env_diff:raise ValueError('Unexpected environment differences: '+str(env_diff))
    agent=yaml.safe_load((run/'params/agent.yaml').read_text())
    base_agent=yaml.safe_load((baseline/'params/agent.yaml').read_text())
    agent_diff=differences(base_agent,agent)
    if any(d['path']!='.run_name' for d in agent_diff):raise ValueError('Unexpected PPO differences: '+str(agent_diff))
    contract=json.loads((run/'fixstand.json').read_text())
    validate_landing_contract(env,contract)
    payload=(run/'teacher_retention.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=contract.get('teacher_retention_sha256'):
        raise ValueError('Teacher record hash mismatch')
    teacher=json.loads(payload)
    expected=dict(weight=5.,learning_rate=.0001,batch_size=128,hold_scale=.15,action_scale=.5,action_clip=8.,samples=408,sampler_seed=76)
    for k,v in expected.items():
        if teacher.get(k)!=v:raise ValueError('Unexpected retention setting '+k)
    manifest=(run/'teacher_manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest()!=teacher['manifest_sha256']:
        raise ValueError('Teacher manifest hash mismatch')
    report=dict(actual_config_verified=True,capability_verified=False,env_diff=env_diff,agent_diff=agent_diff,
        teacher_retention=teacher,files_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (run/'params/env.yaml',run/'params/agent.yaml',run/'teacher_retention.json',run/'fixstand.json')})
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.baseline,a.output),indent=2))
