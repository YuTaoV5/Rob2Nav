"""Build a first-flip-only reference dataset from physically checked native rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def build(root,output,seeds,absorption_seconds=.04):
    if not 0<=absorption_seconds<=.06: raise ValueError('Only a short first-contact window is allowed')
    obs=[];acts=[];times=[];groups=[];sources=[];policy_hash=None
    for seed in seeds:
        p=root/f'seed_{seed}'
        meta=json.loads((p/'teacher_capture.json').read_text())
        summary=json.loads((p/'summary.json').read_text()); e=summary['episodes'][0]
        if summary['total']!=1 or not e['success'] or not e['fixstand']['passed'] or e['unexpected_body_ground_contact'] or e['max_joint_speed_ratio']>1 or e['max_joint_limit_excess_rad']>1e-6:
            raise ValueError('Teacher physical rollout did not pass')
        if digest(p/'teacher_samples.npz')!=meta['samples_sha256']: raise ValueError('Teacher sample hash mismatch')
        if policy_hash is not None and policy_hash!=meta['policy_sha256']:raise ValueError('Mixed teachers')
        policy_hash=meta['policy_sha256']
        d=np.load(p/'teacher_samples.npz',allow_pickle=False)
        o,a,t=d['observations'],d['raw_actions'],d['policy_time']
        if o.shape!=(len(t),60) or a.shape!=(len(t),12) or not all(np.isfinite(x).all() for x in (o,a,t)) or not np.allclose(t,np.arange(len(t))*.02,rtol=0,atol=1e-9): raise ValueError('Teacher array contract mismatch')
        trace=json.loads((p/'trace.json').read_text()); air_start=None; contact=None
        for row in trace:
            if row['time']<1:continue
            loaded=max(row['foot_forces_N'].values())>5
            if not loaded:
                if air_start is None:air_start=row['time']
            elif air_start is not None:
                if row['time']-air_start>=.08:contact=row['time'];break
                air_start=None
        if contact is None or not 1<contact<2.5:raise ValueError('First landing not located')
        cutoff=contact+absorption_seconds
        mask=t<=cutoff+1e-9
        obs.append(o[mask]);acts.append(a[mask]);times.append(t[mask]);groups.append(np.full(mask.sum(),seed,dtype=np.int64))
        sources.append(dict(seed=seed,summary_sha256=digest(p/'summary.json'),trace_sha256=digest(p/'trace.json'),samples_sha256=meta['samples_sha256'],first_contact_trace_s=contact,cutoff_s=cutoff,retained=int(mask.sum())))
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'samples.npz',observations=np.concatenate(obs),raw_actions=np.concatenate(acts),policy_time=np.concatenate(times),seed=np.concatenate(groups))
    report=dict(version='first_flip_retention_data_v1',teacher_policy_sha256=policy_hash,samples_sha256=digest(output/'samples.npz'),samples=sum(len(x) for x in obs),sources=sources,absorption_seconds=absorption_seconds,contact_time_resolution_s=.02,action_space='raw actor before entry mapping and clipping',task_qualified=False,note='Reference data only. No single-flip terminal standing samples; no evidence of continuous triple-flip success.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.root,a.output,range(29500,29504)),indent=2))
