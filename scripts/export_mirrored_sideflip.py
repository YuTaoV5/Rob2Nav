"""Derive a right-side candidate and reflected reference; acceptance is separate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from skills.sideflip.reflection import joint_map,observation_map,motion


class ReflectedPolicy(torch.nn.Module):
    def __init__(self,actor,joints):
        super().__init__();self.actor=actor
        p,s=joint_map(joints);op,os=observation_map(joints)
        self.register_buffer('joint_index',torch.tensor(p,dtype=torch.long))
        self.register_buffer('joint_sign',torch.tensor(s,dtype=torch.float32))
        self.register_buffer('observation_index',torch.tensor(op,dtype=torch.long))
        self.register_buffer('observation_sign',torch.tensor(os,dtype=torch.float32))
    def forward(self,x):
        y=self.actor(x.index_select(-1,self.observation_index)*self.observation_sign)
        return y.index_select(-1,self.joint_index)*self.joint_sign


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    meta=json.loads(a.policy.with_suffix('.json').read_text())
    if sha(a.policy)!=meta['policy_sha256'] or sha(a.motion)!=meta['reference_sha256']:raise ValueError('Source resource hash mismatch')
    if meta['actor_input_dim']!=69 or meta['normalization']!='identity':raise ValueError('Requires the reviewed 69-input identity-normalized actor')
    actor=torch.jit.load(str(a.policy),map_location='cpu').eval()
    candidate=ReflectedPolicy(actor,meta['joint_names']).eval();a.output.mkdir(parents=True,exist_ok=False)
    torch.jit.script(candidate).save(str(a.output/'policy.pt'))
    with np.load(a.motion,allow_pickle=False) as data:mirrored=motion(dict(data),meta['joint_names'],meta['body_names'])
    np.savez_compressed(a.output/'rightflip.npz',**mirrored)
    ref=json.loads(a.motion.with_suffix('.json').read_text());ref.update(npz_sha256=sha(a.output/'rightflip.npz'),sideflip_direction='right',source_reference_sha256=sha(a.motion))
    (a.output/'rightflip.json').write_text(json.dumps(ref,indent=2)+'\n')
    meta.update(policy_sha256=sha(a.output/'policy.pt'),reference_sha256=sha(a.output/'rightflip.npz'),
                sideflip_direction='right',source_policy_sha256=sha(a.policy),
                derivation='Sagittal reflection of local left candidate, not a newly trained or accepted right policy',
                reflection_source_sha256=sha(Path(__file__).resolve().parents[1]/'skills/sideflip/reflection.py'),capability_verified=False,hardware_connected=False)
    meta['fixstand_roundtrip']['reference_sha256']=meta['reference_sha256']
    (a.output/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(policy_sha256=meta['policy_sha256'],capability_verified=False)))


if __name__=='__main__':main()
