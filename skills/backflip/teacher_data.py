"""Pinned teacher dataset validation usable by Studio without PyTorch."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def load_reference(directory, manifest_sha256):
    directory=Path(directory)
    payload=(directory/'manifest.json').read_bytes()
    if len(manifest_sha256)!=64 or hashlib.sha256(payload).hexdigest()!=manifest_sha256:
        raise ValueError('Teacher manifest hash mismatch')
    meta=json.loads(payload)
    if meta.get('version')!='first_flip_retention_data_v1' or meta.get('action_space')!='raw actor before entry mapping and clipping':
        raise ValueError('Unknown teacher observation/action semantics')
    data_path=directory/'samples.npz'
    if hashlib.sha256(data_path.read_bytes()).hexdigest()!=meta.get('samples_sha256'):
        raise ValueError('Teacher data hash mismatch')
    with np.load(data_path,allow_pickle=False) as saved:
        data={k:saved[k].copy() for k in ('observations','raw_actions','policy_time','seed')}
    n=meta.get('samples')
    if type(n) is not int or n<1 or data['observations'].shape!=(n,60) or data['raw_actions'].shape!=(n,12) or data['policy_time'].shape!=(n,) or data['seed'].shape!=(n,):
        raise ValueError('Teacher shape mismatch')
    if not all(np.isfinite(v).all() for v in data.values()):raise ValueError('Nonfinite teacher data')
    if (data['policy_time']<0).any() or (data['policy_time']>2.1).any():raise ValueError('Teacher includes unsupported/terminal time')
    absorption=meta.get('absorption_seconds')
    if type(absorption) not in (int,float) or not math.isfinite(absorption) or not 0<=absorption<=.06:
        raise ValueError('Invalid teacher absorption window')
    if not meta.get('sources'):raise ValueError('Missing teacher sources')
    if len(set(s['seed'] for s in meta['sources']))!=len(meta['sources']):raise ValueError('Duplicate teacher seeds')
    if set(data['seed'].tolist())!={s['seed'] for s in meta['sources']}:raise ValueError('Teacher source mismatch')
    for s in meta['sources']:
        contact=s.get('first_contact_trace_s');cutoff=s.get('cutoff_s')
        if type(contact) not in (int,float) or type(cutoff) not in (int,float) or not math.isfinite(contact) or not math.isfinite(cutoff) or contact<1 or abs(cutoff-contact-absorption)>1e-9:
            raise ValueError('Teacher contact cutoff mismatch')
        t=data['policy_time'][data['seed']==s['seed']]
        if len(t)!=s['retained'] or not np.allclose(t,np.arange(len(t))*.02,rtol=0,atol=1e-9) or (t>s['cutoff_s']+1e-9).any():
            raise ValueError('Teacher time coverage mismatch')
    return data,meta

