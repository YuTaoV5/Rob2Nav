"""Export the reviewed 60-input ELU backflip actor, with ONNX numerical parity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from rob2nav.export_actor import ConfigLoader
from .control import control_contract,validate_entry_contract


def export(checkpoint,output):
    checkpoint=Path(checkpoint);output=Path(output)
    cfg=yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())
    if cfg['policy']['activation']!='elu':raise ValueError('Expected reviewed ELU architecture')
    contract=json.loads((checkpoint.parent/'fixstand.json').read_text()) if (checkpoint.parent/'fixstand.json').exists() else None
    env=yaml.load((checkpoint.parent/'params/env.yaml').read_text(),Loader=ConfigLoader)
    validate_entry_contract(env,contract)
    controller=control_contract(env['control'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);state=saved['model_state_dict']
    if any('normalizer' in key for key in state):raise ValueError('Explicit normalization export required')
    weights={k[6:]:v for k,v in state.items() if k.startswith('actor.')}
    indexes=sorted(int(k.split('.')[0]) for k in weights if k.endswith('.weight'))
    if indexes!=list(range(0,2*len(indexes),2)) or weights['0.weight'].shape[1]!=60:raise ValueError('Wrong actor graph')
    layers=[]
    for n,index in enumerate(indexes):
        w=weights[f'{index}.weight'];layers.append(nn.Linear(w.shape[1],w.shape[0]))
        if n<len(indexes)-1:layers.append(nn.ELU())
    actor=nn.Sequential(*layers).eval();actor.load_state_dict(weights,strict=True)
    output.mkdir(parents=True,exist_ok=False)
    sample=torch.zeros(1,60)
    torch.onnx.export(actor,sample,str(output/'policy.onnx'),input_names=['obs'],output_names=['actions'],
                      dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},opset_version=17,dynamo=False)
    torch.jit.script(actor).save(str(output/'policy.pt'))
    import onnxruntime as ort
    probe=np.random.default_rng(61).normal(0,.1,(16,60)).astype(np.float32)
    session=ort.InferenceSession(str(output/'policy.onnx'),providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=actor(torch.from_numpy(probe)).numpy()
    actual=session.run(None,{'obs':probe})[0]
    error=float(np.max(abs(actual-expected)))
    if not np.isfinite(actual).all() or actual.shape!=(16,12) or error>1e-5:raise RuntimeError('ONNX parity failed')
    (output/'export.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((output/'policy.onnx').read_bytes()).hexdigest(),actor_input_dim=60,
        actor_output_dim=12,onnx_max_error=error,fixstand_roundtrip=contract,control_contract=controller,
        reward_scale_overrides=env.get('reward_scale_overrides',{}),actual_reward_scales=env['rewards']['scales'],
        entry_training_contract_verified=True,capability_verified=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.checkpoint,a.output)
