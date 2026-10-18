"""Check actual native PPO parameter changes, not only requires_grad flags."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .model import load_base
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--smoke',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.smoke);summary=json.loads((root/'completed.json').read_text())
paths=[root/'model_0.pt',root/f"model_{summary['last_iteration']}.pt"]
states=[torch.load(path,map_location='cpu',weights_only=False)['model_state_dict'] for path in paths]
base=load_base(a.moe).state_dict()
for name,value in base.items():
    assert torch.equal(value,states[0]['actor.base.'+name])
    assert torch.equal(value,states[1]['actor.anchor.'+name])
actor_changed=[n for n in base if n.startswith('actor.') and not torch.equal(states[0]['actor.base.'+n],states[1]['actor.base.'+n])]
encoder_changed=[n for n in base if n.startswith('student_moe_encoder.') and not torch.equal(states[0]['actor.base.'+n],states[1]['actor.base.'+n])]
assert actor_changed and encoder_changed
assert not torch.equal(states[0]['actor.terrain.cnn.0.weight'],states[1]['actor.terrain.cnn.0.weight'])
record={'value_phase_base_exact':True,'reference_anchor_unchanged':True,'terrain_cnn_updated':True,
    'actor_updated_tensors':actor_changed,'encoder_updated_tensors':encoder_changed,
    'checkpoints':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
Path(a.output).write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
