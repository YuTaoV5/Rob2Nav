"""Export a locally trained, unnormalized RSL-RL ELU actor without launching Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import yaml


class ConfigLoader(yaml.SafeLoader):
    """Read Isaac config tags as plain data; never construct Python objects."""


def plain_python_tag(loader, suffix, node):
    if isinstance(node,yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node,yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/',plain_python_tag)


def export(checkpoint, output):
    checkpoint = Path(checkpoint).resolve()
    cfg = yaml.safe_load((checkpoint.parent/'agent.yaml').read_text())
    if cfg['policy']['activation'] != 'elu':
        raise ValueError('This exporter supports only the reviewed ELU actor contract')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)  # Trusted, locally produced checkpoints only.
    state = saved['model_state_dict']
    if any('actor_obs_normalizer' in k for k in state):
        raise ValueError('Actor contains normalization state; explicit normalizer export is required')
    weights = {k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')}
    indexes = sorted(int(k.split('.')[0]) for k in weights if k.endswith('.weight'))
    if not indexes or indexes != list(range(0,2*len(indexes),2)):
        raise ValueError('Unexpected actor layer layout; do not guess an incompatible graph')
    layers = []
    for i,index in enumerate(indexes):
        w = weights[f'{index}.weight']
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i != len(indexes)-1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).eval()
    actor.load_state_dict(weights, strict=True)
    obs_dim = int(weights['0.weight'].shape[1])
    with torch.inference_mode():
        sample = actor(torch.zeros(1,obs_dim))
    if sample.shape != (1,12) or not torch.isfinite(sample).all():
        raise ValueError('Expected a finite twelve-action Go2 actor')
    env = yaml.load((checkpoint.parent/'env.yaml').read_text(),Loader=ConfigLoader)
    contract = json.loads((checkpoint.parent/'contract.json').read_text())
    if (checkpoint.parent/'fixstand.json').is_file():
        contract['fixstand_roundtrip']=json.loads((checkpoint.parent/'fixstand.json').read_text())
    default = env['scene']['robot']['init_state']['joint_pos']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.jit.script(actor).save(str(output/'policy.pt'))
    contract.update(actor_input_dim=obs_dim, actor_output_dim=12, normalization='identity',
        actor_observation_terms=[dict(name=name, scale=term.get('scale'), clip=term.get('clip'))
            for name,term in env['observations']['policy'].items() if isinstance(term,dict) and 'func' in term],
        default_joint_pos=[float(default[name]) for name in contract['joint_names']],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((output/'policy.pt').read_bytes()).hexdigest(),
        export_verified_finite=True, capability_verified=False, hardware_connected=False)
    (output/'policy.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(json.dumps(dict(policy=str(output/'policy.pt'), observations=obs_dim, actions=12,
        policy_sha256=contract['policy_sha256'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export(args.checkpoint,args.output)
