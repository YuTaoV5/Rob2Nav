"""Check actor freezing, value learning, resumed actor learning and labels."""
import argparse,hashlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for key in ('initializer','warmup','ppo','output'):p.add_argument('--'+key,required=True)
a=p.parse_args();paths=[Path(a.initializer),Path(a.warmup),Path(a.ppo)]
models=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
states=[d['model_state_dict'] for d in models]
actor=[k for k in states[0] if not k.startswith(('critic.','critic_proprio_embedding.'))]
critic=[k for k in states[0] if k.startswith(('critic.','critic_proprio_embedding.'))]
assert all(torch.equal(states[0][k],states[1][k]) for k in actor)
assert any(not torch.equal(states[0][k],states[1][k]) for k in critic)
changed=[k for k in actor if not torch.equal(states[1][k],states[2][k])]
assert any(k.startswith('actor.') for k in changed)
for key in ('protocol','teacher_sha256'):
    assert all(d['infos'][key]==models[0]['infos'][key] for d in models)
assert models[0]['infos']['ppo_resume_allowed'] is False
assert all(d['infos']['ppo_resume_allowed'] is True for d in models[1:])
summary=json.loads((paths[1].parent/'train_summary.json').read_text())
assert summary['critic_only'] and summary['frozen_actor_unchanged']
loaded=json.loads((paths[2].parent/'loaded_checkpoint.json').read_text())
assert loaded['load_optimizer'] and loaded['next_iteration']==models[1]['iter']
assert not json.loads((paths[2].parent/'train_summary.json').read_text())['critic_only']
report={'actor_unchanged_during_value_warmup':True,'critic_changed_during_warmup':True,
        'actor_changed_during_resumed_ppo':changed,'external_teacher_labels_preserved':True,
        'optimizer_resumed':True,'checkpoints':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'next_iteration':d['iter']} for path,d in zip(paths,models)]}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
