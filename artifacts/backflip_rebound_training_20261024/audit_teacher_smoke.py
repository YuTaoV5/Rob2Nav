"""Audit an executed two-update teacher smoke; never certifies triple flips."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from rob2nav.export_actor import ConfigLoader
from skills.backflip.landing_contract import validate_landing_contract
from skills.backflip.export_cpu import export


def audit(run, output):
    run=Path(run);output=Path(output)
    completion=json.loads((run/'completion.json').read_text())
    if not completion['execution_complete'] or completion['requested_iterations']!=2:
        raise ValueError('Expected completed two-update smoke')
    record=json.loads((run/'teacher_retention.json').read_text())
    expected=dict(weight=5.,learning_rate=.0001,batch_size=128,hold_scale=.15,action_scale=.5,action_clip=8.,samples=408,sampler_seed=76)
    for key,value in expected.items():
        if record.get(key)!=value:raise ValueError('Unexpected teacher setting '+key)
    env=yaml.load((run/'params/env.yaml').read_text(),Loader=ConfigLoader)
    contract=json.loads((run/'fixstand.json').read_text())
    validate_landing_contract(env,contract)
    if env['rewards']['scales']['landing_impact']!=-10.:raise ValueError('Wrong impact baseline')
    accumulator=EventAccumulator(str(run),size_guidance={'scalars':0});accumulator.Reload()
    scalars={}
    for name in ('teacher_retention_mse','teacher_retention_grad_norm'):
        tags=[tag for tag in accumulator.Tags()['scalars'] if tag.endswith('/'+name) or tag==name]
        if len(tags)!=1:raise ValueError('Missing or ambiguous auxiliary scalar '+name)
        events=accumulator.Scalars(tags[0]);values=[dict(step=e.step,value=e.value) for e in events]
        if len(values)!=2 or len({e['step'] for e in values})!=2 or not all(math.isfinite(e['value']) and e['value']>=0 for e in values) or not any(e['value']>0 for e in values):
            raise ValueError('Auxiliary updates not evidenced by two finite nonzero scalar samples')
        scalars[name]=values
    checkpoint=run/completion['last_checkpoint']
    exported=output/'exported'
    if exported.exists():raise FileExistsError('Do not overwrite an existing smoke export')
    output.mkdir(parents=True,exist_ok=True)
    export(checkpoint,exported,landing=True)
    metadata=json.loads((exported/'export.json').read_text())
    if metadata['teacher_retention']!=record:raise ValueError('Export lost teacher training provenance')
    result=dict(configuration_and_update_chain_verified=True,capability_verified=False,
        checkpoint=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        completion=completion,teacher_retention=record,scalars=scalars,onnx_max_error=metadata['onnx_max_error'])
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(audit(args.run,args.output),indent=2))
