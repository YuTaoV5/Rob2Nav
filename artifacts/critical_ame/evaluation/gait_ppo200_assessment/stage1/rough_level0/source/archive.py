"""Archive completed checkpoints, evaluations, failures and training metadata."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
p.add_argument('--evaluations',nargs='+',default=[]);p.add_argument('--logs',nargs='+',default=[])
p.add_argument('--upstream-manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'runs':{},'evaluations':[],
          'note':'Archiving records progress only. No training success is inferred from completed commands.'}
shutil.copy2(a.upstream_manifest,out/'upstream_manifest.json')
for folder in a.runs:
    source=Path(folder);target=out/'training'/source.name;target.mkdir(parents=True,exist_ok=True)
    for f in source.iterdir():
        if f.is_file() and f.suffix in ('.json','.jsonl','.yaml') or f.is_file() and f.name.startswith('events.out.tfevents.'):
            shutil.copy2(f,target/f.name)
        elif f.name=='source' and f.is_dir():shutil.copytree(f,target/'source',dirs_exist_ok=True)
    weights=sorted(source.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
    record={'completed':(source/'train_summary.json').exists()}
    if weights:
        weight=weights[-1];shutil.copy2(weight,target/weight.name)
        record['checkpoint']={'path':str((target/weight.name).relative_to(out)),
            'sha256':hashlib.sha256(weight.read_bytes()).hexdigest()}
    manifest['runs'][source.name]=record
for folder in a.evaluations:
    source=Path(folder);target=out/'evaluation'/source.name
    shutil.copytree(source,target,dirs_exist_ok=True)
    manifest['evaluations'].append(str(target.relative_to(out)))
logdir=out/'logs';logdir.mkdir(exist_ok=True)
for source in map(Path,a.logs):shutil.copy2(source,logdir/source.name)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
