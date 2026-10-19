"""Bounded progress snapshot; selected weights, exact source and evaluation data."""
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--pipeline',required=True)
p.add_argument('--smoke',required=True);p.add_argument('--probe',required=True);p.add_argument('--checks',nargs='+',required=True)
p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def metadata(source,target):
    target.mkdir(parents=True,exist_ok=True)
    for f in source.iterdir():
        if f.is_file() and (f.suffix in ('.json','.jsonl','.yaml','.log') or f.name.startswith('events.out.tfevents')):
            shutil.copy2(f,target/f.name)
        elif f.is_dir() and f.name=='source':shutil.copytree(f,target/f.name,dirs_exist_ok=True)
run=Path(a.run);target=out/'training'/run.name;metadata(run,target)
weights=sorted(run.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]))
if not weights:raise FileNotFoundError('No atomic training checkpoint exists')
latest=weights[-1];shutil.copy2(latest,target/latest.name)
metadata(Path(a.smoke),out/'smoke')
shutil.copytree(a.probe,out/'baseline_probe',dirs_exist_ok=True,ignore=shutil.ignore_patterns('actor.pt'))
pipeline=Path(a.pipeline);metadata(pipeline,out/'pipeline')
for name in ('baseline','trained'):
    folder=pipeline/name;commands=folder/'commands.json'
    if not commands.exists():continue
    target_cases=out/name;target_cases.mkdir(exist_ok=True)
    records=json.loads(commands.read_text());shutil.copy2(commands,target_cases/'commands.json')
    # A successful process return ensures that the MP4 writer has closed. Do
    # not archive a currently running case's partially written media/source.
    for record in records:
        if record['returncode']!=0:continue
        command=record['command'];trial=Path(command[command.index('--output')+1])
        if not (trial/'summary.json').exists():continue
        shutil.copytree(trial,target_cases/trial.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('actor.pt'))
        log=folder/(trial.name+'.log')
        if log.exists():shutil.copy2(log,target_cases/log.name)
export=pipeline/'trained'/'flat_0.00'/'actor.pt'
if export.exists():
    (out/'export').mkdir(exist_ok=True);shutil.copy2(export,out/'export'/'actor.pt')
checks=out/'checks';checks.mkdir(exist_ok=True)
for path in map(Path,a.checks):shutil.copy2(path,checks/path.name)
manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'run_completed':(run/'completed.json').exists(),
    'selected_checkpoint':str((target/latest.name).relative_to(out)),
    'note':'Progress only. Smoke weights are omitted; selected main checkpoint is not a success claim.',
    'files':{str(f.relative_to(out)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size}
        for f in out.rglob('*') if f.is_file() and f!=out/'manifest.json'}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('ARCHIVED',len(manifest['files']),manifest['selected_checkpoint'])
