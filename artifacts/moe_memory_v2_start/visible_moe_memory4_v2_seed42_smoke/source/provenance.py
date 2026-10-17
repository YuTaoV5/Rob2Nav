"""Exact source snapshots for each new experiment, without modifying upstream."""
import hashlib,json,platform
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path


def capture(out,args):
    source=Path(out)/'source';source.mkdir(exist_ok=True);hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        data=path.read_bytes();(source/path.name).write_bytes(data)
        hashes[path.name]=hashlib.sha256(data).hexdigest()
    dependencies={};depdir=source/'perceptive_rl';depdir.mkdir(exist_ok=True)
    for name in ('terrain.py','geometry.py','ame_terrain.py','sensors.py'):
        path=Path(__file__).parents[1]/'perceptive_rl'/name;data=path.read_bytes()
        (depdir/name).write_bytes(data);dependencies[name]=hashlib.sha256(data).hexdigest()
    packages={}
    for name in ('torch','rsl-rl','isaaclab','isaaclab-rl','isaacsim'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    record={'args':vars(args),'source_sha256':hashes,'python':platform.python_version(),
            'dependency_sha256':dependencies,'packages':packages,
            'moe_sha256':hashlib.sha256(Path(args.moe).read_bytes()).hexdigest()}
    (Path(out)/'source_manifest.json').write_text(json.dumps(record,indent=2))
    return record
