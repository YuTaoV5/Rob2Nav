"""Immutable run-start copies of our wrapper source and upstream identity."""
import hashlib,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path


def capture(out,args,upstream):
    out=Path(out);source=out/'source';source.mkdir(exist_ok=True)
    hashes={}
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path,source/path.name)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=upstream,text=True).strip()
    if dirty:raise RuntimeError('Upstream tracked source was changed: '+dirty)
    data={'args':vars(args),'upstream_revision':revision,'upstream_tracked_clean':True,
        'wrapper_sha256':hashes,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'torch':sys.modules['torch'].__version__}
    if args.checkpoint:data['checkpoint_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(data,indent=2))
