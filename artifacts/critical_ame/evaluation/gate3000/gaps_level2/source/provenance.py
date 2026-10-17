"""Capture the exact upstream checkout and original training artifacts."""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PIN='15d02c891273d3f4c9b55c9afe3073bcd2d9152f'
p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.upstream)
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
revision=git('rev-parse','HEAD');assert revision==PIN,revision
dirty=git('status','--porcelain','--untracked-files=no');assert not dirty,dirty
sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
    for folder in ('rsl_rl/rsl_rl','source/ame_locomotion','scripts/go2') for f in (root/folder).rglob('*.py')}
packages={}
for name in ('torch','tensordict','isaaclab','isaaclab_tasks','onnx'):
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]=None
result={'upstream':'https://github.com/criticaldamped/AME_Locomotion','revision':revision,
    'tracked_files_clean':True,'source_sha256':sources,'packages':packages,
    'timestamp_utc':datetime.now(timezone.utc).isoformat(),
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
print(revision,len(sources),'source files')
