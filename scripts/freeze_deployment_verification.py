"""Create an immutable, minimal verification workspace while the main repo evolves."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for relative in ['rob2nav','go2_nav','deployment','skills/navigation','vendor/RoboGauge/robogauge','vendor/RoboGauge/resources/robots/go2']:
        shutil.copytree(root/relative,out/relative,ignore=shutil.ignore_patterns('__pycache__','.git','logs','evidence'))
    for file in (root/'deployment/skills').glob('*.json'):
        data=json.loads(file.read_text())
        if data.get('policy'):
            src=root/data['policy'];dst=out/data['policy'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            if src.with_suffix('.json').is_file():shutil.copyfile(src.with_suffix('.json'),dst.with_suffix('.json'))
    manifest={f.relative_to(out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
    (out/'verification_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(path=str(out),files=len(manifest),source_manifest_sha256=hashlib.sha256((out/'verification_sources.json').read_bytes()).hexdigest())))


if __name__=='__main__':main()
