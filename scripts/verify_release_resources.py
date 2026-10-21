"""Verify the published resource snapshot without importing/executing model pickles."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/release_20261021'));a=p.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    total=0
    for entry in manifest['files']:
        path=(root/entry['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Manifest path escapes snapshot')
        if path.stat().st_size!=entry['bytes']:raise ValueError('Size mismatch: '+entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Hash mismatch: '+entry['path'])
        total+=entry['bytes']
    published=0
    if (root/'SHA256SUMS').is_file():
        for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            digest,name=line.split('  ',1);path=(root/name).resolve()
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Published checksum mismatch: '+name)
            published+=1
    print(json.dumps(dict(verified=len(manifest['files']),bytes=total,published_files_verified=published,missing=manifest['missing'],excluded=manifest['excluded'])))


if __name__=='__main__':main()
