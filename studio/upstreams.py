"""Pinned external-cache contracts. Unlicensed upstream assets are not vendored."""
import hashlib
import json
from pathlib import Path

LOCK=Path(__file__).with_name('upstream_lock.json')


def source(root,project):
    entries=json.loads(LOCK.read_text(encoding='utf-8'))
    if project not in entries:raise ValueError('This reviewed upstream cache is not installed yet')
    entry=entries[project]
    folder=(Path(root).resolve().parent/'Rob2Nav-upstreams'/entry['folder']).resolve()
    if not folder.is_dir():raise ValueError('Upstream private cache is missing; follow docs/UNIFIED_CAPABILITIES.md')
    for name,digest in entry['sha256'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder) or not p.is_file():raise ValueError('Pinned source file missing: '+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Pinned source hash mismatch: '+name)
    return folder,entry


def pie_python(root):return Path(root).resolve().parent/'envs/go2-pie/bin/python'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Verify reviewed external source assets without executing upstream code')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--project',choices=('pie','lsd'),required=True)
    args=parser.parse_args()
    folder,entry=source(args.root,args.project)
    print(json.dumps({'folder':str(folder),'commit':entry.get('commit'),'verified_files':len(entry['sha256'])}))
