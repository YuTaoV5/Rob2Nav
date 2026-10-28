#!/usr/bin/env python3
"""Fetch author release assets, verify identity/hash and safely extract regular files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def extract(archive, destination):
    root=destination.resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:bz2') as tar:
        for member in tar:
            parts=Path(member.name).parts
            if len(parts)<=1:continue
            relative=Path(*parts[1:]);target=(root/relative).resolve()
            if os.path.commonpath([str(root),str(target)])!=str(root):raise ValueError('Unsafe archive path')
            if member.isdir():target.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
            else:raise ValueError('Archive links/device entries are not allowed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['asr','tts']);p.add_argument('--offline-archive',type=Path)
    a=p.parse_args()
    if a.offline_archive and not a.only:p.error('--offline-archive requires --only')
    lock=json.loads((ROOT/'locks/voice-models.json').read_text(encoding='utf-8'));out=ROOT/'data/models/voice';out.mkdir(parents=True,exist_ok=True)
    for name,model in lock['models'].items():
        if a.only and a.only!=name:continue
        target=out/name;archive=a.offline_archive or out/model['asset_name']
        if target.exists() and any(target.iterdir()):
            record=json.loads((target/'download-manifest.json').read_text(encoding='utf-8'))
            if record['asset_id']!=model['asset_id'] or model.get('sha256') and record['archive_sha256']!=model['sha256']:
                raise RuntimeError('Installed model identity differs: '+name)
            expected=record['files']
            actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() and p.name!='download-manifest.json'}
            if actual!=set(expected):raise RuntimeError('Installed model file set changed: '+name)
            for relative,digest in expected.items():
                path=(target/relative).resolve()
                if not path.is_relative_to(target.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Installed model hash mismatch: '+name)
            print(name,'already installed and hashes verified',flush=True);continue
        if not a.offline_archive:
            # Metadata identity is checked even when the author did not publish a SHA256.
            req=urllib.request.Request('https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/assets/'+str(model['asset_id']),headers={'Accept':'application/vnd.github+json'})
            with urllib.request.urlopen(req,timeout=30) as response:metadata=json.load(response)
            if metadata['id']!=model['asset_id'] or metadata['name']!=model['asset_name'] or metadata['size']!=model['size']:
                raise RuntimeError('Author asset identity changed; review the source lock')
            temporary=archive.with_suffix('.partial')
            if archive.exists() and archive.stat().st_size<model['size']:
                if temporary.exists():raise RuntimeError('Two incomplete archives exist; inspect before retrying')
                archive.replace(temporary)
            if not archive.exists():
                for attempt in range(16):
                    offset=temporary.stat().st_size if temporary.exists() else 0
                    if offset==model['size']:break
                    if offset>model['size']:raise RuntimeError('Partial archive is larger than source')
                    headers={'Range':'bytes=%d-'%offset} if offset else {}
                    request=urllib.request.Request(model['url'],headers=headers)
                    try:
                        with urllib.request.urlopen(request,timeout=60) as response:
                            if offset and response.status!=206:raise RuntimeError('Server ignored resume range; preserve partial and retry another source')
                            if response.status==206 and not response.headers.get('Content-Range','').startswith('bytes %d-'%offset):
                                raise RuntimeError('Unexpected HTTP range')
                            with temporary.open('ab' if offset else 'wb') as file:shutil.copyfileobj(response,file,1024*1024)
                    except (TimeoutError,ConnectionError,urllib.error.URLError) as exc:
                        print(name,'network interrupted; preserving partial:',type(exc).__name__,flush=True)
                        if attempt==15:raise
                        time.sleep(1);continue
                    new_size=temporary.stat().st_size
                    print(name,'download bytes',new_size,'/',model['size'],flush=True)
                    if new_size<=offset:raise RuntimeError('Download made no progress')
                if temporary.stat().st_size!=model['size']:raise RuntimeError('Download incomplete; rerun to resume')
                if model.get('sha256') and hashlib.sha256(temporary.read_bytes()).hexdigest()!=model['sha256']:
                    raise RuntimeError('Downloaded archive hash mismatch; incomplete file preserved for inspection')
                temporary.replace(archive)
        if archive.stat().st_size!=model['size']:raise RuntimeError('Model archive size mismatch: '+name)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        if model.get('sha256') and digest!=model['sha256']:raise RuntimeError('Model archive SHA256 mismatch: '+name)
        if target.exists() and any(target.iterdir()):raise RuntimeError('Model folder already populated; inspect it before replacement: '+str(target))
        extract(archive,target)
        manifest={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file()}
        (target/'download-manifest.json').write_text(json.dumps(dict(asset_id=model['asset_id'],archive_sha256=digest,files=manifest),indent=2))
        print(name,'archive SHA256',digest,'files',len(manifest),flush=True)


if __name__=='__main__':main()
