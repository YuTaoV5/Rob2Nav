"""Build independently extractable, checksummed research-data release archives.

No credentials are read. Existing runs are read-only; SQLite is backed up online.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import tarfile

GROUPS={'project-artifacts':'Rob2Nav/artifacts','project-outputs':'Rob2Nav/outputs',
        'moe-runs':'MoE-runs','ame-runs':'AME-runs','ame-upstream':'AME-criticaldamped',
        'moe-upstream':'MoE-upstream','legacy-initial':'go2-mapless',
        'legacy-terrain':'rob2nav-terrain','legacy-perceptive':'Rob2Nav-perceptive-63f5bcf'}
SKIP_DIRS={'.git','.ssh','.aws','node_modules','__pycache__','.pytest_cache','.venv','venv'}
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def excluded(path,relative):
    if any(p in SKIP_DIRS or p.startswith('pytest-cache-files-') for p in relative.parts):return 'cache, dependency runtime, or private metadata'
    if path.is_symlink():return 'symlink; external runtime must be installed separately'
    n=path.name.lower()
    if n.startswith('.env') or n in {'.npmrc','.pypirc','credentials','credentials.json','id_rsa','id_ed25519'} or path.suffix in {'.pem','.key'}:return 'credential file'
    if n.endswith(('.pyc','.pyo','.sqlite3-wal','.sqlite3-shm')):return 'runtime cache or SQLite transient journal'
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tag',default='full-archive-2026-10-19');p.add_argument('--part-mib',type=int,default=512)
    p.add_argument('--group',action='append',choices=GROUPS);p.add_argument('--refresh',action='store_true');a=p.parse_args()
    base=a.data_root.resolve();out=a.output.resolve()
    if out.is_relative_to(base/'Rob2Nav/outputs'):raise SystemExit('Archive output must be outside the input trees')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':1,'tag':a.tag,'created_utc':datetime.now(timezone.utc).isoformat(),'layout':'Extract into the parent of the Rob2Nav clone.',
              'notes':['All stages are historical evidence, not a claim of successful AME narrow-stake traversal.',
                       'Environment binaries and NVIDIA assets must be installed under their own terms.',
                       'Original transfer archives are retained as historical files, in addition to unpacked evidence.'],
              'groups':{},'archives':[],'files':[],'excluded':[]}
    if a.refresh:
        if not a.group:raise SystemExit('--refresh requires an explicit --group')
        manifest=json.loads((out/'data-manifest.json').read_text(encoding='utf8'))
        selected=set(a.group)
        manifest['files']=[x for x in manifest['files'] if x['group'] not in selected]
        manifest['archives']=[x for x in manifest['archives'] if x['group'] not in selected]
        manifest['excluded']=[x for x in manifest['excluded'] if not any(x['path'].startswith(GROUPS[g]+'/') for g in selected)]
    for group,relative in GROUPS.items():
        if a.group and group not in a.group:continue
        source=base/relative
        if not source.is_dir():manifest['groups'][group]={'missing':True};continue
        candidates=[]
        for path in sorted(source.rglob('*')):
            rel=path.relative_to(base);reason=excluded(path,rel)
            if path.is_dir():continue
            if reason:manifest['excluded'].append({'path':rel.as_posix(),'reason':reason});continue
            if not path.is_file():continue
            if path.stat().st_size<16*1024*1024:
                blob=path.read_bytes()
                if SECRET.search(blob):raise SystemExit('Secret-like content detected; review file: '+rel.as_posix())
            actual=path
            if path.name=='studio.sqlite3':
                actual=out/'studio-state-snapshot.sqlite3'
                with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as src,sqlite3.connect(actual) as dst:src.backup(dst)
            candidates.append((actual,rel.as_posix()))
        batches=[];batch=[];size=0
        for actual,rel in candidates:
            count=actual.stat().st_size
            if count>=2*1024**3:raise SystemExit('Single file exceeds GitHub asset limit: '+rel)
            if batch and size+count>a.part_mib*1024**2:batches.append(batch);batch=[];size=0
            batch.append((actual,rel));size+=count
        if batch:batches.append(batch)
        counts=Counter();total=0
        for index,batch in enumerate(batches,1):
            name=f'{group}.part{index:03}.tar.gz';target=out/name
            with tarfile.open(target,'w:gz',compresslevel=3) as tar:
                for actual,rel in batch:
                    before=actual.stat();sha=digest(actual)
                    tar.add(actual,arcname=rel,recursive=False)
                    after=actual.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise SystemExit('Input changed while packaging: '+rel)
                    manifest['files'].append({'path':rel,'bytes':before.st_size,'sha256':sha,'archive':name,'group':group})
                    counts[Path(rel).suffix.lower()]+=1;total+=before.st_size
            entry={'name':name,'bytes':target.stat().st_size,'sha256':digest(target),'group':group,'file_count':len(batch)}
            manifest['archives'].append(entry);print(json.dumps({'archive':name,'bytes':entry['bytes'],'files':len(batch)}),flush=True)
        manifest['groups'][group]={'root':relative,'file_count':len(candidates),'bytes':total,'types':dict(counts),'archives':len(batches)}
    manifest['total_files']=len(manifest['files']);manifest['total_bytes']=sum(x['bytes'] for x in manifest['files'])
    (out/'data-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    sums=[f"{x['sha256']}  {x['name']}" for x in manifest['archives']]
    sums.append(f"{digest(out/'data-manifest.json')}  data-manifest.json")
    (out/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf8')
    summary={k:v for k,v in manifest.items() if k not in ('files','excluded')};summary['excluded_count']=len(manifest['excluded'])
    (out/'release-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPLETE',manifest['total_files'],manifest['total_bytes'],flush=True)


if __name__=='__main__':main()
