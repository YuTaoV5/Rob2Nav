"""Verify portable inputs, or preflight/explicitly submit one job through Studio API."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.pipeline_api import request
PACK=ROOT/'artifacts/空翻续训交接_20261023'


def verify():
    manifest=json.loads((PACK/'资源清单.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():raise ValueError('Missing/invalid resource: '+row['path'])
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Resource hash mismatch: '+row['path'])
        if p.suffix=='.pt':
            with zipfile.ZipFile(p) as z:
                if z.testzip():raise ValueError('Invalid checkpoint archive: '+row['path'])
    print(json.dumps(dict(verified_files=len(manifest['files']),training_submitted=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true')
    p.add_argument('--config',type=Path)
    p.add_argument('--api',default='http://127.0.0.1:8765/api/')
    p.add_argument('--submit',action='store_true',help='Explicitly submit ONE configuration after its documented gates pass')
    a=p.parse_args()
    if a.verify or a.config is None:verify()
    if a.submit and a.config is None:p.error('--submit requires --config')
    if a.config:
        c=json.loads(a.config.read_text(encoding='utf-8'));base=a.api.rstrip('/')+'/'
        check=request('preflight',c,base);print(json.dumps(check,ensure_ascii=False,indent=2))
        if not check['valid']:raise SystemExit(2)
        if not a.submit:return
        matches=[j for j in request('jobs',base=base) if j['name']==c['name']]
        if len(matches)>1:raise RuntimeError('Multiple same-name jobs; inspect before submitting')
        if matches:
            j=matches[0]
            # Normalize schema defaults before comparing a saved config to the API representation.
            from studio.config import RunConfig
            if RunConfig.model_validate(j['config']).model_dump()!=RunConfig.model_validate(c).model_dump():
                raise RuntimeError('Existing job has different configuration; do not silently reuse it')
        else:j=request('jobs',c,base)
        if j['state']=='draft':j=request('jobs/'+j['id']+'/submit',{},base)
        print(json.dumps(dict(job=j['id'],state=j['state'],created_or_reused=True),ensure_ascii=False))


if __name__=='__main__':main()
