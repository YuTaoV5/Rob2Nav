"""Check every indexed download and the explicit necessary model set, without unpickling."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8765');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    with urlopen(a.base+'/api/catalog?refresh=true') as f:c=json.load(f)
    bypath={m['path']:m for m in c['models']};checks=[];resource_checks=[]
    for m in json.loads((root/'studio/model_assets.json').read_text())['models']:
        path=root/m['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert m['path'] in bypath,m['path']
        assert digest==m['sha256'],m['path']
        checks.append(dict(path=m['path'],sha256=digest,bytes=path.stat().st_size,role=bypath[m['path']]['role']))
        for x in m.get('resources',[]):
            assert hashlib.sha256((root/x['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==x['sha256'],x['path']
            resource_checks.append(x['path'])
    downloads=[]
    for m in c['models']:
        with urlopen(Request(a.base+m['download_url'],headers={'Range':'bytes=0-0'})) as f:
            assert f.status==206,(m['path'],f.status)
            assert len(f.read())==1
            assert f.headers['Content-Range'].endswith('/'+str(m['bytes']))
        downloads.append(m['path'])
    assert not c['issues'],c['issues']
    report=dict(necessary_models=checks,companion_resource_checks=len(set(resource_checks)),indexed_count=len(c['models']),download_checks=len(downloads),missing=[],
                formats=dict(collections.Counter(m['format'] for m in c['models'])),
                roles=dict(collections.Counter(m['role'] for m in c['models'])),
                note='All listed files and range downloads checked; model bytes were not deserialized. Hardware qualification remains false.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='necessary_models'},ensure_ascii=False))
if __name__=='__main__':main()
