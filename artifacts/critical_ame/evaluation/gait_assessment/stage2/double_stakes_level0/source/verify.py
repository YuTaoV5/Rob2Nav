"""Read-only verification of an archived original-project reproduction run."""
import argparse,hashlib,json
from pathlib import Path
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/critical_ame');a=p.parse_args()
root=Path(a.root);manifest=json.loads((root/'manifest.json').read_text());weights=sources=0;videos=[]
for run in manifest['runs'].values():
    if 'checkpoint' in run:
        record=run['checkpoint'];assert hashlib.sha256((root/record['path']).read_bytes()).hexdigest()==record['sha256'];weights+=1
for path in root.glob('**/manifest.json'):
    data=json.loads(path.read_text())
    for name,sha in data.get('wrapper_sha256',{}).items():
        assert hashlib.sha256((path.parent/'source'/name).read_bytes()).hexdigest()==sha;sources+=1
for path in root.glob('evaluation/**/summary.json'):
    d=json.loads(path.read_text());assert d['total']==len(d['episodes'])
    assert d['successes']==sum(x['success'] for x in d['episodes'])
    for video in path.parent.glob('*.mp4'):
        reader=imageio.get_reader(video);frames=reader.count_frames();reader.close()
        assert frames==d['video_frames'] and frames>0
        videos.append({'path':str(video.relative_to(root)),'frames':frames})
print(json.dumps({'checkpoints':weights,'wrapper_source_references':sources,'videos':videos},indent=2))
