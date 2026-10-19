"""Read-only validation of exact snapshot bytes, sources, counts and videos."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/moe_perceptive')
p.add_argument('--manifest-name',default='manifest.json');a=p.parse_args()
root=Path(a.root);manifest=json.loads((root/a.manifest_name).read_text());videos=[];sources=0
for name,record in manifest['files'].items():
    path=root/name;assert path.stat().st_size==record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
for path in root.rglob('source_manifest.json'):
    d=json.loads(path.read_text())
    for name,sha in d['source_sha256'].items():
        assert hashlib.sha256((path.parent/'source'/name).read_bytes()).hexdigest()==sha;sources+=1
    for name,sha in d.get('dependency_sha256',{}).items():
        assert hashlib.sha256((path.parent/'source'/'perceptive_rl'/name).read_bytes()).hexdigest()==sha;sources+=1
for path in root.rglob('summary.json'):
    d=json.loads(path.read_text())
    if 'episodes' not in d:continue
    assert d['total']==len(d['episodes']) and d['successes']==sum(r['success'] for r in d['episodes'])
    if d.get('video_frames',0):
        import imageio.v2 as imageio
        reader=imageio.get_reader(path.parent/'episode_0.mp4');n=reader.count_frames();reader.close()
        assert n==d['video_frames'];videos.append({'trial':str(path.parent.relative_to(root)),'frames':n})
print(json.dumps({'files':len(manifest['files']),'source_references':sources,'videos':videos,
    'selected_checkpoint':manifest.get('selected_checkpoint'),'run_completed':manifest.get('run_completed')},indent=2))
