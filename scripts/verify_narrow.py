"""Verify narrow terrain evidence without modifying archived records."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio

base=Path(__file__).resolve().parents[1]/'artifacts/narrow'
proof=json.loads((base/'layout_equivalence.json').read_text())
assert len(proof['checks'])==6 and all(x['support_mask_identical'] for x in proof['checks'])
sources=0;videos=[]
for engine in ('native','mujoco'):
    commands=json.loads((base/engine/'commands.json').read_text())
    assert len(commands)==3 and all(x['returncode']==0 for x in commands)
    for path in (base/engine).glob('*/summary.json'):
        data=json.loads(path.read_text())
        for name,sha in data['source_sha256'].items():
            assert hashlib.sha256((path.parent/'source'/name).read_bytes()).hexdigest()==sha
            sources+=1
        for video in path.parent.glob('*.mp4'):
            reader=imageio.get_reader(video);frames=reader.count_frames();reader.close()
            trace=json.loads((path.parent/'trajectory_0.json').read_text())
            assert frames==(len(trace)+1)//2 and frames>0
            videos.append({'path':str(video.relative_to(base)),'frames':frames})
print(json.dumps({'layout_checks':6,'source_references':sources,'videos':videos},indent=2))
