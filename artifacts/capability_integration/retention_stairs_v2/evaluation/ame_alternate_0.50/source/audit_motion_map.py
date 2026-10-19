"""Offline audit of privileged robot0 maps in moving diagnostic frames."""
import argparse,json
from pathlib import Path
import numpy as np
from perceptive_rl.ame_terrain import heightfield as original
from moe_perceptive.sparse_course import heightfield as wide
p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();root=Path(a.input)
s=json.loads((root/'summary.json').read_text());args=s['args'];h=(wide if args.get('course')=='wide_deep' else original)(args['difficulty'],args['kind'])
y,x=np.meshgrid(np.linspace(-.5,.5,21),np.linspace(-.55,1.05,33),indexing='ij');rows=[]
for frame in json.loads((root/'map_diagnostic.json').read_text()):
 px,py,pz=frame['position'];yaw=frame['yaw'];wx=px+x*np.cos(yaw)-y*np.sin(yaw)+4;wy=py+x*np.sin(yaw)+y*np.cos(yaw)+4
 ix=np.floor(wx/.05).astype(int);iy=np.floor(wy/.05).astype(int);inside=(ix>=1)&(ix<159)&(iy>=1)&(iy<159)
 ix=ix.clip(1,158);iy=iy.clip(1,158);truth=h[ix,iy]
 # Exclude neighborhoods of height edges, not all grid boundaries.
 edge=np.zeros_like(inside)
 for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:edge|=np.abs(h[ix+dx,iy+dy]-truth)>.001
 maps=np.array(frame['visible_map']);known=maps[1]>.5
 # Compare the encoded height itself, retaining the policy's documented clip.
 expected=np.clip(pz-truth-.5,-1.5,2.)
 usable=inside&~edge&known
 error=np.abs(maps[0]-expected)
 rows.append({'time':frame['time'],'position':frame['position'],'yaw':yaw,'samples':int(usable.sum()),'correct':int((error[usable]<.002).sum()),'max_error_m':float(error[usable].max()) if usable.any() else None})
count=sum(v['samples'] for v in rows);correct=sum(v['correct'] for v in rows)
report={'scope':'robot0 privileged map; expected yaw-aligned grid and analytic terrain; one-cell terrain edges excluded; encoded clipping respected','samples':count,'correct':correct,'fraction':correct/max(count,1),'frames':rows}
(root/'motion_map_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'}))
