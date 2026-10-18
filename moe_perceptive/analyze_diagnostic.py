"""Offline foot-cell coverage diagnostic; never modifies policy actions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--trials',nargs='+',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
report=[];fig,axes=plt.subplots(1,len(a.trials),figsize=(6*len(a.trials),5),squeeze=False)
for col,path in enumerate(map(Path,a.trials)):
    rows=json.loads((path/'map_diagnostic.json').read_text());summary=json.loads((path/'summary.json').read_text())
    nearby=[]
    for row in rows:
        mask=np.asarray(row['visible_map'])[1];coverage=[]
        for x,y in row['foot_xy_body_yaw']:
            ix=int(np.rint((x+.55)/.05));iy=int(np.rint((y+.5)/.05))
            patch=mask[max(0,iy-1):min(21,iy+2),max(0,ix-1):min(33,ix+2)] if 0<=ix<33 and 0<=iy<21 else np.zeros(1)
            coverage.append(float(patch.mean()))
        nearby.append(coverage)
    edge=[i for i,row in enumerate(rows) if .5<row['position'][0]<1.5]
    report.append({'trial':str(path),'memory_lifetime':summary['args']['memory_lifetime'],
        'sensor_rig':summary['args'].get('sensor_rig','original'),
        'successes':summary['successes'],'total':summary['total'],
        'failures':sum(e['failure'] for e in summary['episodes']),
        'mean_final_x':float(np.mean([e['position'][0] for e in summary['episodes']])),
        'feet_order':rows[0]['feet_order'],'env0_samples':len(rows),'env0_edge_samples':len(edge),
        'env0_exact_cell_known_fraction':np.mean([r['foot_cell_known'] for r in rows],axis=0).tolist(),
        'env0_edge_neighborhood_known_fraction':np.mean(np.asarray(nearby)[edge],axis=0).tolist() if edge else None,
        'scope':'Coverage is robot 0 over time; neighborhood is a 3x3-cell average, not foothold success or all-env evidence.'})
    row=min(rows,key=lambda r:abs(r['position'][0]-.8));grid=np.asarray(row['visible_map']);ax=axes[0,col]
    ax.imshow(np.ma.masked_where(grid[1]<.5,grid[0]),origin='lower',extent=(-.575,1.075,-.525,.525),vmin=-.5,vmax=1.5,cmap='terrain',aspect='equal')
    xy=np.asarray(row['foot_xy_body_yaw']);ax.scatter(xy[:,0],xy[:,1],c='red',marker='x')
    for name,(x,y) in zip(row['feet_order'],xy):ax.text(x,y,name[:2],fontsize=9)
    ax.set_facecolor('#cccccc');ax.set_xlabel('forward x (m)');ax.set_ylabel('left y (m)')
    ax.set_title(f"{summary['args'].get('sensor_rig','original')}; memory {summary['args']['memory_lifetime']} s\nx={row['position'][0]:.2f} m; grey=unknown; red=feet",fontsize=10)
fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
