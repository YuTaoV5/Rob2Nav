"""Offline first-contact/heading evidence for matched shallow/deep probes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from perceptive_rl.ame_terrain import boxes

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
fig,axes=plt.subplots(1,3,figsize=(15,5));report=[]
for ax,name in zip(axes,['deep05','shallow05','deep1']):
    folder=root/name;summary=json.loads((folder/'summary.json').read_text())
    trace=json.loads((folder/'trace.json').read_text());diag=json.loads((folder/'map_diagnostic.json').read_text())
    d=summary['args']['difficulty'];r=np.array([x['position'] for x in trace]);feet=np.array([x['feet'] for x in trace]);force=np.array([x['forces'] for x in trace])
    contact=force[:,:,2]>5;first=contact&~np.vstack((np.zeros((1,4),dtype=bool),contact[:-1]))
    for x0,x1,y0,y1,top,_ in boxes(d,'ame_stakes'):
        x0-=4;x1-=4;y0-=4;y1-=4
        if top>-.1 and x1>-.4 and x0<3.8 and y1>-.8 and y0<.8:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#cccccc',edgecolor='#999999',lw=.3))
    ax.plot(r[:,0],r[:,1],'k-',label='base path')
    for i,(leg,color) in enumerate(zip(['FL','FR','RL','RR'],['red','blue','orange','purple'])):
        points=feet[first[:,i],i];ax.scatter(points[:,0],points[:,1],s=12,c=color,label=leg)
    edge=next((x for x in diag if x['position'][0]>.75),None)
    below=contact&(feet[:,:,2]<-.1)
    record={'trial':name,'geometry':summary['geometry'],'successes':summary['successes'],'total':summary['total'],
        'failures':sum(x['failure'] for x in summary['episodes']),
        'env0_edge_yaw_degrees':float(np.degrees(edge['yaw'])) if edge else None,
        'env0_edge_lateral_m':edge['position'][1] if edge else None,
        'env0_below_platform_contact_samples':int(below.sum()),
        'env0_final_position':trace[-1]['position'],'note':'Contact samples may repeat; robot 0 only. Pit-bottom dependency is not established by this counter.'}
    report.append(record);ax.set_xlim(-.4,3.8);ax.set_ylim(-.8,.8);ax.set_aspect('equal')
    ax.set_title(f"{name}: {summary['successes']}/{summary['total']}\nfoot dots=first >5N contacts",fontsize=10)
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
axes[0].legend(fontsize=7);fig.suptitle('V3 matched probes: original support layout; shallow floor is diagnostic only')
fig.tight_layout();fig.savefig(out/'foot_contacts.png',dpi=160);plt.close(fig)
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
