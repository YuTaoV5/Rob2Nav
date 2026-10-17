"""Plot measured foot trajectories against the initial ideal height scan.

The scan only covers its actual initial footprint; white outside means unknown.
This is a failure diagnostic, not a planned foothold plot.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('trial');a=p.parse_args();root=Path(a.trial)
scan=json.loads((root/'initial_map.json').read_text())
trace=json.loads((root/'trace.json').read_text());summary=json.loads((root/'summary.json').read_text())
points=np.array(scan['points']);feet=np.array([r['feet'] for r in trace])
forces=np.array([r['foot_forces'] for r in trace]);times=np.array([r['time'] for r in trace])
pos=np.array([r['position'] for r in trace]);upright=np.array([r['upright'] for r in trace])
fig,axes=plt.subplots(1,3,figsize=(14,4))
axes[0].scatter(points[:,0],points[:,1],c=points[:,2],vmin=-1.2,vmax=.02,cmap='Greys_r',marker='s',s=25)
for i,name in enumerate(scan['foot_names']):
    line=axes[0].plot(feet[:,i,0],feet[:,i,1],label=name,linewidth=1)[0]
    contact=forces[:,i,2]>5
    axes[0].scatter(feet[contact,i,0],feet[contact,i,1],s=9,color=line.get_color())
    axes[1].plot(times,feet[:,i,2],label=name)
axes[0].plot(pos[:,0],pos[:,1],color='black',linestyle='--',label='base')
axes[0].set_aspect('equal');axes[0].set_xlabel('route x (m)');axes[0].set_ylabel('route y (m)')
axes[0].set_title('Measured paths; dots: vertical contact >5 N')
axes[0].legend(fontsize=7)
axes[1].plot(times,pos[:,2],color='black',linestyle='--',label='base')
axes[1].set_xlabel('time (s)');axes[1].set_ylabel('height from tile origin (m)');axes[1].legend(fontsize=7)
axes[1].set_title('Measured foot / base height')
axes[2].plot(times,upright,label='upright cos(tilt)')
axes[2].plot(times,pos[:,0],label='base forward distance (m)')
axes[2].set_xlabel('time (s)');axes[2].legend(fontsize=8);axes[2].set_title('First episode, environment 0')
for ax in axes:ax.grid(alpha=.2)
episode=next(r for r in summary['episodes'] if r['env']==0)
fig.suptitle(f"{summary['args']['terrain']}: {episode.get('termination_terms',[])}; success={episode['success']}")
fig.tight_layout();fig.savefig(root/'foot_diagnostic.png',dpi=160)
print('PLOT_COMPLETE',root/'foot_diagnostic.png')
