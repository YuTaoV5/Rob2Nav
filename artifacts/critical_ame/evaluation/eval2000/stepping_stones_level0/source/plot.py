"""Plot the retained checkpoint chain and measured first-episode outcomes."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--evaluation',nargs='+',default=[]);p.add_argument('--output',required=True)
a=p.parse_args();runs=Path(a.runs);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
segments=[('stage1_0000_1000',0,700),('stage1_fixed0701',701,999),('stage1_1000_3000',1000,2999)]
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
for name,lo,hi in segments:
    path=runs/name;files=list(path.glob('events.out.tfevents.*'))
    if not files:continue
    events=EventAccumulator(str(files[0])).Reload()
    for ax,tag in zip(axes[:2],('Train/mean_reward','Curriculum/terrain_levels')):
        samples=[v for v in events.Scalars(tag) if lo<=v.step<=hi]
        ax.plot([v.step for v in samples],[v.value for v in samples],color='#167bb6')
    rows=[]
    for line in (path/'iterations.jsonl').read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if lo<=row['iteration']<=hi:rows.append(row)
    axes[2].plot([r['iteration'] for r in rows],[r['std_max'] for r in rows],color='#167bb6')
    axes[2].plot([r['iteration'] for r in rows],[r['std_min'] for r in rows],color='#999999')
for ax,title in zip(axes,('Training return (not validation)','Curriculum mean (not validation)','Action std: min / max')):
    ax.set_title(title);ax.set_xlabel('PPO iteration');ax.grid(alpha=.2)
    for boundary in (701,1000):ax.axvline(boundary,color='#bbbbbb',linestyle=':',linewidth=1)
fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)
rows=[]
for folder in a.evaluation:
    for path in Path(folder).glob('*/summary.json'):
        d=json.loads(path.read_text());rows.append({'checkpoint':Path(d['args']['checkpoint']).name,
            'terrain':d['args']['terrain'],'level':d['args']['level'],
            'reset_jitter':d['args'].get('reset_jitter',0),'yaw_jitter':d['args'].get('yaw_jitter',0),
            'successes':d['successes'],'total':d['total']})
(out/'measured_results.json').write_text(json.dumps(rows,indent=2))
print('PLOTS_COMPLETE',len(rows),'evaluation cases')
