"""V4: true deep pits, measured stage gates, plateau detection at any level."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--initializer',required=True)
p.add_argument('--output',required=True);p.add_argument('--run-prefix',required=True);p.add_argument('--smoke',required=True)
p.add_argument('--chunks',type=int,default=5);p.add_argument('--chunk-iterations',type=int,default=500)
p.add_argument('--num-envs',type=int,default=512);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
smoke=Path(a.smoke)
if not (smoke/'completed.json').exists():raise RuntimeError('Native short training must pass')
if json.loads((smoke/'manifest.json').read_text())['args'].get('course')!='wide_deep':raise RuntimeError('Wrong smoke experiment')


def run(name,cmd):
    (out/'status.json').write_text(json.dumps({'state':name,'command':cmd},indent=2))
    with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':cmd,'returncode':r.returncode});(out/'commands.json').write_text(json.dumps(records,indent=2))
    if r.returncode:
        (out/'status.json').write_text(json.dumps({'state':'failed','stage':name,'returncode':r.returncode}));raise SystemExit(r.returncode)


previous=None;last_metrics=None;plateau=0
for chunk in range(a.chunks):
    folder=Path(f'{a.run_prefix}_chunk{chunk+1}')
    cmd=['/root/isaacsim/python.sh','-m','moe_perceptive.train','--headless','--moe',a.moe,
        '--navigation','--teacher-map','--course','wide_deep','--num-envs',str(a.num_envs),
        '--iterations',str(a.chunk_iterations),'--warmup','0','--fusion','0','--actor','0','--output',str(folder)]
    cmd+=['--resume',previous] if previous else ['--initialize-from',a.initializer]
    run(f'train_chunk{chunk+1}',cmd)
    previous=str(folder/f'model_{(chunk+1)*a.chunk_iterations-1}.pt')
    gate=out/f'gate_chunk{chunk+1}'
    run(f'strict_chunk{chunk+1}',['python3','-m','moe_perceptive.suite','--moe',a.moe,'--checkpoint',previous,
        '--navigation','--teacher-map','--video-narrow','--output',str(gate)])
    strict=json.loads((gate/'commands.json').read_text());sparse=[r for r in strict if r['kind'].startswith('ame_')]
    # Additional precursor tests are explicitly distinct from original scores.
    precursors=[]
    for kind in ('ame_stakes','ame_alternate'):
        target=out/f'precursor_chunk{chunk+1}_{kind}'
        run(f'precursor_chunk{chunk+1}_{kind}',['/root/isaacsim/python.sh','-m','moe_perceptive.evaluate',
            '--headless','--moe',a.moe,'--checkpoint',previous,'--navigation','--teacher-map','--course','wide_deep',
            '--kind',kind,'--difficulty','0','--check-map','--output',str(target)])
        d=json.loads((target/'summary.json').read_text());precursors.append(d['successes']/d['total'])
    h=json.loads((folder/'status.json').read_text())['terrain_histograms'][4:]
    mean_level=sum(sum(i*n for i,n in enumerate(v)) for v in h)/sum(sum(v) for v in h)
    metrics={'chunk':chunk+1,'checkpoint':previous,'sparse_mean_level':mean_level,
        'strict_sparse_successes':sum(r['successes'] for r in sparse),'precursor_success_rates':precursors}
    with (out/'gates.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
    if all(r['successes']/r['total']>=.9 for r in strict):
        (out/'status.json').write_text(json.dumps({'state':'teacher_gate_passed',**metrics,'note':'Still privileged teacher; student/independent seeds/MuJoCo required.'}));break
    if last_metrics:
        improved=(mean_level>last_metrics['sparse_mean_level']+.15 or
                  metrics['strict_sparse_successes']>last_metrics['strict_sparse_successes'] or
                  sum(precursors)>sum(last_metrics['precursor_success_rates'])+.1)
        plateau=0 if improved else plateau+1
    last_metrics=metrics
    if plateau>=2:
        (out/'status.json').write_text(json.dumps({'state':'needs_diagnosis',**metrics,'reason':'No material level, strict-success or precursor improvement over two gates at any level.'}));break
else:
    (out/'status.json').write_text(json.dumps({'state':'budget_finished','checkpoint':previous,'note':'Inspect measured gates; no success claim.'}))
