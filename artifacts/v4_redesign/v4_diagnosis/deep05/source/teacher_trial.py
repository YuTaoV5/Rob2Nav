"""Checkpointed teacher experiment with periodic strict original-terrain gates.

Stops for diagnosis if two consecutive chunks fail to leave easiest sparse
levels. A successful teacher is never promoted as a sensor-based solution.
"""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--moe',required=True);p.add_argument('--initializer',required=True)
p.add_argument('--output',required=True);p.add_argument('--run-prefix',required=True)
p.add_argument('--chunks',type=int,default=5);p.add_argument('--chunk-iterations',type=int,default=1000)
p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--smoke',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
if not (Path(a.smoke)/'completed.json').exists():raise RuntimeError('Native smoke must pass first')


def run(name,command):
    (out/'status.json').write_text(json.dumps({'state':name,'command':command},indent=2))
    with (out/(name+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    records.append({'name':name,'command':command,'returncode':r.returncode})
    (out/'commands.json').write_text(json.dumps(records,indent=2))
    if r.returncode:
        (out/'status.json').write_text(json.dumps({'state':'failed','stage':name,'returncode':r.returncode}))
        raise SystemExit(r.returncode)


previous=None;stalled_chunks=0
for chunk in range(a.chunks):
    run_path=f'{a.run_prefix}_chunk{chunk+1}'
    command=['/root/isaacsim/python.sh','-m','moe_perceptive.train','--headless','--moe',a.moe,
        '--navigation','--teacher-map','--num-envs',str(a.num_envs),'--iterations',str(a.chunk_iterations),
        '--warmup','0','--fusion','0','--actor','0','--output',run_path]
    command+=['--resume',previous] if previous else ['--initialize-from',a.initializer]
    run(f'train_chunk{chunk+1}',command)
    final=(chunk+1)*a.chunk_iterations-1;previous=str(Path(run_path)/f'model_{final}.pt')
    evaluation=out/f'gate_chunk{chunk+1}'
    run(f'evaluate_chunk{chunk+1}',['python3','-m','moe_perceptive.suite','--moe',a.moe,
        '--navigation','--teacher-map','--checkpoint',previous,'--output',str(evaluation),'--video-narrow'])
    cases=json.loads((evaluation/'commands.json').read_text())
    success=all(c['successes']/c['total']>=.9 for c in cases)
    if success:
        (out/'status.json').write_text(json.dumps({'state':'teacher_gate_passed','checkpoint':previous,
            'note':'Privileged teacher only. Next: sensor student training, independent seeds, MuJoCo. No completion claim.'}))
        break
    status=json.loads((Path(run_path)/'status.json').read_text())
    h=status['terrain_histograms'][4:]  # sparse types in this eight-column mixture
    stuck=all(sum(v[1:])==0 for v in h)
    stalled_chunks=stalled_chunks+1 if stuck else 0
    if stalled_chunks>=2:
        (out/'status.json').write_text(json.dumps({'state':'needs_diagnosis','checkpoint':previous,
            'reason':'Sparse curriculum stayed at level zero for two chunks; do not blindly extend.'}))
        break
else:
    (out/'status.json').write_text(json.dumps({'state':'budget_finished','checkpoint':previous,
        'note':'Five chunk experiment exhausted; inspect strict evaluations before another design.'}))
