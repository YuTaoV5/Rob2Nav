from pathlib import Path
import subprocess,json,sys
p=Path('outputs/studio_diagnostics/left_v12_model12000_20261023');records=[]
for seed in [27001,27002,27003]:
 out=p/f'seed_{seed}'
 if out.exists():raise RuntimeError('Do not overwrite diagnostic evidence')
 cmd=[sys.executable,'-m','rob2nav.mujoco_skill','--kind','sideflip','--policy',str(p/'exported/policy.pt'),'--motion','artifacts/web_continuity_iteration_20261023/inputs/side_v12/sideflip.npz','--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip','--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--diagnostics','--seed',str(seed),'--output',str(out)]
 with (p/f'seed_{seed}.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Diagnostic execution failed')
 d=json.loads((out/'summary.json').read_text());records.append(dict(seed=seed,summary=str(out/'summary.json'),successes=d['successes'],total=d['total']))
 (p/'diagnostic_progress.json').write_text(json.dumps(dict(diagnostic_only=True,records=records),indent=2))
