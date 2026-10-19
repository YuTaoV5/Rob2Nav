"""One bounded spatial-attention probe, not an automatic continuation controller."""
import json,subprocess
from pathlib import Path
out=Path('outputs/moe_perceptive/spatial_v7_probe')
out.mkdir(parents=True,exist_ok=False)
moe='vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'
run='/root/gpufree-data/MoE-runs/spatial_v7_probe'
initializer='/root/gpufree-data/MoE-runs/foothold_v6_probe/model_199.pt'
commands=[('train',['/root/isaacsim/python.sh','-m','moe_perceptive.train','--headless','--moe',moe,'--navigation','--teacher-map','--course','wide_deep','--arrival-aware','--foothold-aware','--numeric-debug','--spatial-attention','--num-envs','512','--iterations','200','--warmup','0','--fusion','0','--actor','0','--initialize-from',initializer,'--output',run])]
ck=run+'/model_199.pt'
commands.append(('strict',['python3','-m','moe_perceptive.suite','--moe',moe,'--checkpoint',ck,'--navigation','--teacher-map','--video-narrow','--output',str(out/'strict')]))
for kind in ['ame_stakes','ame_alternate']:
 commands.append((kind,['/root/isaacsim/python.sh','-m','moe_perceptive.evaluate','--headless','--moe',moe,'--checkpoint',ck,'--navigation','--teacher-map','--course','wide_deep','--kind',kind,'--difficulty','0','--diagnostic','--video','--output',str(out/kind)]))
(out/'plan.json').write_text(json.dumps({'initializer':initializer,'note':'Fresh optimizer and curriculum; exploratory probe, not isolated causal ablation. Original geometry success criteria unchanged. No further training automatically.','commands':commands},indent=2))
for stage,cmd in commands:
 (out/'status.json').write_text(json.dumps({'state':stage,'command':cmd},indent=2))
 with (out/(stage+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 valid=True
 if stage=='train':
  valid=(Path(run)/'completed.json').exists() and not (Path(run)/'failed.json').exists() and Path(ck).exists()
 elif stage=='strict':
  valid=len(list((out/'strict').glob('*/summary.json')))==9
 else:valid=(out/stage/'summary.json').exists()
 if r.returncode or not valid:
  (out/'status.json').write_text(json.dumps({'state':'failed','stage':stage,'returncode':r.returncode,'outputs_valid':valid}));raise SystemExit(r.returncode or 1)
(out/'status.json').write_text(json.dumps({'state':'evaluations_completed','note':'Inspect results; not task completion.'}))


