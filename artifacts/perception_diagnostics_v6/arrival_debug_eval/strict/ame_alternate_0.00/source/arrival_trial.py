"""One bounded arrival-reward probe, not an automatic continuation controller."""
import json,subprocess
from pathlib import Path
out=Path('outputs/moe_perceptive/arrival_v5_probe')
out.mkdir(parents=True,exist_ok=False)
moe='vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt'
run='/root/gpufree-data/MoE-runs/arrival_v5_probe'
initializer='/root/gpufree-data/MoE-runs/wide_deep_v4_chunk4/model_1999.pt'
commands=[('train',['/root/isaacsim/python.sh','-m','moe_perceptive.train','--headless','--moe',moe,'--navigation','--teacher-map','--course','wide_deep','--arrival-aware','--num-envs','512','--iterations','200','--warmup','0','--fusion','0','--actor','0','--initialize-from',initializer,'--output',run])]
ck=run+'/model_199.pt'
commands.append(('strict',['python3','-m','moe_perceptive.suite','--moe',moe,'--checkpoint',ck,'--navigation','--teacher-map','--video-narrow','--output',str(out/'strict')]))
for kind in ['ame_stakes','ame_alternate']:
 commands.append((kind,['/root/isaacsim/python.sh','-m','moe_perceptive.evaluate','--headless','--moe',moe,'--checkpoint',ck,'--navigation','--teacher-map','--course','wide_deep','--kind',kind,'--difficulty','0','--diagnostic','--video','--output',str(out/kind)]))
(out/'plan.json').write_text(json.dumps({'initializer':initializer,'note':'Fresh optimizer and curriculum; exploratory probe, not isolated causal ablation. Original geometry success criteria unchanged. No further training automatically.','commands':commands},indent=2))
for stage,cmd in commands:
 (out/'status.json').write_text(json.dumps({'state':stage,'command':cmd},indent=2))
 with (out/(stage+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  (out/'status.json').write_text(json.dumps({'state':'failed','stage':stage,'returncode':r.returncode}));raise SystemExit(r.returncode)
(out/'status.json').write_text(json.dumps({'state':'evaluations_completed','note':'Inspect results; not task completion.'}))
