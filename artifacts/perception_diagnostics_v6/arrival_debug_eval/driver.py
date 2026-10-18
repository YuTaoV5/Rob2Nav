import subprocess,json
from pathlib import Path
r=Path('outputs/moe_perceptive/arrival_debug_eval')
commands=[['strict', ['python3', '-m', 'moe_perceptive.suite', '--moe', 'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt', '--checkpoint', '/root/gpufree-data/MoE-runs/arrival_debug_cont140/model_199.pt', '--navigation', '--teacher-map', '--video-narrow', '--output', 'outputs/moe_perceptive/arrival_debug_eval/strict']], ['ame_stakes', ['/root/isaacsim/python.sh', '-m', 'moe_perceptive.evaluate', '--headless', '--moe', 'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt', '--checkpoint', '/root/gpufree-data/MoE-runs/arrival_debug_cont140/model_199.pt', '--navigation', '--teacher-map', '--course', 'wide_deep', '--kind', 'ame_stakes', '--difficulty', '0', '--diagnostic', '--video', '--output', 'outputs/moe_perceptive/arrival_debug_eval/ame_stakes']], ['ame_alternate', ['/root/isaacsim/python.sh', '-m', 'moe_perceptive.evaluate', '--headless', '--moe', 'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt', '--checkpoint', '/root/gpufree-data/MoE-runs/arrival_debug_cont140/model_199.pt', '--navigation', '--teacher-map', '--course', 'wide_deep', '--kind', 'ame_alternate', '--difficulty', '0', '--diagnostic', '--video', '--output', 'outputs/moe_perceptive/arrival_debug_eval/ame_alternate']]]
for stage,c in commands:
 (r/"status.json").write_text(json.dumps({"state":stage}))
 with (r/(stage+".log")).open("w") as f:code=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT).returncode
 valid=len(list((r/"strict").glob("*/summary.json")))==9 if stage=="strict" else (r/stage/"summary.json").exists()
 if code or not valid:
  (r/"status.json").write_text(json.dumps({"state":"failed","stage":stage,"returncode":code,"outputs_valid":valid}));raise SystemExit(1)
(r/"status.json").write_text(json.dumps({"state":"evaluations_completed"}))
