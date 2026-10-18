import subprocess,json
from pathlib import Path
commands=[['/root/isaacsim/python.sh', '-m', 'moe_perceptive.evaluate', '--headless', '--moe', 'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt', '--checkpoint', '/root/gpufree-data/MoE-runs/wide_deep_v4_chunk4/model_1999.pt', '--navigation', '--teacher-map', '--course', 'wide_deep', '--kind', 'ame_stakes', '--difficulty', '0', '--diagnostic', '--check-map', '--video', '--output', 'outputs/moe_perceptive/v4_plateau_diagnostic/ame_stakes'], ['/root/isaacsim/python.sh', '-m', 'moe_perceptive.evaluate', '--headless', '--moe', 'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt', '--checkpoint', '/root/gpufree-data/MoE-runs/wide_deep_v4_chunk4/model_1999.pt', '--navigation', '--teacher-map', '--course', 'wide_deep', '--kind', 'ame_alternate', '--difficulty', '0', '--diagnostic', '--check-map', '--video', '--output', 'outputs/moe_perceptive/v4_plateau_diagnostic/ame_alternate']]
for i,c in enumerate(commands):
 r=subprocess.run(c)
 Path('outputs/moe_perceptive/v4_plateau_diagnostic/status.json').write_text(json.dumps({"completed_case":i,"returncode":r.returncode}))
 if r.returncode:break
