import subprocess,sys,json
from pathlib import Path
p=Path(__file__).parent
reports=[]
for seed,yaw in [(12700,0),(12701,-15),(12702,15),(12703,30)]:
 out=p/f"seed_{seed}"
 cmd=[sys.executable,"-m","skills.backflip.simulate","--source","/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip","--onnx","outputs/fixstand_backflip_upstream_actor_v7_20261022/exported/policy.onnx","--physics-hz","1000","--seconds","5","--fixstand-roundtrip","--diagnostics","--seed",str(seed),"--initial-yaw-deg",str(yaw),"--output",str(out)]
 subprocess.run(cmd,check=True)
 d=json.loads((out/"summary.json").read_text());reports.append(dict(seed=seed,yaw=yaw,success=d["successes"]==d["total"],constraints=d["hardware_constraints_pass"]))
 (p/"evaluation.json").write_text(json.dumps(reports,indent=2))
