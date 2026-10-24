"""Complete remaining fixed-seed entry substep diagnostics, no training."""
import json
import subprocess
from pathlib import Path

base=Path("outputs/backflip_triple_20261024")
state=base/"entry_hold_independent_state.json"
results=[]
for seed,yaw in [(28051,90),(28052,180),(28053,-90)]:
    out=base/f"entry_stats_{seed}"
    if out.exists():raise FileExistsError(out)
    command=["/root/gpufree-data/envs/go2-mapless/bin/python","-m","rob2nav.deployment.verify_mujoco",
             "--asset","source","--sequence",str(base/"sequence.json"),"--seed",str(seed),
             "--initial-yaw-deg",str(yaw),"--max-seconds","150",
             "--experimental-compact-backflip-repeat","--output",str(out)]
    state.write_text(json.dumps(dict(active_seed=seed,results=results)))
    with (base/f"entry_stats_{seed}.log").open("w") as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    report=json.loads((out/"summary.json").read_text()) if (out/"summary.json").exists() else {}
    holds=[x.get("entry_hold_physics") for x in report.get("backflip_validation",[])]
    ok=completed.returncode==0 and report.get("passed",False) and len(holds)==3 and all(x and x["coverage_complete"] for x in holds)
    results.append(dict(seed=seed,returncode=completed.returncode,passed=ok,entry_holds=holds))
    state.write_text(json.dumps(dict(active_seed=None,results=results),indent=2))
    if not ok:raise RuntimeError(f"Verification failed for {seed}; inspect evidence before retry")
