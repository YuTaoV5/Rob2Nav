"""Sequential benchmark runner. All commands and exit codes are recorded."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["native", "mujoco"], required=True)
p.add_argument("--python", default=sys.executable)
p.add_argument("--checkpoint")
p.add_argument("--actor")
p.add_argument("--io")
p.add_argument("--backend", choices=["moe", "ik", "learned"], default="learned")
p.add_argument("--blind", action="store_true")
p.add_argument("--fused", action="store_true")
p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--seed", type=int, help="Explicit replay seed; omitted uses each evaluator's default")
p.add_argument("--video", action="store_true")
p.add_argument("--difficulties", nargs="+", type=float, default=[.25, .5, .75])
p.add_argument("--kinds", nargs="+", default=["stairs", "platform", "stones"])
p.add_argument("--output", required=True)
args = p.parse_args()
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
commands = []
for difficulty in args.difficulties:
    for kind in args.kinds:
        dest = out/f"{kind}_{difficulty:.2f}"
        if args.mode == "native":
            if not args.checkpoint: p.error("native requires --checkpoint")
            cmd = [args.python, "-m", "perceptive_rl.evaluate", "--headless", "--checkpoint", args.checkpoint,
                   "--num-envs", str(args.num_envs)]
            if args.blind: cmd += ["--blind"]
            if args.fused: cmd += ["--fused"]
        else:
            cmd = [args.python, "-m", "perceptive_rl.mujoco_compare", "--backend", args.backend,
                   "--episodes", str(args.episodes)]
            if args.actor: cmd += ["--actor", args.actor]
            if args.io: cmd += ["--io", args.io]
        cmd += ["--kind", kind, "--difficulty", str(difficulty), "--output", str(dest)]
        if args.seed is not None: cmd += ["--seed", str(args.seed)]
        if args.video: cmd += ["--video"]
        with (out/f"{kind}_{difficulty:.2f}.log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        record = {"command": cmd, "returncode": proc.returncode}
        if (dest/"summary.json").exists():
            result = json.loads((dest/"summary.json").read_text())
            record["successes"] = result["successes"]
        commands.append(record)
        (out/"commands.json").write_text(json.dumps(commands, indent=2))
        print(json.dumps(record), flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)
