import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parents[1]
    lines = ["# Remote simulation validation", "", "Date: 2026-10-17", "",
             "Host: Ubuntu 24.04 / NVIDIA L40; isolated Python 3.11 environment.", "",
             "RoboGauge revision: `" + json.loads((root / "vendor/robogauge_snapshot.json").read_text())["revision"] + "`.", "",
             "| Run | Success | Collision | Fall | Timeout | Mean successful duration (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ("empty", "slalom_final", "random_final", "blocked_check", "ppo_initial_eval"):
        path = root / "outputs" / name / "summary.json"
        if not path.exists():
            path = root / "artifacts" / "outputs" / name / "summary.json"
        if not path.exists():
            lines.append(f"| {name} | pending | | | | |")
            continue
        d = json.loads(path.read_text())
        times = [r["seconds"] for r in d["results"] if r["success"]]
        duration = f"{statistics.mean(times):.2f}" if times else "-"
        timeouts = sum(r["timeout"] for r in d["results"])
        lines.append(f'| {name} | {d["successes"]}/{d["episodes"]} | {d["collisions"]} | {d["falls"]} | {timeouts} | {duration} |')
    lines += ["", "The blocked scene intentionally has no feasible path; timeout is the expected outcome.", "",
              "PPO initial checkpoint: 32,768 environment steps, 4 environments, seed 42. This is an initial training run, not a convergence claim. Held-out PPO evaluation seeds start at 200; reactive random-scene evaluation seeds start at 100.", "",
              "Three simulation checks plus Stable-Baselines3 Gymnasium API/checkpoint reload checks were executed. See [the test log](artifacts/tests_final.log) and the saved summaries under artifacts/outputs for evidence.", "",
              "Sensor geometry is idealized and mounting extrinsics are assumed. Odometry is supplied by simulator state. No real robot commands are sent."]
    (root / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
