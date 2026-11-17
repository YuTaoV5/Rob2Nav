#!/usr/bin/env python
"""FS stair line: standalone flat evaluation (low-speed approach + stop).

This evaluates the *stair* policy on flat ground only. It is deliberately not a
full flat-driving evaluation: forward/reverse/turn/arc tracking, braking
distance and the 10 s dwell belong to the rolling policy.

Two cases per seed, both on the ``flat_approach`` stage geometry:

* ``approach`` -- the policy drives from the back-off start towards the
  approach line. Measured: forward progress, mean body speed, lateral drift,
  upright, non-wheel contact.
* ``stop`` -- the natural stop at the approach line: the last
  ``stop_window_s`` of the episode. Measured: drift, mean speed, upright.
  The stop is *not* imposed by zeroing the wheels; it is wherever the policy
  settles, which is what the curriculum rewards.

Dry-run is the default; ``--run`` is required to execute episodes, and
``--checkpoint`` is required for a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# fs_stair_train inserts the pinned r152 runtime into sys.path and imports the
# environment module; import it first so fs_stair_env can resolve mujoco_screen.
import fs_stair_train as train  # noqa: E402
import fs_stair_env as fse  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=tuple(train.ARCHES), default="6frame")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="SB3 .zip or torch state dict")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-pass", action="store_true",
                        help="exit non-zero when a gate fails")
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser


def load_config(args) -> dict:
    path = args.config or (HERE / "configs" / f"fs_stair_{args.arch}.json")
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    cfg = json.loads(path.read_text())
    if cfg["arch"] != args.arch:
        raise SystemExit(f"config arch {cfg['arch']} != --arch {args.arch}")
    args.config = path
    return cfg


def load_policy(path: Path, cfg: dict, arch: str):
    if path.suffix == ".zip":
        from stable_baselines3 import PPO
        model = PPO.load(str(path), device="cpu")
        return model, model.policy
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model = train.make_policy(cfg, arch, cfg["seed"])
    model.policy.load_state_dict(state)
    return model, model.policy


def episode(cfg, arch: str, policy, seed: int, flat_eval: dict) -> dict:
    env = train.make_env(cfg, arch, stage=train.stage_ladder(cfg)[0], trace=True)
    dt = float(cfg["control_period_s"])
    steps = int(round(flat_eval["episode_seconds"] / dt))
    window = int(round(flat_eval["stop_window_s"] / dt))
    approach_steps = int(round(flat_eval["approach_seconds"] / dt))
    obs, _ = env.reset(seed=int(seed), options={"scene_seed": int(seed)})
    terminated = truncated = False
    with torch.inference_mode():
        for _ in range(steps):
            action, _ = model_predict(policy, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    base = env.unwrapped
    rows = base.rows
    if not rows:
        raise RuntimeError("empty trace")
    xs = np.array([row["x"] for row in rows])
    ys = np.array([row["y"] for row in rows])
    vx = np.array([row["vx_body"] for row in rows])
    upright = np.array([row["upright"] for row in rows])
    impact = np.array([row["impact_n"] for row in rows])
    head = slice(0, min(approach_steps, len(rows)))
    tail = rows[-window:]
    tail_x = np.array([row["x"] for row in tail])
    tail_vx = np.array([row["vx_body"] for row in tail])
    tail_y = np.array([row["y"] for row in tail])
    line = base.stage.approach_line_m
    record = {
        "seed": int(seed), "arch": arch, "stage": base.stage.name,
        "steps": len(rows), "terminated": bool(terminated), "truncated": bool(truncated),
        "progress_m": float(xs.max() - xs[0]),
        "mean_approach_speed_m_s": float(np.abs(vx[head]).mean()) if len(vx[head]) else 0.,
        "max_lateral_m": float(np.abs(ys).max()),
        "min_upright": float(upright.min()),
        "peak_nonwheel_impact_n": float(impact.max()),
        "reached_line": bool(line is not None and xs.max() >= line),
        "stop_drift_m": float(np.abs(tail_x - tail_x[0]).max()) if len(tail_x) else 0.,
        "stop_speed_m_s": float(np.abs(tail_vx).mean()) if len(tail_vx) else 0.,
        "stop_lateral_m": float(np.abs(tail_y - tail_y[0]).max()) if len(tail_y) else 0.,
        "final_x_m": float(xs[-1]), "final_y_m": float(ys[-1]),
        "failure_kinds": base.failure_kinds(),
    }
    gates = flat_eval["gates"]
    checks = {
        "progress": record["progress_m"] >= gates["min_forward_progress_m"],
        "speed_range": (gates["min_mean_approach_speed_m_s"]
                        <= record["mean_approach_speed_m_s"]
                        <= gates["max_mean_approach_speed_m_s"]),
        "lateral": record["max_lateral_m"] <= gates["max_lateral_abs_y_m"],
        "upright": record["min_upright"] >= gates["upright_ge"],
        "nonwheel_contact": record["peak_nonwheel_impact_n"] < gates["nonwheel_impact_n_lt"],
        "peak_impact": record["peak_nonwheel_impact_n"] < gates["peak_impact_n_lt"],
        "no_fall": record["min_upright"] >= gates["fall_upright_lt"],
        "not_terminated": not record["terminated"],
        "reached_line": record["reached_line"],
        "stop_drift": record["stop_drift_m"] <= gates["stop_drift_m_le"],
        "stop_speed": record["stop_speed_m_s"] <= gates["stop_speed_m_s_le"],
    }
    record["checks"] = checks
    record["passed"] = bool(all(checks.values()))
    return record


def model_predict(policy, obs):
    tensor, _ = policy.predict(obs, deterministic=True)
    return tensor, None


def print_plan(cfg, args, flat_eval: dict) -> None:
    train.say("=" * 78)
    train.say("FS STAIR FLAT EVALUATION --", "RUN" if args.run else "DRY-RUN")
    train.say("=" * 78)
    train.say("[source] config", str(Path(args.config).resolve()),
              train.sha256_file(Path(args.config).resolve()))
    train.say("[source] LD_LIBRARY_PATH", train.check_ld_library_path(),
              "| torch threads", torch.get_num_threads(),
              "| enforced at", train.threads_location()["primary"])
    train.say("[assets]", train.ASSETS)
    train.say("[eval  ] stage", flat_eval["stage"], "| seeds", flat_eval["seeds"])
    train.say("[eval  ] episode", flat_eval["episode_seconds"], "s | approach window",
              flat_eval["approach_seconds"], "s | stop window", flat_eval["stop_window_s"], "s")
    for key, value in flat_eval["gates"].items():
        train.say(f"[gate  ] {key} = {value}")
    train.say("[eval  ] scope", flat_eval["note"])
    train.say("[eval  ] checkpoint", args.checkpoint)


def main(argv=None) -> int:
    torch.set_num_threads(1)
    args = parse_args().parse_args(argv)
    train.check_ld_library_path()
    cfg = load_config(args)
    flat_eval = cfg["flat_eval"]
    if args.seeds:
        flat_eval = dict(flat_eval, seeds=args.seeds)
    print_plan(cfg, args, flat_eval)
    if not args.run:
        train.say("\n未获启动授权（no --run）：只打印预检报告，退出 0。")
        train.say("AUTHORIZATION: evaluation NOT started")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "dryrun.log").write_text("\n".join(train._LOG) + "\n")
            (args.out / "inputs.sha256").write_text(
                f"{train.sha256_file(Path(args.config).resolve())}  "
                f"{Path(args.config).resolve()}\n"
                f"{train.sha256_file(HERE / 'fs_stair_eval_flat.py')}  "
                f"{HERE / 'fs_stair_eval_flat.py'}\n"
                f"{train.sha256_file(HERE / 'fs_stair_env.py')}  {HERE / 'fs_stair_env.py'}\n")
            (args.out / "cmd.sh").write_text(
                "#!/bin/bash\n" + " ".join(
                    train.shlex.quote(part) for part in
                    [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n")
            (args.out / "selfcheck.json").write_text(json.dumps({
                "status": "not_applicable",
                "reason": "the flat evaluation shares fs_stair_env.py with the training "
                          "entry; its PASS/FAIL gate list is fs_stair_train.py --selfcheck",
            }, indent=2) + "\n")
        return 0
    if args.checkpoint is None:
        raise SystemExit("--run requires --checkpoint")
    if args.out is None:
        raise SystemExit("--run requires --out")
    model, policy = load_policy(args.checkpoint, cfg, args.arch)
    rows = [episode(cfg, args.arch, model, seed, flat_eval) for seed in flat_eval["seeds"]]
    passed = sum(row["passed"] for row in rows)
    summary = {
        "arch": args.arch, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": train.sha256_file(args.checkpoint),
        "seeds": flat_eval["seeds"], "passed": passed, "total": len(rows),
        "gates": flat_eval["gates"], "scope": flat_eval["note"],
        "per_gate_pass": {key: sum(row["checks"][key] for row in rows)
                          for key in rows[0]["checks"]},
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "flat_eval.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "inputs.sha256").write_text(
        f"{train.sha256_file(args.checkpoint)}  {args.checkpoint}\n"
        f"{train.sha256_file(Path(args.config).resolve())}  {Path(args.config).resolve()}\n"
        f"{train.sha256_file(HERE / 'fs_stair_eval_flat.py')}  {HERE / 'fs_stair_eval_flat.py'}\n")
    for row in rows:
        train.say(f"[seed {row['seed']}] passed={row['passed']} progress={row['progress_m']:.3f} "
                  f"speed={row['mean_approach_speed_m_s']:.3f} lat={row['max_lateral_m']:.3f} "
                  f"stop_drift={row['stop_drift_m']:.3f} stop_speed={row['stop_speed_m_s']:.3f}")
    train.say("FLAT_EVAL", json.dumps({"passed": passed, "total": len(rows),
                                       "per_gate_pass": summary["per_gate_pass"]}))
    return 0 if (passed == len(rows) or not args.require_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
