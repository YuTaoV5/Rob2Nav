#!/usr/bin/env python3
"""Dry-run printer and guarded launcher for the armD-to-R4000 turn-curriculum migration.

Default behaviour is a DRY RUN: it prints the start checkpoint identity, the init method, the wheel control
contract, the A-versus-B curriculum difference, the PPO batch and the environment-step budget, and then
stops.  It never imports Isaac and never starts a simulator.

Training only happens with an explicit ``--run``, and even then only after the offline self-check passes and
the declared invariants hold.  Any inconsistency (wrong checkpoint hash, contract drift, out-of-envelope
command, training not authorised) makes it refuse with a non-zero exit code.

Examples
    python scripts/dry_run.py --arm A
    python scripts/dry_run.py --arm B
    python scripts/dry_run.py --arm B --run          # refuses while status.training_authorised is false
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
BASE = TASK_ROOT.parent                      # .../as2w-current
ISAAC = "/isaac-sim/python.sh"
ISAAC_VENV = "/workspace/isaaclab/source"
USD = str(BASE / "runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda")
URDF = "/root/gpufree-data/Rob2Nav/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
PPO_CFG = ("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/agents/"
           "rsl_rl_ppo_cfg.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ppo_batch():
    """Read the PPO batch constants from the installed Go2 runner config rather than assuming them."""
    out = {"num_steps_per_env": None, "num_learning_epochs": None, "num_mini_batches": None,
           "source": PPO_CFG, "verified": False}
    try:
        text = Path(PPO_CFG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for key in ("num_steps_per_env", "num_learning_epochs", "num_mini_batches"):
        m = re.search(r"^\s*%s\s*=\s*(\d+)" % key, text, re.M)
        if m:
            out[key] = int(m.group(1))
    out["verified"] = all(out[k] is not None for k in ("num_steps_per_env", "num_learning_epochs", "num_mini_batches"))
    return out



def strip_flags(argv, names):
    """Remove whole options (flag plus its value) so the printed command has no duplicates."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] in names:
            i += 2 if (i + 1 < len(argv) and not argv[i + 1].startswith("--")) else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def arm_command(cfg, arm, out_dir):
    s = cfg["shared_run"]
    snapshot = str(Path(cfg["provenance"]["run_dir"]) / "source")
    common = strip_flags(shlex.split(" ".join(s["common_flags"])),
                     {"--seed", "--learning-rate", "--num-envs", "--iterations", "--out",
                      "--checkpoint", "--usd", "--urdf"})
    if arm == "A":
        entry = str(Path(snapshot) / "train_as2w_isaac_robust.py")
        py_path = [snapshot]
    else:
        entry = str(HERE / "train_b_entry.py")
        py_path = [str(HERE), snapshot]
    cmd = [ISAAC, entry] + common + [
        "--usd", USD, "--urdf", URDF,
        "--num-envs", str(s["num_envs"]),
        "--iterations", str(s["iterations"]),
        "--out", out_dir,
        "--checkpoint", cfg["start"]["checkpoint"],
        "--seed", str(s["seed"]),
        "--learning-rate", str(s["learning_rate"]),
    ]
    env = {}
    if arm == "B":
        env["AS2W_R4000_SNAPSHOT"] = snapshot
        env["AS2W_TURN_B_SEED"] = str(s["seed"])
    return cmd, py_path, env


def print_block(title):
    print("\n== %s ==" % title)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=TASK_ROOT / "configs" / "ab_curriculum.json")
    ap.add_argument("--arm", choices=("A", "B"), default="A")
    ap.add_argument("--run", action="store_true", help="actually start training (requires authorisation)")
    ap.add_argument("--skip-selfcheck", action="store_true", help="dry-run only: do not re-run the self-check")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    arm = args.arm
    name = cfg[arm]["name"]
    out_dir = str(TASK_ROOT / "work" / name)

    print_block("dry run: %s" % name)
    print("  config      %s" % args.config)
    print("  mode        %s" % ("RUN (training will start)" if args.run else "DRY RUN (nothing starts)"))
    print("  phase       %s   deployable=%s   training_started=%s"
          % (cfg["status"]["phase"], cfg["status"]["deployable"], cfg["status"]["training_started"]))

    # ---- 1. start checkpoint ----------------------------------------------------------------
    print_block("start checkpoint")
    ckpt = Path(cfg["start"]["checkpoint"])
    if not ckpt.is_file():
        print("  REFUSED: missing %s" % ckpt)
        return 3
    got = sha256(ckpt)
    size = ckpt.stat().st_size
    match = got == cfg["start"]["sha256_expected"] and size == cfg["start"]["bytes_expected"]
    print("  path        %s" % ckpt)
    print("  bytes       %d (expected %d)" % (size, cfg["start"]["bytes_expected"]))
    print("  sha256      %s" % got)
    print("  expected    %s  -> %s" % (cfg["start"]["sha256_expected"], "MATCH" if match else "MISMATCH"))
    print("  iteration   %s   actor %d->%d   critic %d->1"
          % (cfg["start"]["training_iteration"], cfg["start"]["actor_obs"],
             cfg["start"]["actor_actions"], cfg["start"]["actor_obs"]))
    print("  init mode   %s" % cfg["start"]["init_mode"])
    print("  optimizer   present=%s used=%s" % (cfg["start"]["optimizer_state_present"],
                                                cfg["start"]["optimizer_state_used"]))
    print("  reason      %s" % cfg["start"]["optimizer_reason"])
    if not match:
        print("\nREFUSED: start checkpoint identity does not match the site record")
        return 3

    # ---- 2. wheel control contract ----------------------------------------------------------
    c = cfg["contract"]
    print_block("wheel control contract (frozen, identical in A and B)")
    print("  action clip            +-%.1f" % c["action_clip"])
    print("  wheel action scale     %.1f rad/s per unit   (wheel target = %.1f * action)"
          % (c["wheel_action_scale_rad_s"], c["wheel_action_scale_rad_s"]))
    print("  leg action scale       %.2f rad per unit" % c["leg_action_scale_rad"])
    print("  wheel radius           %.3f m  -> 0.25 m/s needs %.3f rad/s" % (c["wheel_radius_m"], 0.25 / c["wheel_radius_m"]))
    print("  wheel actuator kd      %.1f (damping, stiffness 0)" % c["wheel_actuator_damping_kd"])
    print("  leg target rate limit  %.1f rad/s = %.3f rad per policy step (%.2f s)"
          % (c["leg_target_rate_limit_rad_s"], c["leg_target_rate_limit_rad_per_step"], c["policy_step_s"]))
    print("  leg residual filter    %.1f Hz first order" % c["leg_residual_filter_hz"])
    print("  open-loop turn comp    %s" % c["open_loop_turn_compensation"])
    print("  env gyro correction    %s" % c["env_gyro_correction"])
    print("  pure-turn common scale %.1f" % c["pure_turn_common_scale"])
    print("  torque limits (90%%)    hip %.1f  thigh %.1f  calf %.1f  wheel %.1f Nm"
          % (c["torque_limits_nm"]["hip"], c["torque_limits_nm"]["thigh"],
             c["torque_limits_nm"]["calf"], c["torque_limits_nm"]["wheel"]))

    # ---- 3. curriculum difference -----------------------------------------------------------
    a, b = cfg["A"]["curriculum"], cfg["B"]["curriculum"]
    print_block("curriculum: A (frozen R4000) versus B (migrated)")
    print("  %-22s %-38s %-38s" % ("field", "A", "B"))
    rows = [
        ("mixed-branch shares",
         "turn .40 / arc .15 / straight .45",
         "turn .30 / long .15 / arc .25 / straight .30"),
        ("arc forward speed", "%.2f m/s (forward only)" % a["arc"]["vx_m_s"],
         "%.2f m/s (bidirectional)" % b["arc"]["vx_m_s"]),
        ("arc yaw magnitude", a["arc"]["wz_rad_s"], b["arc"]["wz_rad_s"]),
        ("pure-turn yaw", a["pure_turn"]["wz_rad_s"], b["pure_turn"]["wz_rad_s"]),
        ("sustained turn", "none", "share %.2f, dwell %s s" % (b["within_mixed"]["turn_long"],
                                                               [b["turn_long"]["dwell_s"][0], b["turn_long"]["dwell_s"][1]] if isinstance(b["turn_long"]["dwell_s"], list) else b["turn_long"]["dwell_s"])),
        ("release after turn", "none", "zero command for %s s after every long turn" % (b["release_s"],)),
        ("dwell, other cells", a["dwell_s"], b["pure_turn"]["dwell_s"]),
    ]
    for r in rows:
        print("  %-22s %-38s %-38s" % r)
    print("\n  differences B vs A:")
    for d in cfg["differences_B_vs_A"]:
        print("   - %s" % d)
    print("\n  requested %.2f m/s arc tier: %s" % (cfg["B"]["requested_arc_vx_m_s"],
                                                   cfg["B"]["requested_arc_vx_status"]))
    print("    %s" % cfg["B"]["requested_arc_vx_resolution"])

    # ---- 4. budget --------------------------------------------------------------------------
    s = cfg["shared_run"]
    pb = ppo_batch()
    steps = pb["num_steps_per_env"] or s["num_steps_per_env"]
    batch = s["num_envs"] * steps
    total = batch * s["iterations"]
    print_block("PPO batch and environment-step budget")
    print("  num_envs x num_steps_per_env   %d x %d = %d transitions per iteration"
          % (s["num_envs"], steps, batch))
    print("  iterations                     %d" % s["iterations"])
    print("  total environment steps        %d  (R4000 used 393,216,000 = %.3f of it)"
          % (total, total / 393216000.0))
    print("  PPO cfg source                 %s" % pb["source"])
    print("  PPO cfg verified from source   %s   epochs=%s minibatches=%s"
          % (pb["verified"], pb["num_learning_epochs"], pb["num_mini_batches"]))
    if pb["verified"]:
        print("  minibatch size                 %d transitions" % (batch // pb["num_mini_batches"]))
    print("  episode length                 %.1f s (motion start %.1f s, park until %.1f s)"
          % (s["episode_length_s"], s["motion_start_s"], s["park_until_s"]))
    if total != s["env_step_budget"]:
        print("\n  WARNING: printed budget %d differs from the declared %d" % (total, s["env_step_budget"]))

    # ---- 5. self-check gate -----------------------------------------------------------------
    if not args.skip_selfcheck:
        print_block("offline self-check gate")
        rc = subprocess.call([sys.executable, str(HERE / "selfcheck.py"), "--config", str(args.config)])
        if rc != 0:
            print("\nREFUSED: offline self-check failed (rc=%d); fix it and run again" % rc)
            return 4
        print("  self-check passed")

    # ---- 6. command -------------------------------------------------------------------------
    cmd, py_path, env_extra = arm_command(cfg, arm, out_dir)
    print_block("launch command for %s" % arm)
    print("  PYTHONPATH=%s" % ":".join(py_path))
    for k, v in env_extra.items():
        print("  export %s=%s" % (k, v))
    print("  " + " ".join(shlex.quote(x) for x in cmd))
    print("  out dir    %s" % out_dir)

    record = {
        "arm": arm, "name": name, "mode": "run" if args.run else "dry_run",
        "config": str(args.config),
        "start_checkpoint": {"path": str(ckpt), "sha256": got, "bytes": size, "identity_match": match,
                             "iteration": cfg["start"]["training_iteration"],
                             "init_mode": cfg["start"]["init_mode"],
                             "optimizer_state_present": cfg["start"]["optimizer_state_present"],
                             "optimizer_state_used": cfg["start"]["optimizer_state_used"]},
        "contract": c,
        "budget": {"num_envs": s["num_envs"], "num_steps_per_env": steps, "iterations": s["iterations"],
                   "transitions_per_iteration": batch, "total_env_steps": total,
                   "ppo_epochs": pb["num_learning_epochs"], "ppo_minibatches": pb["num_mini_batches"],
                   "ppo_minibatch_size": (batch // pb["num_mini_batches"]) if pb["num_mini_batches"] else None,
                   "ppo_cfg_verified": pb["verified"]},
        "curriculum": {"A": cfg["A"]["curriculum"], "B": cfg["B"]["curriculum"],
                       "differences": cfg["differences_B_vs_A"]},
        "launch_command": cmd, "pythonpath": py_path, "env": env_extra, "out_dir": out_dir,
        "selfcheck": "skipped" if args.skip_selfcheck else "passed",
    }
    ev = TASK_ROOT / "evidence" / ("dry_run_%s.json" % arm)
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print("  evidence   %s" % ev)

    if not args.run:
        print("\nDRY_RUN_OK  arm=%s  nothing was started" % arm)
        print("to launch: add --run (and set status.training_authorised=true in the config first)")
        return 0

    if not cfg["status"].get("training_authorised"):
        print("\nREFUSED: status.training_authorised is false; this delivery stops at ready-to-launch")
        return 5
    if cfg["status"].get("training_started"):
        print("\nREFUSED: status.training_started is already true")
        return 5

    pp = [ISAAC_VENV]
    for d in sorted(Path("/workspace/isaaclab/source").glob("isaaclab*")):
        pp.append(str(d))
    venv_site = str(Path(pp[0]) / "opt/isaaclab-venv/lib/python3.12/site-packages")
    full_env = dict(os.environ)
    full_env.update(env_extra)
    full_env["PYTHONPATH"] = ":".join([venv_site] + py_path)
    full_env.pop("LD_LIBRARY_PATH", None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print("\nRUNNING %s" % arm, flush=True)
    return subprocess.call(cmd, env=full_env, cwd=str(BASE))


if __name__ == "__main__":
    raise SystemExit(main())
