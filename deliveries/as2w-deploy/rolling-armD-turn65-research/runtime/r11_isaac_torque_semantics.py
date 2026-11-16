#!/usr/bin/env python3
"""What does Isaac's torque readback actually mean?

Operator step 2.  The articulation exposes several torque-ish fields and we must not guess which is which
before using any of them to judge saturation:

    applied_torque        computed_torque        joint_effort_target        joint_effort_limits

Method: with the base pinned and the wheels suspended, command a KNOWN constant effort directly on one
wheel and read all of them back.  Three cases make the semantics unambiguous:

    command 1.0 N*m   -> whichever field reads 1.0 is the applied/commanded quantity
    command 5.0 N*m   -> same check at a second magnitude, so a fixed scale factor cannot fool us
    command 20.0 N*m  -> above the 13.5 N*m cap, so limit clamping shows up in the fields that represent
                         real applied effort and does NOT show up in the ones that merely echo the request

If applied_torque tracks 1.0 and 5.0 and clamps at 13.5 for 20.0, it is the solver-side applied effort and
may be used for the saturation judgement.  If instead it stays zero while joint_effort_target echoes the
command, then applied_torque is not usable here and the saturation fraction must stay not_measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

CASES = [1.0, 5.0, 20.0]
CAP = 13.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        from as2w_isaac_robust_env import RobustEnvironment, robust_config
        from as2w_training_safety import apply_safety

        cfg = robust_config(SimpleNamespace(**vars(args), profile="pose-delay", direction_focus=True,
                                             command_stage="all", zero_memory=True, zero_focus=True), physics)
        cfg.scene.robot.actuators["foot"].damping = 6.0
        apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
        env = RobustEnvironment(cfg)
        try:
            robot = env.scene["robot"]
            names = list(robot.joint_names)
            wid = names.index("FL_foot")
            env.seed(args.seed)
            env.reset()
            root_pose = robot.data.root_pose_w.torch[0].clone()
            root_pose[2] = 2.0
            robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))
            robot.write_data_to_sim()
            env.sim.step()
            robot.update(0.005)

            def read():
                out = {}
                for f in ("applied_torque", "computed_torque", "joint_effort_target", "joint_effort_limits"):
                    try:
                        t = getattr(robot.data, f).torch[0][wid]
                        out[f] = round(float(t), 4)
                    except Exception as exc:
                        out[f] = "ERR:%r" % (exc,)
                return out

            rows = []
            for cmd in CASES:
                pin_pose = robot.data.root_pose_w.torch[0].clone()
                pin_pose[2] = 2.0
                robot.write_root_pose_to_sim(pin_pose.unsqueeze(0))
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))
                robot.set_joint_effort_target_index(
                    target=torch.full((1, 1), float(cmd), dtype=torch.float32, device=env.device), joint_ids=[wid])
                robot.write_data_to_sim()
                for _ in range(4):
                    env.sim.step()
                robot.update(0.005)
                r = {"commanded_Nm": cmd}
                r.update(read())
                rows.append(r)
                print("cmd=%5.1f  ->  %s" % (cmd, {k: v for k, v in r.items() if k != "commanded_Nm"}), flush=True)

            # which field behaves like a real applied effort?
            verdict = {}
            for f in ("applied_torque", "computed_torque", "joint_effort_target"):
                vals = []
                for r in rows:
                    v = r.get(f)
                    vals.append(v if isinstance(v, (int, float)) else None)
                if any(v is None for v in vals):
                    verdict[f] = "unreadable"
                    continue
                tracks = abs(vals[0] - CASES[0]) < 0.2 and abs(vals[1] - CASES[1]) < 0.2
                clamps = vals[2] <= CAP + 0.2
                if tracks and clamps:
                    verdict[f] = "APPLIED (tracks low commands and clamps at %.1f)" % CAP
                elif tracks:
                    verdict[f] = "echoes the request without clamping"
                else:
                    verdict[f] = "neither tracks nor clamps (values %s)" % vals
            print()
            for f, v in verdict.items():
                print("  %-22s %s" % (f, v))

            rep = {"cap_Nm": CAP, "rows": rows, "verdict": verdict,
                   "saturation_fraction_status": "not_measured until a field is confirmed APPLIED"}
            (args.out / "torque_semantics.json").write_text(json.dumps(rep, indent=1))
            print("TORQUE_SEMANTICS_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
