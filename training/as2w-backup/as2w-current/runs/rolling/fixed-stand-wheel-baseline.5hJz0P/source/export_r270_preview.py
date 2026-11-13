"""Export the exact r270 preview actor and record simulator observations for replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from as2w_history import History
from render_as2w_checkpoint import FLAT
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    if state["actor.0.weight"].shape[1] != History.size:
        parser.error("Expected a 720-feature history actor")
    actor = make_actor(state).eval()
    policy = torch.nn.Sequential(actor, torch.nn.Hardtanh(-6., 6.))
    destination = args.output / "unitree_cpp_deploy/logs/as2w/r270-122880"
    path = destination / "exported/policy.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(policy, torch.zeros(1, History.size), str(path), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    model = onnx.load(path)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    evaluator = ReferenceEvaluator(model)
    infer = numpy_actor(actor)
    replay, cases = [], []
    for stairs, seed in ((False, 27001), (True, 27004)):
        env = OmniEnvironment(args.assets, 0 if stairs else 1, stair_rise=.06,
                              stop_training=True, low_speed_course=stairs)
        obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10,
                                             "target": (0., 0., 0.)})
        env.apply_motion_prior = False
        env.next_command_tick = 10**9
        env.stair_hold_enabled = stairs
        history = History()
        maximum = 0.
        duration = env.stair_seconds if stairs else 2. * len(FLAT)
        for tick in range(round(duration / env.dt)):
            if not stairs:
                env.target_goal = np.array(FLAT[min(len(FLAT) - 1, int(tick * env.dt / 2.))])
            features = history.push(obs)
            action = infer(features)
            actual, = evaluator.run(None, {"obs": features[None, :]})
            maximum = max(maximum, float(np.max(np.abs(actual[0] - action))))
            replay.append(np.r_[tick == 0, obs, action])
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        cases.append(dict(terrain="stairs" if stairs else "flat", seed=seed,
                          frames=tick + 1, max_action_error=maximum,
                          wheel_levels=env.levels.tolist(), terminated=terminated))
    evidence = args.output / "evidence"
    evidence.mkdir(exist_ok=True)
    np.savetxt(evidence / "action-replay.txt.gz", replay, fmt="%.9g")
    (evidence / "onnx-reference.json").write_text(json.dumps(cases, indent=2) + "\n")
    if max(row["max_action_error"] for row in cases) > 1e-4:
        raise RuntimeError("ONNX action replay exceeded tolerance")
    parameters = destination / "params"
    parameters.mkdir(exist_ok=True)
    control = dict(
        status="simulation-preview",
        control_dt=env.dt,
        joint_names=env.contract["joint_names"],
        default_joint_pos=env.defaults.tolist(),
        leg_lower_rad=env.leg_low.tolist(),
        leg_upper_rad=env.leg_high.tolist(),
        wheel_speed_limit_rad_s=env.wheel_limit.tolist(),
        action_scale=[.35] * 12 + [16.] * 4,
        action_clip=[-6., 6.],
        history_offsets_ticks=list(History.offsets),
        history_slice=[3, 58],
        input_shape=[1, History.size],
        output_shape=[1, 16],
    )
    (parameters / "r270_contract.json").write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(cases))


if __name__ == "__main__":
    main()
