"""Export r284 and verify ONNX against the same PyTorch actor in flat rollouts."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from as2w_zero_memory import ZeroWheelMemory
from eval_as2w_static_memory import episode as static_episode
from eval_as2w_strict_hold import ALL_FLAT_COMMANDS, episode as dynamic_episode
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["iter"] != 500:
        parser.error("Expected r284 checkpoint index 500")
    actor = make_actor(state["model_state_dict"]).eval()
    torch_actor = torch.nn.Sequential(actor, torch.nn.Hardtanh(-6., 6.))
    destination = args.output / "unitree_cpp_deploy/logs/as2w/r284-500"
    output = destination / "exported/policy.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(torch_actor, torch.zeros(1, 65), str(output), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    graph = onnx.load(output)
    graph.ir_version = 8
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    ref = numpy_actor(actor)
    evidence = args.output / "evidence"
    evidence.mkdir(exist_ok=True)

    class Replay:
        def __init__(self, stream):
            self.memory = ZeroWheelMemory()
            self.stream = stream
            self.frames = 0
            self.maximum = 0.

        def __call__(self, obs):
            features = self.memory.push(obs)
            action = session.run(["actions"], {"obs": features[None, :]})[0][0]
            expected = ref(features)
            self.maximum = max(self.maximum, float(np.max(np.abs(expected - action))))
            np.savetxt(self.stream, np.r_[self.frames == 0, obs, expected][None, :], fmt="%.9g")
            self.frames += 1
            return action

    env = OmniEnvironment(args.assets, 1, stop_training=True)
    env.apply_motion_prior = False
    report, parity = {}, []
    with gzip.open(evidence / "action-replay.txt.gz", "wt") as stream:
        for split, first_seed in (("standard", 27001), ("heldout", 37001)):
            report[split] = {"static": [], "dynamic": []}
            for seed in range(first_seed, first_seed + 4):
                for name in ("static", "dynamic"):
                    replay = Replay(stream)
                    if name == "static":
                        rows = [static_episode(env, replay, False, seed)]
                    else:
                        rows = dynamic_episode(env, replay, seed, False, commands=ALL_FLAT_COMMANDS,
                                               strict_hold=True)
                    report[split][name].extend(rows)
                    parity.append(dict(split=split, seed=seed, scenario=name, frames=replay.frames,
                                       max_action_error=replay.maximum))
    if max(r["max_action_error"] for r in parity) > 1e-4:
        raise RuntimeError("ONNX exceeds PyTorch action error tolerance")
    (evidence / "onnx-flat.json").write_text(json.dumps(report, indent=2) + "\n")
    (evidence / "onnx-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    params = destination / "params"
    params.mkdir(exist_ok=True)
    control = dict(
        status="simulation-preview", model="../exported/policy.onnx", control_dt=env.dt,
        input_name="obs", input_shape=[1, 65], output_name="actions", output_shape=[1, 16],
        joint_names=env.contract["joint_names"], default_joint_pos=env.defaults.tolist(),
        leg_lower_rad=env.leg_low.tolist(), leg_upper_rad=env.leg_high.tolist(),
        wheel_speed_limit_rad_s=env.wheel_limit.tolist(), action_scale=[.35] * 12 + [16.] * 4,
        action_clip=[-6., 6.], kp=env.kp.tolist(), kd=env.kd.tolist(),
        filtered_command_alpha=.12, filtered_command_zero_threshold=.005,
        planar_limit_m_s=.10, yaw_limit_rad_s=.30, wheel_reference_scale_rad=.20,
        wheel_reference_clip=[-3., 3.], wheel_period_rad="4*pi", gait_period_s=.62,
        measured_base_linear_velocity=False,
    )
    (params / "deploy.yaml").write_text(json.dumps(control, indent=2) + "\n")
    print(f"frames={sum(r['frames'] for r in parity)} resets={len(parity)} "
          f"max_action_error={max(r['max_action_error'] for r in parity):.9g}")


if __name__ == "__main__":
    main()
