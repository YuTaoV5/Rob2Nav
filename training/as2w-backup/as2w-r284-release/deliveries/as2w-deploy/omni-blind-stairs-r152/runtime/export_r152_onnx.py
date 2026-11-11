"""Export R152 and check sequential ONNX parity against its Python runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

import evaluate_omni_stairs as evaluation
from evaluate_omni_joint import make_controller
from r152_onnx_model import OBS_SIZE, STATE_SIZE, R152Step
from search_lateral_omni_teacher import course
from train_omni_lateral_history import screen
from train_omni_stop_transition import load


class OnnxActor:
    def __init__(self, session, defaults):
        self.session = session
        self.defaults = defaults
        self.state = np.zeros((1, STATE_SIZE), dtype=np.float32)

    def observation(self, obs, command):
        return np.r_[command, obs[3:9], obs[10:26] + self.defaults, obs[26:42]].astype(np.float32)

    def step(self, observation):
        actions, self.state = self.session.run(
            ["actions", "next_state"],
            {"obs": observation.reshape(1, OBS_SIZE), "state": self.state},
        )
        return actions[0]

    def plan(self, model, data, obs, command, tick, advance):
        return self.step(self.observation(obs, command))


def session_for(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def export(root, checkpoint, path):
    policy = torch.load(checkpoint, weights_only=True, map_location="cpu")
    contract, _, geometry = course(root)
    graph = R152Step(policy, contract, geometry).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            graph, (torch.zeros(1, OBS_SIZE), torch.zeros(1, STATE_SIZE)), str(path),
            input_names=["obs", "state"], output_names=["actions", "next_state"],
            opset_version=17, dynamo=False,
        )
    # Older ORT builds shipped with the upstream Orin target use ONNX IR 8.
    model = onnx.load(path)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return session_for(path)


def write_deploy_yaml(root, path, onnx_path):
    import os

    contract, limits, _ = course(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "policy_file": os.path.relpath(onnx_path.resolve(), path.resolve().parent),
        "step_dt": contract["control_dt"],
        "joint_names": contract["joint_names"],
        "default_joint_pos": contract["default_joint_pos"],
        "position_lower": [*limits[:12, 0].tolist(), 0., 0., 0., 0.],
        "position_upper": [*limits[:12, 1].tolist(), 0., 0., 0., 0.],
        "velocity_limit": limits[:, 2].tolist(),
        "command_limit": [.6, .35, .8],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def audit(root, checkpoint, session, output, full):
    policy, contract, limits, base, controller = load(root, checkpoint)
    defaults = np.asarray(contract["default_joint_pos"])
    measured = {"max_abs_action_error": 0., "frames": 0, "per_case": {}}
    fixture = []

    class Paired:
        def __init__(self, route, label, record=False):
            self.reference = make_controller(policy, contract, controller, route)
            self.onnx = OnnxActor(session, defaults)
            self.label = label
            self.record = record

        def plan(self, model, data, obs, command, tick, advance):
            expected = self.reference.plan(model, data, obs, command, tick, advance)
            values = self.onnx.observation(obs, command)
            actual = self.onnx.step(values)
            diff = float(np.max(np.abs(expected - actual)))
            measured["max_abs_action_error"] = max(measured["max_abs_action_error"], diff)
            measured["frames"] += 1
            measured["per_case"][self.label] = max(measured["per_case"].get(self.label, 0.), diff)
            if self.record:
                fixture.append([self.label, *values.tolist(), *actual.tolist()])
            return expected

    # Dynamic sign changes, one straight segment and two different kinds of stops.
    commands = np.array(((.35, 0., 0.), (-.35, -.26, -.03), (-.29, .25, .10), (0., 0., 0.)))
    actor = Paired(base(), 1, record=True)

    def transition(model, data, obs, unused, tick, advance):
        command = commands[min(tick // 70, len(commands) - 1)]
        frame = obs.copy()
        frame[:3] = command - (.4, 0., 0.)
        return actor.plan(model, data, frame, command, tick, advance)

    evaluation.flat_episode(None, contract, limits, commands[0], 81201, evaluation.gait_clock, transition)
    stair_actor = Paired(base(), 2, record=True)
    sim = evaluation.sim
    model = sim.load_flat_model(True, True, "stairs", 27001)
    stress = sim.sample_dynamics_stress(27001, model.nbody - 1, 16)
    dt = contract["control_dt"]

    def stair(data, obs, elapsed, advance):
        action = stair_actor.plan(model, data, obs, np.array((.4, 0., 0.)), round(elapsed / dt), advance)
        legs = np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T)
        wheels = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
        return action, legs, wheels

    sim.evaluate(model, contract, policy["anchor"], 27001, 0., limits=limits, action_clip=6.,
                 wheel_scale=16., wheel_angle_period=4. * np.pi, dynamics_stress=stress,
                 terrain="stairs", observation_suffix=evaluation.gait_clock, native_planner=stair)
    if full:
        counter = iter(range(3, 300))
        for confirmation in (False, True):
            result = screen(base, policy["anchor"], None, np.asarray(policy["parameters"]), contract,
                            limits, lambda route: Paired(route, next(counter)), confirmation)
            name = "confirmation" if confirmation else "fixed"
            (output / f"{name}-python-reference.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item()))
            onnx_result = screen(base, policy["anchor"], None, np.asarray(policy["parameters"]), contract,
                                 limits, lambda route: OnnxActor(session, defaults), confirmation)
            (output / f"{name}-onnx.json").write_text(json.dumps(onnx_result, indent=2, default=lambda x: x.item()))
    np.savetxt(output / "r152-trace.csv", np.asarray(fixture), fmt="%.9g", delimiter=",")
    (output / "parity.json").write_text(json.dumps(measured, indent=2))
    print(json.dumps({k: v for k, v in measured.items() if k != "per_case"}), flush=True)
    if measured["max_abs_action_error"] > .001:
        raise RuntimeError("ONNX differs from R152 by more than 1e-3 on the Python rollout inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--deploy-yaml", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.evidence.mkdir(parents=True, exist_ok=True)
    runtime = export(args.assets.resolve(), args.checkpoint, args.onnx)
    audit(args.assets.resolve(), args.checkpoint, runtime, args.evidence, args.full)
    if args.deploy_yaml:
        write_deploy_yaml(args.assets.resolve(), args.deploy_yaml, args.onnx)


if __name__ == "__main__":
    main()
