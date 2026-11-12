"""Audit shared contract, ONNX identity and reported gates in a rolling evidence tar."""
import argparse
import hashlib
import json
import tarfile
from contextlib import nullcontext
from pathlib import Path

CASES = {"forward", "back", "turn_left", "turn_right", "arc_left", "arc_right"}
SHARED = (
    "checkpoint_sha256", "rolling_fixed_exit", "rolling_residual", "mirror_rolling",
    "rolling_stand_compensation", "rolling_wheel_hold", "park_precalibrated",
    "rolling_loaded_stance", "park_odometry", "park_common_gain_scale",
    "park_common_speed_damping", "park_settle_joints", "park_continuous_bias",
    "park_straight_bias", "wrapped_leg_slew_rad_s", "rolling_leg_filter_hz",
    "pure_turn_common_scale",
)


def audit(archive, replay=None):
    root = Path(archive)
    source = nullcontext(None) if root.is_dir() else tarfile.open(archive, 'r:*')
    with source as package:
        def read(path):
            if package is None:
                if path == 'onnx/policy.onnx':
                    return (root / 'weights/policy.onnx').read_bytes()
                if path == 'onnx/onnx-parity.json':
                    return (root / 'evidence/onnx-parity.json').read_bytes()
                return (root / 'evidence' / path).read_bytes()
            with package.extractfile(path) as stream:
                return stream.read()

        mujoco = json.loads(read("mujoco/mujoco-contract.json"))
        isaac = json.loads(read("isaac/contract.json"))
        mujoco_result = json.loads(read("mujoco/mujoco-rolling-acceptance.json"))
        isaac_result = json.loads(read("isaac/speed-evaluation.json"))
        parity = json.loads(read("onnx/onnx-parity.json"))
        actual_onnx = hashlib.sha256(read("onnx/policy.onnx")).hexdigest()

    mismatches = {}
    for key in SHARED:
        if key not in mujoco or key not in isaac or mujoco[key] != isaac[key]:
            mismatches[key] = {"mujoco": mujoco.get(key), "isaac": isaac.get(key)}
    for left, right in (("speed", "speed_cap_m_s"), ("turn_yaw_slew", "turn_yaw_slew_rad_s2"),
                        ("park_policy_rad", "park_policy_rad")):
        if left == "park_policy_rad" and left not in mujoco and right not in isaac:
            continue
        if left not in mujoco or right not in isaac or mujoco[left] != isaac[right]:
            mismatches[left] = {"mujoco": mujoco.get(left), "isaac": isaac.get(right)}

    identical_model = (mujoco.get("checkpoint_sha256") is not None and
                       mujoco.get("checkpoint_sha256") == isaac.get("checkpoint_sha256") ==
                       parity.get("checkpoint_sha256"))
    onnx_integrity = actual_onnx == parity.get("onnx_sha256")
    onnx_inference = (parity.get("test_input") == "recorded_observations" and
                      parity.get("frames", 0) > 0 and parity.get("max_action_error", 1) <= 1e-4)
    missing = {
        "mujoco": sorted(CASES - mujoco_result["passed"].keys()),
        "isaac": sorted(CASES - (isaac_result["summary"].keys() | isaac_result["turns"].keys())),
    }
    report = {
        "matching_checkpoint": identical_model,
        "shared_contract_mismatches": mismatches,
        "onnx_integrity": onnx_integrity,
        "onnx_inference_parity": onnx_inference,
        "missing_cases": missing,
        "mujoco": {k: mujoco_result[k] for k in ("gate", "safety_gate", "passed")},
        "isaac": {k: isaac_result[k] for k in ("gate", "safety_gate")},
        "not_covered": ["source equivalence", "full joystick timing", "hardware mapping", "ARM64 shadow"],
    }
    if replay is not None:
        replay_report = json.loads(Path(replay).read_text(encoding="utf-8"))
        replay_inputs = replay_report.get("input_sha256", {})
        replay_onnx = replay_report.get("onnx_hardware_replay", {})
        report["hardware_log_replay"] = {
            "matching_checkpoint": identical_model and (
                replay_inputs.get("candidate_checkpoint") == mujoco["checkpoint_sha256"]
            ),
            "input_sha256": replay_inputs,
            "moving_frames": replay_report.get("moving_frames"),
            "matching_onnx": onnx_integrity and replay_onnx.get("onnx_sha256") == actual_onnx,
            "onnx_max_action_error": replay_onnx.get("max_action_error"),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--replay", help="Hardware log replay report for the same candidate checkpoint")
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.replay), indent=2))
