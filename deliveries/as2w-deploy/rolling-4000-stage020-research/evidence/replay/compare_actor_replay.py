import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Replay AS2-W rolling actor on recorded observations.")
for field in ("archive", "baseline", "challenger", "report"):
    parser.add_argument(field, type=Path)
parser.add_argument("--onnx", type=Path, help="Evaluate the exact exported ONNX on the same frames.")
args = parser.parse_args()
archive, baseline, challenger, report = args.archive, args.baseline, args.challenger, args.report
with ZipFile(archive) as z:
    name = next(n for n in z.namelist() if n.endswith("/session_replay_cxx_output.log"))
    fixture = next(n for n in z.namelist() if n.endswith("/session_replay_fixture.txt"))
    commands = {int(values[0]): list(map(float, values[-3:])) for line in z.read(fixture).decode().splitlines() if (values := line.split())}
    groups = {"R1199_OBS": [], "R1199_TARGET": []}
    for row in csv.reader(z.read(name).decode().splitlines()):
        if row and row[0] in groups:
            groups[row[0]].append(list(map(float, row[1:])))
obs, target = (np.asarray(x, dtype=np.float32) for x in groups.values())
assert obs.shape[1] == 66 and target.shape[1] == 33, (obs.shape, target.shape)
assert np.array_equal(obs[:, 0], target[:, 0])
logged = target[:, 1:17]
active = (abs(logged).max(axis=1) > 1e-6)
adjacent = np.diff(obs[:, 0]) == 20
vx, vy, wz = np.asarray([commands[int(tick)] for tick in obs[:, 0]]).T
masks = {
    "all_active": active,
    "moving": active & ((abs(vx) >= .01) | (abs(wz) >= .01)),
    "zero": active & (abs(vx) < .005) & (abs(wz) < .005),
    "reverse_017_plus": active & (vx <= -.17) & (abs(wz) < .01),
    "pure_left": active & (abs(vx) < .005) & (wz >= .02),
    "pure_right": active & (abs(vx) < .005) & (wz <= -.02),
}
torch.set_num_threads(1)
def actor(path):
    state = torch.load(path, map_location="cpu")["actor_state_dict"]
    layers = sorted(int(key.split(".")[1]) for key in state if key.startswith("mlp.") and key.endswith(".weight"))
    def forward(frames):
        x = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
        for i in layers:
            x = torch.nn.functional.linear(x, state[f"mlp.{i}.weight"], state[f"mlp.{i}.bias"])
            if i != layers[-1]:
                x = torch.nn.functional.elu(x)
        return x.numpy()
    return forward


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensor_sensitivity(forward):
    """Matched-input perturbations, with scales selected for diagnosis."""
    samples = obs[masks["moving"], 1:]
    nominal = forward(samples)[:, :12]
    groups = {
        "gyro_rad_s": (slice(3, 6), .01),
        "projected_gravity": (slice(6, 9), .003),
        "leg_q_rad": (slice(10, 22), .003),
        "wheel_q_rad": (slice(22, 26), .003),
        "leg_dq_rad_s": (slice(26, 38), .03),
        "wheel_dq_rad_s": (slice(38, 42), .03),
        "previous_action": (slice(42, 58), .01),
    }
    rng = np.random.default_rng(311)
    result = {}
    for group, (columns, amplitude) in groups.items():
        changes = []
        clipped_changes = []
        for _ in range(16):
            perturbed = samples.copy()
            perturbed[:, columns] += amplitude * rng.choice((-1., 1.), perturbed[:, columns].shape)
            changed = forward(perturbed)[:, :12]
            changes.append(abs(changed - nominal))
            clipped_changes.append(abs(changed.clip(-6, 6) - nominal.clip(-6, 6)))
        result[group] = dict(perturbation_amplitude=amplitude,
                             raw_p99=float(np.quantile(changes,.99)),
                             clipped_p99=float(np.quantile(clipped_changes,.99)))
    return result

def stats(actions):
    out = {}
    for name, mask in masks.items():
        delta = abs(np.diff(actions[:, :12], axis=0))[adjacent & mask[1:] & mask[:-1]]
        if not delta.size:
            continue
        out[name] = dict(frames=int(mask.sum()), pairs=len(delta), mean=float(delta.mean()),
                         p95=float(np.quantile(delta,.95)), p99=float(np.quantile(delta,.99)),
                         pct_saturated=float((abs(actions[mask, :12]) >= 6).mean()*100),
                         pct_full_swing=float((delta >= 11.99).mean()*100),
                         pct_above_1=float((delta > 1).mean()*100))
    return out
def leg_target_stats(actions, cutoff_hz, slew_rad_s):
    dt = .020
    alpha = -np.expm1(-2 * np.pi * cutoff_hz * dt) if cutoff_hz else 1.
    filtered = np.zeros(12)
    previous = np.zeros(12)
    targets = np.zeros((len(actions), 12))
    motion = np.minimum(1., np.maximum(abs(vx) / .10, abs(wz) / .20))
    for i, raw in enumerate(actions[:, :12]):
        if i == 0 or not adjacent[i - 1]:
            filtered.fill(0.)
            previous.fill(0.)
        filtered += alpha * (np.clip(raw, -6., 6.) - filtered)
        residual = .35 * .18 * motion[i] * np.clip(filtered, -1., 1.)
        if slew_rad_s:
            residual = previous + np.clip(residual - previous, -slew_rad_s*dt, slew_rad_s*dt)
        targets[i] = previous = residual
    out = {}
    for name, mask in masks.items():
        delta = abs(np.diff(targets, axis=0))[adjacent & mask[1:] & mask[:-1]]
        if delta.size:
            out[name] = dict(pairs=len(delta), p99_rad=float(np.quantile(delta, .99)),
                             max_rad=float(delta.max()), rms_rad=float(np.sqrt((delta**2).mean())))
    return out
baseline_actor = actor(baseline)
candidate_actor = actor(challenger)
ref = baseline_actor(obs[:, 1:])
candidate = candidate_actor(obs[:, 1:])
out = dict(replay_frames=len(obs), moving_frames=int(masks["moving"].sum()),
           baseline_clipped=stats(np.clip(ref, -6, 6)), candidate_clipped=stats(np.clip(candidate, -6, 6)),
           baseline_unclipped=stats(ref), candidate_unclipped=stats(candidate))
out["input_sha256"] = {"fixture_archive": sha256(archive), "baseline_checkpoint": sha256(baseline),
                       "candidate_checkpoint": sha256(challenger)}
out["leg_residual_only"] = {
    "baseline_no_filter": leg_target_stats(ref, 0., 0.),
    "baseline_filter_slew": leg_target_stats(ref, 5., .9),
    "candidate_no_filter": leg_target_stats(candidate, 0., 0.),
    "candidate_filter_slew": leg_target_stats(candidate, 5., .9),
}
def action_response(actions):
    moving = actions[masks["moving"]]
    wheel = moving[:, 12:]
    bounded_wheel = np.clip(wheel, -1., 1.)
    common = bounded_wheel.mean(axis=1)
    steering = (bounded_wheel * np.array((1., -1., 1., -1.))).mean(axis=1)
    return dict(channel_min=moving.min(axis=0).tolist(), channel_max=moving.max(axis=0).tolist(),
                channel_std=moving.std(axis=0).tolist(),
                leg_clip_fraction=float((abs(moving[:, :12]) >= 6).mean()),
                wheel_clip_fraction=float((abs(wheel) >= 6).mean()),
                wheel_residual_bound_fraction=(abs(wheel) >= 1.).mean(axis=0).tolist(),
                wheel_bounded_residual_span=np.ptp(bounded_wheel, axis=0).tolist(),
                wheel_common_range=[float(common.min()), float(common.max())],
                wheel_steering_range=[float(steering.min()), float(steering.max())])

out["moving_action_response"] = {
    "baseline": action_response(ref),
    "candidate": action_response(candidate),
}
out["sensor_sensitivity"] = {
    "baseline": sensor_sensitivity(baseline_actor),
    "candidate": sensor_sensitivity(candidate_actor),
}
if args.onnx:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    frames = np.ascontiguousarray(obs[:, 1:])
    onnx_actions = np.concatenate([runtime.run(["actions"], {"obs": frame[None]})[0] for frame in frames])
    expected = np.clip(candidate, -6., 6.)
    if onnx_actions.shape != expected.shape or not np.isfinite(onnx_actions).all():
        raise ValueError("Invalid ONNX replay output")
    error = np.abs(onnx_actions - expected)
    out["onnx_hardware_replay"] = {
        "onnx_sha256": sha256(args.onnx), "frames": len(frames),
        "max_action_error": float(error.max()),
        "moving_max_action_error": float(error[masks["moving"]].max()),
        "passed": bool(error.max() <= 1e-4),
        "moving_action_response": action_response(onnx_actions),
        "leg_residual_filter_slew": leg_target_stats(onnx_actions, 5., .9),
    }
report.write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, indent=2))
