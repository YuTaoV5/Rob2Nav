"""Record blind AS2-W checkpoint rollouts with the commands and actual velocity."""
import argparse
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

from as2w_history import History
from as2w_zero_memory import ZeroWheelMemory
from eval_as2w_strict_hold import ALL_FLAT_COMMANDS
from train_stair_low_speed_course import OmniEnvironment


FLAT = (("static zero", (0., 0., 0.), 10.),) + tuple(
    (name, cmd, 10. if name == "stop" else 2.) for name, cmd in ALL_FLAT_COMMANDS
)


def record(args, actor, has_history, mode):
    stairs = mode == "stairs"
    seed = 27002
    env = OmniEnvironment(args.assets, 0 if stairs else 1, stair_rise=.06,
                          stop_training=True, low_speed_course=stairs)
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10,
                                         "target": (0., 0., 0.)})
    env.apply_motion_prior = False
    env.next_command_tick = 10**9
    env.stair_hold_enabled = stairs
    history = History()
    camera = mujoco.MjvCamera()
    camera.azimuth = 132
    camera.elevation = -15
    camera.distance = 1.7
    camera.lookat[:] = env.data.xpos[env.base]
    camera.lookat[2] = .28
    width, height = 640, 360
    boundaries = np.cumsum([duration for _, _, duration in FLAT])
    phase_last = -1
    phase_origin = env.data.qpos[:2].copy()
    phase_drift = 0.
    output = args.output / f"{args.label}-{mode}.mp4"
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "2", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    try:
        with mujoco.Renderer(env.model, height=height, width=width) as renderer:
            seconds = boundaries[-1]
            for tick in range(round(seconds / env.dt)):
                elapsed = tick * env.dt
                if not stairs:
                    phase = int(np.searchsorted(boundaries, elapsed, side="right"))
                    label, target, _ = FLAT[phase]
                    env.target_goal = np.array(target)
                    local_time = elapsed - (boundaries[phase - 1] if phase else 0.)
                    if phase_last != phase or local_time < 1.:
                        phase_origin = env.data.qpos[:2].copy()
                        phase_drift = 0.
                        phase_last = phase
                    else:
                        phase_drift = max(phase_drift, float(np.linalg.norm(env.data.qpos[:2] - phase_origin)))
                command = env.command.copy()
                obs, _, terminated, _, _ = env.step(actor(history.push(obs) if has_history else obs))
                if tick % 2 and not terminated:
                    continue
                renderer.update_scene(env.data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 52), fill=(13, 18, 25))
                rotation = env.data.xmat[env.base].reshape(3, 3)
                vx, vy, _ = rotation.T @ env.data.qvel[:3]
                yaw = float(env.data.qvel[5])
                detail = f"Flat | seed {seed} | {label}"
                draw.text((9, 5), f"{args.label} | MuJoCo | {detail} | {elapsed:4.1f} s", fill="white")
                draw.text((9, 21), f"Cmd vx/vy/wz {command[0]:+.2f} / {command[1]:+.2f} / {command[2]:+.2f}"
                          f"    Actual {vx:+.2f} / {vy:+.2f} / {yaw:+.2f}", fill="white")
                if not any(target):
                    draw.text((9, 36), f"10 s hold | drift after 1 s: {phase_drift * 100:.1f} cm",
                              fill=(240, 205, 133))
                if terminated:
                    draw.text((9, 33), "Episode terminated", fill=(255, 136, 110))
                encoder.stdin.write(frame.tobytes())
                if terminated:
                    for _ in range(20):
                        encoder.stdin.write(frame.tobytes())
                    break
    finally:
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"ffmpeg could not encode {output}")
    print(f"{output}: seed={seed} duration={elapsed:.2f}s levels={env.levels.tolist()} terminated={terminated}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])

    def actor(features):
        return model.run(["actions"], {"obs": features[None, :]})[0][0]
    memory = ZeroWheelMemory()
    record(args, lambda obs: actor(memory.push(obs)), False, "flat")


if __name__ == "__main__":
    main()
