"""Record blind AS2-W checkpoint rollouts with the commands and actual velocity."""
import argparse
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from as2w_history import History
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


FLAT = ((.10, 0., 0.), (0., .10, 0.), (-.071, .071, -.20),
        (0., 0., .30), (-.10, 0., 0.), (0., 0., 0.))


def record(args, actor, has_history, mode):
    stairs = mode == "stairs"
    seed = 27004 if stairs else 27001
    env = OmniEnvironment(args.assets, 0 if stairs else 1, stair_rise=.06,
                          stop_training=True, low_speed_course=stairs)
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10,
                                         "target": (0., 0., 0.)})
    env.apply_motion_prior = False
    env.next_command_tick = 10**9
    env.stair_hold_enabled = stairs
    history = History()
    camera = mujoco.MjvCamera()
    camera.azimuth = 112 if stairs else 128
    camera.elevation = -15
    camera.distance = 1.7
    width, height = 640, 360
    output = args.output / f"{args.label}-{mode}.mp4"
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
         "-threads", "2", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    try:
        with mujoco.Renderer(env.model, height=height, width=width) as renderer:
            seconds = env.stair_seconds if stairs else 2. * len(FLAT)
            for tick in range(round(seconds / env.dt)):
                elapsed = tick * env.dt
                if not stairs:
                    env.target_goal = np.array(FLAT[min(len(FLAT) - 1, int(elapsed / 2.))])
                command = env.command.copy()
                obs, _, terminated, _, _ = env.step(actor(history.push(obs) if has_history else obs))
                if tick % 2 and not terminated:
                    continue
                camera.lookat[:] = env.data.xpos[env.base]
                camera.lookat[0] += .20 if stairs else 0.
                renderer.update_scene(env.data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 47), fill=(13, 18, 25))
                rotation = env.data.xmat[env.base].reshape(3, 3)
                vx, vy, _ = rotation.T @ env.data.qvel[:3]
                yaw = float(env.data.qvel[5])
                detail = f"6 cm x 5 | wheel levels {env.levels.tolist()}" if stairs else "Flat | dynamic joystick"
                draw.text((9, 5), f"{args.label} | MuJoCo | {detail} | {elapsed:4.1f} s", fill="white")
                draw.text((9, 21), f"Cmd vx/vy/wz {command[0]:+.2f} / {command[1]:+.2f} / {command[2]:+.2f}"
                          f"    Actual {vx:+.2f} / {vy:+.2f} / {yaw:+.2f}", fill="white")
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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("label")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    for mode in ("flat", "stairs"):
        record(args, actor, state["actor.0.weight"].shape[1] == History.size, mode)


if __name__ == "__main__":
    main()
