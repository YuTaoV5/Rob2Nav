"""Render real MuJoCo rollouts of the packaged AS2-W joint policy."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import evaluate_omni_stairs as evaluation
from evaluate_omni_joint import make_controller
from train_omni_stop_transition import load


WIDTH, HEIGHT, FPS = 768, 576, 25


class Video:
    def __init__(self, output, title, seed, stairs=False):
        self.output, self.title, self.seed, self.stairs = output, title, seed, stairs
        self.index = 0
        self.renderer = None
        self.process = None
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            self.font = ImageFont.load_default()

    def capture(self, model, data, command):
        self.index += 1
        if self.index % 2:
            return
        if self.renderer is None:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
            model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
            self.renderer = evaluation.sim.mj.Renderer(model, height=HEIGHT, width=WIDTH)
            self.base = model.body("base_link").id
            self.camera = evaluation.sim.mj.MjvCamera()
            self.scene_option = evaluation.sim.mj.MjvOption()
            if self.stairs:
                # The benchmark groups terrain separately for range queries.
                self.scene_option.geomgroup[5] = 1
                model.geom("ground").rgba[:] = (.27, .30, .34, 1.)
                for level in range(1, 6):
                    shade = .34 + .048 * level
                    model.geom(f"stair_{level}").rgba[:] = (shade, shade + .10, shade + .14, 1.)
            self.process = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
                "-c:v", "libx264", "-threads", "2", "-crf", "21", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.output),
            ], stdin=subprocess.PIPE)
        center = data.xpos[self.base]
        self.camera.lookat[:] = (center[0] + (.28 if self.stairs else .05), center[1], max(.22, center[2] - .08))
        self.camera.distance = 2.75 if self.stairs else 2.0
        self.camera.azimuth = 132 if self.stairs else 128
        self.camera.elevation = -21 if self.stairs else -28
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.scene_option)
        frame = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, WIDTH, 78), fill=(12, 20, 29))
        draw.text((16, 10), self.title, font=self.font, fill=(245, 248, 255))
        draw.text((16, 42), f"MuJoCo | seed {self.seed} | {self.index * .02:4.2f} s   "
                  f"command vx {command[0]:+.2f}  vy {command[1]:+.2f}  wz {command[2]:+.2f}",
                  font=self.font, fill=(180, 214, 238))
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process is None:
            raise RuntimeError("No video frames were rendered")
        self.process.stdin.close()
        status = self.process.wait()
        self.renderer.close()
        if status:
            raise RuntimeError(f"FFmpeg failed with status {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("fixed_report", type=Path)
    parser.add_argument("confirmation_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stairs-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sim = evaluation.sim
    policy, contract, limits, base, build = load(args.results.resolve(), args.policy)
    defaults = np.asarray(contract["default_joint_pos"])
    dt = contract["control_dt"]
    fixed = json.loads(args.fixed_report.read_text())
    confirmation = json.loads(args.confirmation_report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = []

    def controller():
        return make_controller(policy, contract, build, base())

    flat_files = []
    names = ("forward", "reverse", "left", "right", "turn_left", "turn_right",
             "diagonal_left", "arc_right", "reverse_mixed")
    for name in (() if args.stairs_only else names):
        if name == "reverse_mixed":
            source = next((r for r in confirmation["flat"] if r["passed"]
                           and r["target"][0] < -.12 and abs(r["target"][1]) >= .10
                           and abs(r["target"][2]) >= .25), None)
        else:
            source = next((r for r in fixed["flat"] if r["command"] == name and r["passed"]), None)
        if source is None:
            continue
        target = np.asarray(source["target"])
        seed = source["seed"]
        path = args.output / f"flat-{name}.mp4"
        video = Video(path, f"AS2-W R41 flat | {name.replace('_', ' ')} then stop", seed)
        actor = controller()
        try:
            row = evaluation.flat_episode(None, contract, limits, target, seed, evaluation.gait_clock,
                                          actor.plan, video.capture)
        finally:
            video.close()
        if not row["passed"]:
            raise RuntimeError(f"Flat video did not reproduce the passing result for {name}/{seed}")
        flat_files.append(path)
        rendered.append({"command": name, "target": target.tolist(), "seed": seed, "passed": True})
    if flat_files:
        listing = args.output / "flat-clips.txt"
        listing.write_text("".join(f"file '{path.name}'\n" for path in flat_files))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(args.output / "flat-commands.mp4")], check=True)

    for height, report in ((.08, fixed), (.06, confirmation)):
        source = next((r for r in report["stairs"] if r["passed"] and abs(r["stair_rise_m"] - height) < .001), None)
        if source is None:
            continue
        seed = source["seed"]
        model = sim.load_flat_model(True, True, "stairs", seed)
        stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
        command = np.array((.4, 0., 0.))
        video = Video(args.output / f"stairs-{round(100 * height):02d}cm.mp4",
                      f"AS2-W R41 blind ascent | five {round(100 * height)} cm risers", seed, True)
        actor = controller()

        def plan(data, obs, elapsed, advance):
            action = actor.plan(model, data, obs, command, round(elapsed / dt), advance)
            leg = np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T)
            wheel = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
            return action, leg, wheel

        try:
            with torch.inference_mode():
                row = sim.evaluate(model, contract, policy["anchor"], seed, 0., limits=limits, action_clip=6.,
                                   wheel_scale=16., wheel_angle_period=4. * np.pi, dynamics_stress=stress,
                                   terrain="stairs", observation_suffix=evaluation.gait_clock, native_planner=plan,
                                   capture=lambda data: video.capture(model, data, command))
        finally:
            video.close()
        if not row["passed"]:
            raise RuntimeError(f"Stair video did not reproduce the passing result for seed {seed}")
        rendered.append({"stair_rise_m": height, "seed": seed, "passed": True,
                         "top_support_s": float(row["top_support_s"]), "peak_impact_n": float(row["peak_impact_n"])})
    (args.output / "video-rollouts.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print(json.dumps(rendered), flush=True)


if __name__ == "__main__":
    main()
