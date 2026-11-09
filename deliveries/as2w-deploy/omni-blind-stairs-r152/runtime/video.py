import subprocess
from PIL import Image, ImageDraw, ImageFont
import evaluate_omni_stairs as evaluation
WIDTH, HEIGHT, FPS = 640, 480, 25

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
                "-c:v", "libx264", "-threads", "2", "-crf", "25", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.output),
            ], stdin=subprocess.PIPE)
        center = data.xpos[self.base]
        self.camera.lookat[:] = (center[0] + (.28 if self.stairs else .05), center[1], max(.22, center[2] - .08))
        self.camera.distance = 2.7 if self.stairs else 1.9
        self.camera.azimuth = 108 if self.stairs else 122
        self.camera.elevation = -28 if self.stairs else -27
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


