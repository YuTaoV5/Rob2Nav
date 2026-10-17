"""Render simulator and measured sensor observations for inspection."""
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .env import Go2MaplessEnv


def main():
    env = Go2MaplessEnv(scene="slalom")
    try:
        env.reset(seed=0)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        ax[0].imshow(env.render())
        ax[0].axis("off")
        ax[0].set_title("MuJoCo Go2 / RoboGauge locomotion")
        im = ax[1].imshow(env.depth, cmap="viridis", vmin=0, vmax=env.cfg["max_range"])
        ax[1].set_title("D435i idealized depth (metres)")
        ax[1].set_xlabel("Image column")
        ax[1].set_ylabel("Image row")
        fig.colorbar(im, ax=ax[1], shrink=.65)
        points = env.points
        ax[2].scatter(points[:, 0], points[:, 1], s=2, c=points[:, 2], cmap="viridis")
        ax[2].add_patch(plt.Circle((0, 0), env.cfg["robot_radius"], fill=False, color="red"))
        ax[2].arrow(0, 0, .8, 0, width=.025, color="red")
        ax[2].set(xlim=(-4, 8), ylim=(-4, 4), aspect="equal", xlabel="Body X / forward (m)", ylabel="Body Y / left (m)", title="Current local returns (no accumulated map)")
        output = Path("outputs/sensor_preview.png")
        output.parent.mkdir(exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        np.savez_compressed("outputs/sensor_preview.npz", depth_m=env.depth, lidar_ranges_m=env.lidar_ranges, points_body_m=env.points)
    finally:
        env.close()


if __name__ == "__main__":
    main()
