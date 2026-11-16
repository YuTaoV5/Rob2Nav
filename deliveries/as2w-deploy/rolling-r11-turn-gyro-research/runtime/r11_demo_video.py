#!/usr/bin/env python3
"""Record a demo video of the best current weights, with a visible ground grid.

This is a VISUALISATION artefact, not an acceptance result: the training guide requires demo video
and formal acceptance to be marked separately, so the clip shows only the six cases that pass on BOTH
simulators (hold, straight fwd/bwd 0.25, arc_right 0.25, brake fwd/bwd 0.25) and nothing else.

Ground grid: MuJoCo's ground geom has no material in the tracked MJCF, so a temporary copy is written
with a built-in checker texture applied.  The tracked asset is not modified - the copy lives outside
the repo and only exists for rendering.

Frames are written as PPM (no PIL/imageio in this venv) and encoded with ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import re
import numpy as np
import torch
import torch.nn as nn
import mujoco

DT = 0.000625
SUBSTEPS = 32
STEP_DT = DT * SUBSTEPS
LEG_KD, WHEEL_KD = 2.0, 6.0
LIMIT = {"hip": 54.0, "thigh": 54.0, "calf": 81.0, "foot": 13.5}
LIM = np.array([LIMIT["hip"]] * 4 + [LIMIT["thigh"]] * 4 + [LIMIT["calf"]] * 4 + [LIMIT["foot"]] * 4)
LEG_SCALE, WHEEL_SCALE, CLIP = 0.35, 16.0, 6.0
S_WHEEL = np.array([-1.0, 1.0, -1.0, 1.0])
JOINTS = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]
DEFAULT = {"hip": 0.0, "thigh": 0.8, "calf": -1.6, "foot": 0.0}
DEF = np.array([DEFAULT[n.split("_")[1]] for n in JOINTS])


def build_actor(sd):
    lin = {}
    for k, v in sd.items():
        if not k.startswith("mlp."):
            continue
        _, idx, param = k.split(".")
        if param in ("weight", "bias"):
            lin.setdefault(int(idx), {})[param] = v
    layers = []
    order = sorted(lin)
    for n, i in enumerate(order):
        w, b = lin[i]["weight"], lin[i]["bias"]
        m = nn.Linear(w.shape[1], w.shape[0]); m.weight.data = w.clone(); m.bias.data = b.clone()
        layers.append(m)
        if n != len(order) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).eval()


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_grid_xml(src, dst):
    """Copy the MJCF and give the ground a checker texture so the grid is visible."""
    s = Path(src).read_text()
    asset = ('  <asset>\n'
             '    <texture name="gridtex" type="2d" builtin="checker" rgb1="0.95 0.95 0.95" '
             'rgb2="0.66 0.70 0.74" width="512" height="512"/>\n'
             '    <material name="gridmat" texture="gridtex" texrepeat="12 12" texuniform="true" '
             'reflectance="0.06"/>\n')
    assert "<asset>" in s
    s = s.replace("<asset>", asset, 1)
    # brighter, more even lighting: the default headlight leaves the robot quite dark
    visual = ('  <visual>\n'
              '    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" '
              'specular="0.15 0.15 0.15" active="1"/>\n'
              '    <global offwidth="640" offheight="480"/>\n'
              '  </visual>\n')
    s = s.replace("<worldbody>", visual + "  <worldbody>", 1)
    old_ground = '<geom name="ground" type="plane" size="4 4 .1" friction=".62 .005 .0001" quat="1.0 0 0.0 0" />'
    new_ground = ('<geom name="ground" type="plane" size="8 8 .1" friction=".62 .005 .0001" '
                  'quat="1.0 0 0.0 0" material="gridmat" />')
    assert old_ground in s, "ground geom not found verbatim"
    s = s.replace(old_ground, new_ground, 1)
    # The grid XML is written to the OUTPUT directory, so a RELATIVE compiler meshdir would be
    # resolved against that directory instead of the source XML and the meshes would not be found.
    # Rewrite meshdir to an absolute path pointing back at the tracked STL directory.
    m = re.search(r'meshdir="([^"]*)"', s)
    if m:
        rel = m.group(1)
        base = Path(src).resolve().parent
        abs_mesh = (base / rel).resolve()
        if not abs_mesh.is_dir():
            raise SystemExit("meshdir does not resolve: %s" % abs_mesh)
        s = s.replace('meshdir="%s"' % rel, 'meshdir="%s"' % abs_mesh, 1)
        print("AS2W_VIDEO_MESHDIR %s" % abs_mesh, flush=True)
    Path(dst).write_text(s)
    return dst


def write_ppm(path, rgb):
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--yaw-kp", type=float, default=0.0)
    ap.add_argument("--ff-cap", type=float, default=4.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    frames_dir = a.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    grid_xml = make_grid_xml(a.xml, a.out / "mjcf_grid.xml")

    m = mujoco.MjModel.from_xml_path(str(grid_xml))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    qadr = np.array([int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    vadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS])
    aid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
    kp = np.r_[np.full(12, 55.0), np.full(4, 0.0)]
    kd = np.r_[np.full(12, LEG_KD), np.full(4, WHEEL_KD)]
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    act = build_actor(ck.get("actor_state_dict") or ck.get("model_state_dict"))
    renderer = mujoco.Renderer(m, height=a.height, width=a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.9, -22.0, 125.0

    state = {"cmd": np.zeros(3)}

    def obs_of(prev, ep, zr):
        q = d.qpos[qadr].copy(); dq = d.qvel[vadr].copy()
        R = quat_to_mat(d.qpos[3:7])
        qrel = q - DEF
        qrel[12:] = (qrel[12:] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        o = np.zeros(65)
        o[0:3] = state["cmd"] - np.array([0.4, 0.0, 0.0])
        o[3:6] = R.T @ d.qvel[3:6]
        o[6:9] = R.T @ np.array([0.0, 0.0, -1.0])
        o[9] = 0.4
        o[10:26] = qrel
        o[26:42] = dq
        o[42:58] = prev
        clk = 2 * np.pi * ep * STEP_DT / 0.62
        o[58], o[59] = np.sin(clk), np.cos(clk)
        zero = float(np.all(state["cmd"] == 0.0))
        if zr[0] is None:
            zr[0] = qrel[12:].copy()
        dd = (qrel[12:] - zr[0] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
        o[60:64] = np.clip(dd / 0.20, -3, 3) * zero
        o[64] = zero
        return o

    def step_ctrl(act16):
        tgt = np.r_[act16[:12] * LEG_SCALE + DEF[:12], act16[12:] * WHEEL_SCALE]
        ff = np.zeros(4)
        if a.yaw_kp != 0.0:
            wz = float(d.qvel[5])
            mag = float(np.clip(a.yaw_kp * (state["cmd"][2] - wz), -a.ff_cap, a.ff_cap))
            ff = S_WHEEL * mag
        for _ in range(SUBSTEPS):
            q = d.qpos[qadr]; v = d.qvel[vadr]
            tau = np.r_[kp[:12] * (tgt[:12] - q[:12]) - kd[:12] * v[:12],
                        kd[12:] * (tgt[12:] - v[12:]) + ff]
            d.ctrl[aid] = np.clip(tau, -LIM, LIM)
            mujoco.mj_step(m, d)

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 0.381395)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[qadr] = DEF
        mujoco.mj_forward(m, d)

    frame_i = [0]

    def render(label):
        cam.lookat[:] = d.qpos[0:3]
        renderer.update_scene(d, camera=cam)
        rgb = renderer.render()
        write_ppm(frames_dir / ("f%05d.ppm" % frame_i[0]), rgb)
        frame_i[0] += 1

    script = []
    # (label, command builder, seconds)  -- only the six cases that pass on BOTH simulators
    segs = [
        ("hold  zero command", lambda t: (0.0, 0.0, 0.0), 5.0),
        ("straight_fwd  0.25 m/s", lambda t: (0.25, 0.0, 0.0), 6.0),
        ("straight_bwd  0.25 m/s", lambda t: (-0.25, 0.0, 0.0), 6.0),
        ("arc_right  0.25 m/s  wz -0.05", lambda t: (0.25, 0.0, -0.05), 6.0),
        ("brake_fwd  0.25 -> 0", lambda t: (0.25, 0.0, 0.0) if t < 5.0 else (0.0, 0.0, 0.0), 10.0),
        ("brake_bwd  -0.25 -> 0", lambda t: (-0.25, 0.0, 0.0) if t < 5.0 else (0.0, 0.0, 0.0), 10.0),
    ]
    for label, fn, secs in segs:
        reset()
        state["cmd"] = np.zeros(3)
        prev = np.zeros(16); zr = [None]; ep = 0
        for _ in range(int(2.0 / STEP_DT)):
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
        n = int(secs / STEP_DT)
        for k in range(n):
            state["cmd"] = np.array(fn(k * STEP_DT))
            o = obs_of(prev, ep, zr)
            with torch.no_grad():
                prev = np.clip(act(torch.tensor([o], dtype=torch.float32))[0].numpy(), -CLIP, CLIP)
            step_ctrl(prev); ep += 1
            if k % max(1, int(1.0 / (a.fps * STEP_DT))) == 0:
                render(label)
        script.append({"label": label, "start_frame": script[-1]["end_frame"] if script else 0,
                       "end_frame": frame_i[0]})
        print("SEG %-32s frames %d..%d" % (label, script[-1]["start_frame"], frame_i[0]), flush=True)
    renderer.close()

    # encode with a text label per segment (drawtext if available), then concat
    def have_drawtext():
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
            return "drawtext" in out
        except Exception:
            return False

    font = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(cand):
            font = cand
            break
    use_text = have_drawtext() and font is not None
    print("DRAWTEXT %s font=%s" % (use_text, font), flush=True)

    clips = []
    for i, sg in enumerate(script):
        start = sg["start_frame"]
        cnt = sg["end_frame"] - start
        clip = a.out / ("clip%02d.mp4" % i)
        vf = []
        if use_text:
            vf.append("drawtext=fontfile=%s:text='%s':fontcolor=white:fontsize=26:"
                      "box=1:boxcolor=black@0.55:boxborderw=10:x=18:y=18" % (font, sg["label"].replace(":", "\\:")))
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-framerate", str(a.fps), "-start_number", str(start),
               "-i", str(frames_dir / "f%05d.ppm"), "-frames:v", str(cnt)]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)]
        subprocess.run(cmd, check=True)
        clips.append(clip)
    lst = a.out / "concat.txt"
    lst.write_text("".join("file '%s'\n" % c for c in clips))
    final = a.out / "as2w_rolling_best_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(lst), "-c", "copy", str(final)], check=True)
    (a.out / "segments.json").write_text(json.dumps(script, indent=1))
    print("VIDEO %s  size=%.1f MB  frames=%d  text=%s"
          % (final, final.stat().st_size / 1e6, frame_i[0], use_text), flush=True)
    print("VIDEO_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
