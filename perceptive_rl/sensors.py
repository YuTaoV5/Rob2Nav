"""Occlusion-aware ideal depth/lidar rays -> measured local elevation + validity.

These are simplified ray sensors, not calibrated D435i/Mid-360 drivers. Unknown
cells stay unknown; no terrain mesh queries are used to fill them.
"""
import math
import torch
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def camera_pattern(cfg, device):
    u = (torch.arange(32, device=device)+.5-16)/16*math.tan(math.radians(43.5))
    v = (torch.arange(20, device=device)+.5-10)/10*math.tan(math.radians(29))
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    directions = torch.stack((torch.ones_like(uu), -uu, -vv), -1).reshape(-1, 3)
    pitch = math.radians(20)
    rotation = torch.tensor([[math.cos(pitch), 0., math.sin(pitch)], [0., 1., 0.],
                             [-math.sin(pitch), 0., math.cos(pitch)]], device=device)
    directions = directions @ rotation.T
    return torch.zeros_like(directions), torch.nn.functional.normalize(directions, dim=-1)


def lidar_pattern(cfg, device):
    az = torch.arange(96, device=device)*2*math.pi/96
    # Mid-360 nominal vertical field: approximately -7 to +52 deg. This
    # reduced deterministic grid does not reproduce its non-repetitive scan.
    el = torch.linspace(math.radians(-7), math.radians(52), 12, device=device)
    ee, aa = torch.meshgrid(el, az, indexing="ij")
    directions = torch.stack((ee.cos()*aa.cos(), ee.cos()*aa.sin(), ee.sin()), -1).reshape(-1, 3)
    return torch.zeros_like(directions), directions


@configclass
class CameraPattern(PatternBaseCfg):
    func = camera_pattern


@configclass
class LidarPattern(PatternBaseCfg):
    func = lidar_pattern


def add_sensors(cfg):
    cfg.scene.depth_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", update_period=.10,
        offset=RayCasterCfg.OffsetCfg(pos=(.25, 0., .06)), ray_alignment="base",
        pattern_cfg=CameraPattern(), max_distance=4., mesh_prim_paths=["/World/ground"])
    cfg.scene.lidar_rays = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", update_period=.10,
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., .16)), ray_alignment="base",
        pattern_cfg=LidarPattern(), max_distance=8., mesh_prim_paths=["/World/ground"])


def measured_height(env):
    """187 cells + validity; <=1 s observed-return memory with ideal odometry."""
    from .elevation import ElevationMemory
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
    hits = torch.cat([env.scene[n].data.ray_hits_w for n in ("depth_rays", "lidar_rays")], 1).clone()
    now = float(env.common_step_counter)*env.step_dt
    if not hasattr(env, "_elevation_memory"):
        env._elevation_memory = ElevationMemory(env.num_envs, env.device)
        env._fusion_frame_time = -1e6
    reset = env.episode_length_buf == 0
    new_frame = now-env._fusion_frame_time >= .099
    if new_frame:
        env._fusion_frame_time = now
    fresh = reset | new_frame
    if env.cfg.observations.policy.enable_corruption:
        hits[..., 2] += .01*torch.randn_like(hits[..., 2])
        hits[torch.rand_like(hits[..., 2]) < .05] = torch.inf
    return env._elevation_memory.update(hits, robot.data.root_pos_w, yaw, now, reset, fresh)
