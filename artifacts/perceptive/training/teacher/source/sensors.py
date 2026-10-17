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
    u = torch.linspace(-math.tan(math.radians(43.5)), math.tan(math.radians(43.5)), 32, device=device)
    v = torch.linspace(-math.tan(math.radians(29)), math.tan(math.radians(29)), 20, device=device)
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
    """187 cells + validity, with <=1 s observed-return memory using ideal odometry.

    Store actual return coordinates, not grid centers: repeatedly recentering
    grid centers would incorrectly drag remembered terrain along with the body.
    """
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    # wxyz quaternion -> yaw
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
    new_hits = torch.cat([env.scene[n].data.ray_hits_w for n in ("depth_rays", "lidar_rays")], 1).clone()
    now = float(env.common_step_counter)*env.step_dt
    if not hasattr(env, "_fusion_world"):
        env._fusion_world = torch.zeros((env.num_envs, 187, 3), device=env.device)
        env._fusion_time = torch.full((env.num_envs, 187), -1e6, device=env.device)
        env._fusion_frame_time = -1e6
    reset = env.episode_length_buf == 0
    env._fusion_time[reset] = -1e6
    fresh = now-env._fusion_frame_time >= .099
    if fresh:
        env._fusion_frame_time = now
    if env.cfg.observations.policy.enable_corruption:
        new_hits[..., 2] += .01*torch.randn_like(new_hits[..., 2])
    hits = torch.cat((env._fusion_world, new_hits), 1)
    times = torch.cat((env._fusion_time, torch.full_like(new_hits[..., 2], now)), 1)
    rel = hits-robot.data.root_pos_w[:, None, :]
    x = rel[..., 0]*yaw.cos()[:, None]+rel[..., 1]*yaw.sin()[:, None]
    y = -rel[..., 0]*yaw.sin()[:, None]+rel[..., 1]*yaw.cos()[:, None]
    ix = ((x+.55)/.1).round().long()
    iy = ((y+.5)/.1).round().long()
    valid = torch.isfinite(hits).all(-1) & (ix >= 0) & (ix < 17) & (iy >= 0) & (iy < 11)
    valid[:, :187] &= now-times[:, :187] < 1.
    valid[:, 187:] &= fresh | reset[:, None]
    # Random return dropout and z noise are observation corruption only.
    if env.cfg.observations.policy.enable_corruption:
        valid[:, 187:] &= torch.rand_like(x[:, 187:]) > .05
    index = (iy*17+ix).clamp(0, 186)
    z = hits[..., 2]
    values = torch.where(valid, z, torch.full_like(z, -torch.inf))
    elevation = torch.full((env.num_envs, 187), -torch.inf, device=env.device)
    elevation.scatter_reduce_(1, index, values, reduce="amax", include_self=True)
    mask = torch.isfinite(elevation)
    count = hits.shape[1]
    candidate_ids = torch.arange(count, device=env.device)[None].expand(env.num_envs, -1)
    winners = valid & (values == elevation.gather(1, index))
    chosen = torch.full((env.num_envs, 187), count, device=env.device, dtype=torch.long)
    chosen.scatter_reduce_(1, index, torch.where(winners, candidate_ids, count), reduce="amin", include_self=True)
    selected = chosen.clamp(max=count-1)
    env._fusion_world = hits.gather(1, selected[..., None].expand(-1, -1, 3))
    env._fusion_time = torch.where(mask, times.gather(1, selected), -1e6)
    # Same sign/offset as Isaac Lab height_scan. Unknown -> 0 with mask=0.
    heights = torch.where(mask, (robot.data.root_pos_w[:, 2, None]-elevation-.5).clamp(-1., 1.), 0.)
    return torch.cat((heights, mask.float()), -1)
