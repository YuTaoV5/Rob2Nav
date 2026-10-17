"""Measured-return memory, independent of simulator and sensor drivers."""
import torch


class ElevationMemory:
    def __init__(self, num_envs, device, lifetime=1.):
        self.world = torch.zeros((num_envs, 187, 3), device=device)
        self.time = torch.full((num_envs, 187), -1e6, device=device)
        self.lifetime = lifetime

    def update(self, new_hits, root_pos, yaw, now, reset, fresh):
        """Inputs are world XYZ returns; non-finite entries mean unknown.

        Retain real return coordinates, not moving grid centers. Returns 187
        relative heights and 187 validity values in row-major y/x order.
        """
        self.time[reset] = -1e6
        count_envs = root_pos.shape[0]
        device = root_pos.device
        hits = torch.cat((self.world, new_hits), 1)
        times = torch.cat((self.time, torch.full_like(new_hits[..., 2], now)), 1)
        rel = hits-root_pos[:, None, :]
        x = rel[..., 0]*yaw.cos()[:, None]+rel[..., 1]*yaw.sin()[:, None]
        y = -rel[..., 0]*yaw.sin()[:, None]+rel[..., 1]*yaw.cos()[:, None]
        ix = ((x+.55)/.1).round().long()
        iy = ((y+.5)/.1).round().long()
        valid = torch.isfinite(hits).all(-1) & (ix >= 0) & (ix < 17) & (iy >= 0) & (iy < 11)
        valid[:, :187] &= now-times[:, :187] < self.lifetime
        valid[:, 187:] &= fresh[:, None]
        index = (iy*17+ix).clamp(0, 186)
        values = torch.where(valid, hits[..., 2], -torch.inf)
        elevation = torch.full((count_envs, 187), -torch.inf, device=device)
        elevation.scatter_reduce_(1, index, values, reduce="amax", include_self=True)
        mask = torch.isfinite(elevation)
        count = hits.shape[1]
        ids = torch.arange(count, device=device)[None].expand(count_envs, -1)
        winners = valid & (values == elevation.gather(1, index))
        chosen = torch.full((count_envs, 187), count, device=device, dtype=torch.long)
        chosen.scatter_reduce_(1, index, torch.where(winners, ids, count), reduce="amin", include_self=True)
        selected = chosen.clamp(max=count-1)
        self.world = hits.gather(1, selected[..., None].expand(-1, -1, 3))
        self.time = torch.where(mask, times.gather(1, selected), -1e6)
        heights = torch.where(mask, (root_pos[:, 2, None]-elevation-.5).clamp(-1., 1.), 0.)
        return torch.cat((heights, mask.float()), -1)
