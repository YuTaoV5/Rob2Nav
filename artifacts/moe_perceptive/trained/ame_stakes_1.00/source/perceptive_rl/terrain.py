"""Isaac Lab mesh adapter; shared geometry is also used in MuJoCo."""
import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass
from .geometry import lane_boxes


def lane(difficulty, cfg):
    meshes = []
    for x0, x1, y0, y1, top, bottom in lane_boxes(difficulty, cfg.kind, cfg.size):
        mesh = trimesh.creation.box((x1-x0, y1-y0, top-bottom))
        mesh.apply_translation(((x0+x1)/2, (y0+y1)/2, (top+bottom)/2))
        meshes.append(mesh)
    return meshes, np.array([4.,4.,0.] if cfg.kind.startswith("ame_") else [1., cfg.size[1]/2, 0.])


@configclass
class LaneCfg(SubTerrainBaseCfg):
    function = lane
    kind: str = "flat"
