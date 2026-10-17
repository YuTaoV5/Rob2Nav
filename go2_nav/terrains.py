"""Physical terrain geometry. Metadata is for evaluation, never a planner input."""
import xml.etree.ElementTree as ET

TERRAIN_SCENES = ("water", "stairs", "platform", "gap", "mixed")


def populate(world, scene, cfg):
    world.remove(world.find("geom[@name='floor']"))
    meta = {"water": [], "gaps": [], "goal_height": 0., "surfaces": []}

    def slab(name, x0, x1, top, width=3., y=0., color="0.60 0.64 0.66 1", friction=.8):
        bottom = -1.2
        ET.SubElement(world, "geom", name=name, type="box",
                      pos=f"{(x0+x1)/2} {y} {(top+bottom)/2}",
                      size=f"{(x1-x0)/2} {width} {(top-bottom)/2}",
                      rgba=color, friction=f"{friction} 0.02 0.01", group="0")
        meta["surfaces"].append([x0, x1, top])

    def water(x0, x1, y0, y1, z=0.):
        # Ideal RGB return: this coloured, low-friction risk patch is not a fluid model.
        ET.SubElement(world, "geom", name=f"water_{len(meta['water'])}", type="box",
                      pos=f"{(x0+x1)/2} {(y0+y1)/2} {z+.002}",
                      size=f"{(x1-x0)/2} {(y1-y0)/2} 0.003", rgba="0.03 0.24 0.92 1",
                      friction="0.08 0.005 0.001", priority="2", group="0")
        meta["water"].append([x0, x1, y0, y1])

    h = float(cfg.get("terrain_height", .08 if scene == "stairs" else .20))
    gap = float(cfg.get("gap_width", .30))
    depth = float(cfg.get("gap_depth", .8))
    if scene == "water":
        slab("ground", -2, 10, 0)
        water(2., 3.8, -.9, .9)
    elif scene == "stairs":
        start, run, count = 2.3, .38, 4
        slab("approach", -2, start, 0)
        for i in range(count):
            slab(f"stair_{i}", start+i*run, start+(i+1)*run, (i+1)*h, color="0.64 0.54 0.32 1")
        slab("landing", start+count*run, 10, count*h)
        meta["goal_height"] = count*h
    elif scene == "platform":
        slab("approach", -2, 3., 0)
        slab("platform", 3., 10, h, color="0.60 0.40 0.20 1")
        meta["goal_height"] = h
    elif scene == "gap":
        start = float(cfg.get("gap_start", 3.))
        slab("approach", -2, start, 0)
        slab("pit_bottom", start, start+gap, -depth, color="0.13 0.13 0.16 1")
        slab("landing", start+gap, 10, 0)
        meta["gaps"].append([start, start+gap, -depth])
    elif scene == "mixed":
        slab("approach", -2, 5.5, 0)
        water(1.1, 2.1, -.8, .8)
        slab("pit_bottom", 5.5, 5.5+gap, -depth, color="0.13 0.13 0.16 1")
        slab("landing", 5.5+gap, 7., 0)
        slab("stair_0", 7., 7.4, .08, color="0.64 0.54 0.32 1")
        slab("stair_1", 7.4, 9., .16, color="0.64 0.54 0.32 1")
        slab("platform", 9., 16, .16+h, color="0.60 0.40 0.20 1")
        meta["gaps"].append([5.5, 5.5+gap, -depth])
        meta["goal_height"] = .16+h
        meta["goal"] = [14., 0.]
    else:
        raise ValueError(scene)
    return meta
