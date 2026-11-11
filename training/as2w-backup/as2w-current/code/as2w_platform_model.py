"""Five-step diagnostic with a configurable final platform length."""
import xml.etree.ElementTree as ET


def load_platform_model(sim, seed, rise, platform_length):
    if platform_length <= 0.:
        raise ValueError("Platform length must be positive")
    path = sim.ROOT / "local-step-diagnostic/as2w_flat.xml"
    xml = ET.parse(path).getroot()
    for asset in xml.findall("./asset/mesh"):
        asset.set("file", (path.parent / asset.attrib["file"]).resolve().as_posix())
    world = xml.find("worldbody")
    ground = world.find("geom[@name='ground']")
    ground.set("group", "5")
    tread = (.31, .35, .39)[(seed // 3) % 3]
    rise = (.06, .08, .10)[seed % 3] if rise is None else float(rise)
    for level in range(1, 6):
        start = .7 + (level - 1) * tread
        length = tread if level < 5 else platform_length
        half_top = level * rise / 2.
        ET.SubElement(world, "geom", name=f"stair_{level}", type="box", group="5",
                      size=f"{length / 2.} 1.25 {half_top}", pos=f"{start + length / 2.} 0 {half_top}",
                      friction=ground.attrib["friction"])
    model = sim.mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    surfaces = [model.geom(name).id for name in ["ground", *[f"stair_{n}" for n in range(1, 6)]]]
    model.geom_priority[surfaces] = int(model.geom_priority.max()) + 1
    model.geom_contype[:] = 1
    model.geom_conaffinity[:] = 0
    model.geom_contype[surfaces] = 0
    model.geom_conaffinity[surfaces] = 1
    return model
