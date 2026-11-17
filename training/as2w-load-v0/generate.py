"""Reproduce the AS2W load-v0 training URDF from the pinned vendor asset."""
import argparse
import hashlib
import io
from pathlib import Path
import xml.etree.ElementTree as ET

SOURCE_SHA256 = "05da624085b16c91a9223b77872d568d97c44e480d29f3ce6933a9554bfb7d2c"
RESULT_SHA256 = "7c9740a95042bad7b73cadfda90bdc7c1873ac766afbb7c0e7e492367d45afb0"

def generate(source: Path) -> Path:
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned official AS2W URDF. See README.md.")
    tree = ET.ElementTree(ET.fromstring(payload))
    inertial = tree.find("link[@name='base_link']/inertial")
    mass = inertial.find("mass")
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    mass.set("value", str(float(mass.get("value")) * 1.15))
    xyz = list(map(float, origin.get("xyz").split()))
    xyz[0] -= .015
    origin.set("xyz", " ".join(f"{value:.9f}" for value in xyz))
    for key, value in inertia.attrib.items():
        inertia.set(key, f"{1.15 * float(value):.12g}")

    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    result = buffer.getvalue()
    if hashlib.sha256(result).hexdigest() != RESULT_SHA256:
        raise ValueError("Generated URDF does not match the verified training asset.")
    destination = source.with_name("as2w-load-v0.urdf")
    with destination.open("xb") as output:
        output.write(result)
    return destination

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the official as2w.urdf")
    args = parser.parse_args()
    print(generate(args.source))
