"""Summarize two rolling reports that refer to the same checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(checkpoint, mujoco, isaac):
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contracts = {
        "mujoco": read(mujoco / "mujoco-contract.json"),
        "isaac": read(isaac / "contract.json"),
    }
    for backend, contract in contracts.items():
        if contract.get("checkpoint_sha256") != sha256:
            raise ValueError(f"{backend}: checkpoint SHA-256 does not match")
    mj = read(mujoco / "mujoco-rolling-acceptance.json")
    lab = read(isaac / "speed-evaluation.json")
    if mj["cap_m_s"] != lab["cap_m_s"]:
        raise ValueError("The speed caps differ")
    return {
        "checkpoint_sha256": sha256,
        "speed_cap_m_s": mj["cap_m_s"],
        "mujoco": {key: mj[key] for key in ("gate", "safety_gate", "passed", "trials_per_case")},
        "isaac": {key: lab[key] for key in ("gate", "safety_gate", "summary", "turns")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("mujoco", type=Path)
    parser.add_argument("isaac", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.checkpoint, args.mujoco, args.isaac), indent=2))


if __name__ == "__main__":
    main()
