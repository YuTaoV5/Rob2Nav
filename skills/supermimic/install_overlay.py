"""Install the recorded overlay into a dedicated, pinned upstream checkout.

No dependency installation, process launch, or changes to an existing Rob2Nav tree.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

REVISION = "054a13c13285a1c94b2e69dc8ac7331f9a8c71cf"
HERE = Path(__file__).resolve().parent


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replace-overlay", action="store_true",
                        help="Replace only files owned by this overlay, after reviewing local edits")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Zhengtianjin/go2-supermimic.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    if run("git", "-C", str(source), "rev-parse", "HEAD") != REVISION:
        raise SystemExit("Wrong upstream revision. Use a new dedicated directory; do not reset a working tree.")
    patch = HERE / "resources/mjwarp-compatibility.patch"
    check = subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
    else:
        subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], check=True)
    files = [p for p in (HERE / "overlay").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for item in files:
        target = source / item.relative_to(HERE / "overlay")
        if target.exists() and target.read_bytes() != item.read_bytes() and not args.replace_overlay:
            raise SystemExit(f"Local overlay differs: {target}; inspect it before --replace-overlay")
    for item in files:
        target = source / item.relative_to(HERE / "overlay")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
    print(f"Installed {len(files)} files at {source}; upstream {REVISION}")


if __name__ == "__main__":
    main()
