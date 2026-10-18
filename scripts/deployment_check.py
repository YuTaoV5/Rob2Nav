"""Non-training deployment preflight. Run inside the selected runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['web','mujoco','isaac'],default='web');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    modules={'web':['fastapi','uvicorn','pydantic','psutil','tensorboard','numpy'],
             'mujoco':['mujoco','torch','gymnasium','stable_baselines3','imageio','yaml'],
             'isaac':['isaaclab.app','rsl_rl','torch']}[a.mode]
    missing=[]
    for name in modules:
        try:found=importlib.util.find_spec(name) is not None
        except (ModuleNotFoundError,ImportError):found=False
        if not found:missing.append(name)
    if a.mode=='web' and not (root/'studio/frontend/dist/index.html').is_file():missing.append('npm frontend build')
    for name in ('check_vendor.py','check_nju.py'):
        result=subprocess.run([sys.executable,str(root/'scripts'/name)],cwd=root,capture_output=True,text=True)
        if result.returncode:missing.append(name+': asset checks failed')
    print(json.dumps({'mode':a.mode,'python':sys.version,'executable':sys.executable,'repository':str(root),'missing':missing,
                      'note':'Import/asset preflight only; does not execute training, accept simulator terms or prove GPU rendering.'},indent=2))
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
