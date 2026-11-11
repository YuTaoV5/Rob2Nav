"""Render checked flat cases with the exported R152 ONNX as the controller."""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import torch

import audit_flat_geometry as audit
from video import Video


if __name__=="__main__":
    assets,source,onnx_path,report,out=map(Path,sys.argv[1:6])
    torch.set_num_threads(1)
    audit.assets=assets
    audit.ev.sim.ROOT=assets
    audit.onnx_path=onnx_path
    audit.source=source
    rows=json.loads(report.read_text())
    out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for name in ("forward_slow","reverse","left","turn_left"):
        candidates=[row for row in rows if row["command"]==name and row["qualified"]]
        if candidates:
            selected.append(candidates[0])
    assert len(selected)>=2
    clips=[]
    records=[]
    for row in selected:
        name=row["command"]
        path=out/f"flat-{name}.mp4"
        v=Video(path,f"AS2-W R152 ONNX | {name.replace('_',' ')} then stop",row["seed"])
        audit.video=v
        try:
            trial=audit.evaluate((onnx_path,"fixed",name,row["target"],row["seed"],True))
        finally:
            v.close()
        if not trial["qualified"]:
            raise RuntimeError(f"Video rollout failed: {name}")
        clips.append(path.name)
        records.append(trial)
    listing=out/"clips.txt"
    listing.write_text("".join(f"file '{name}'\n" for name in clips))
    subprocess.run(["ffmpeg","-nostdin","-y","-loglevel","error","-f","concat","-safe","0","-i",str(listing),
                    "-c","copy","-movflags","+faststart",str(out/"flat-commands.mp4")],check=True)
    (out/"video-rollouts.json").write_text(json.dumps(records,indent=2,default=lambda v:v.item()))
    print("rendered",len(clips),flush=True)
