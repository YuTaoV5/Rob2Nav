"""Run an allowlisted simulation, publishing actual MuJoCo frames in wall time."""
import argparse
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
os.environ.setdefault('MUJOCO_GL', 'egl')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--module',required=True);p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.module not in ('rob2nav.mujoco_skill','skills.backflip.simulate','go2_nav.terrain_run','rob2nav.deployment.verify_mujoco','studio.jump_demo'):
        raise ValueError('Unknown simulation module')
    import mujoco
    from PIL import Image
    from .sim_visuals import reference_ground
    stream=a.stream;stream.mkdir(parents=True,exist_ok=True)
    original=mujoco.mj_step;renderer=None;camera=mujoco.MjvCamera()
    camera.distance=2.2;camera.azimuth=130.;camera.elevation=-25.
    start_wall=None;start_sim=None;next_frame=0.;count=0;last=None;last_model=None
    def atomic(name,payload):
        tmp=stream/(name+'.tmp');tmp.write_bytes(payload);tmp.replace(stream/name)
    def step(model,data,*args,**kw):
        nonlocal renderer,start_sim,start_wall,next_frame,count,last,last_model
        result=original(model,data,*args,**kw)
        if model is not last_model:
            if renderer is not None:renderer.close();renderer=None
            last_model=model;start_sim=None;next_frame=0.
        if start_sim is None:start_sim=float(data.time);start_wall=time.monotonic()
        last=(model,data)
        elapsed=float(data.time)-start_sim
        if elapsed>=next_frame:
            try:settings=json.loads((stream/'camera.json').read_text())
            except (OSError,ValueError):settings={}
            camera.azimuth=settings.get('azimuth',130.);camera.elevation=settings.get('elevation',-25.)
            camera.distance=settings.get('distance',2.2)
            camera.lookat[:]=data.qpos[:3]
            camera.lookat[0]+=settings.get('pan_x',0.);camera.lookat[1]+=settings.get('pan_y',0.)
            if renderer is None:
                model.vis.global_.offwidth=max(960,model.vis.global_.offwidth)
                model.vis.global_.offheight=max(640,model.vis.global_.offheight)
                renderer=mujoco.Renderer(model,height=640,width=960)
            renderer.update_scene(data,camera=camera)
            reference_ground(renderer.scene,model,data)
            out=io.BytesIO();Image.fromarray(renderer.render()).save(out,format='JPEG',quality=82)
            atomic('frame.jpg',out.getvalue());count+=1
            atomic('state.json',json.dumps(dict(phase='RUNNING',time=elapsed,frames=count,
                position=data.qpos[:3].tolist(),camera=settings,backend='MuJoCo',hardware_enabled=False)).encode())
            next_frame=elapsed+.1
        delay=elapsed-(time.monotonic()-start_wall)
        if delay>0:time.sleep(min(delay,.1))
        return result
    mujoco.mj_step=step
    sys.argv=[a.module]+(a.args[1:] if a.args[:1]==['--'] else a.args)
    try:
        runpy.run_module(a.module,run_name='__main__')
        # Preserve the final physical state for camera inspection; no further mj_step.
        while last is not None:
            model,data=last
            try:settings=json.loads((stream/'camera.json').read_text())
            except (OSError,ValueError):settings={}
            camera.azimuth=settings.get('azimuth',130.);camera.elevation=settings.get('elevation',-25.)
            camera.distance=settings.get('distance',2.2);camera.lookat[:]=data.qpos[:3]
            camera.lookat[0]+=settings.get('pan_x',0.);camera.lookat[1]+=settings.get('pan_y',0.)
            renderer.update_scene(data,camera=camera)
            reference_ground(renderer.scene,model,data)
            out=io.BytesIO();Image.fromarray(renderer.render()).save(out,format='JPEG',quality=82)
            atomic('frame.jpg',out.getvalue())
            atomic('state.json',json.dumps(dict(phase='COMPLETED',time=float(data.time)-start_sim,frames=count,
                position=data.qpos[:3].tolist(),camera=settings,backend='MuJoCo',hardware_enabled=False)).encode())
            time.sleep(.1)
    finally:
        mujoco.mj_step=original
        if renderer is not None:renderer.close()


if __name__=='__main__':main()
