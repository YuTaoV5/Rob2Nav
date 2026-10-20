"""Loopback-only MuJoCo interaction worker, independently supervised from Studio."""
import argparse
import io
import json
from pathlib import Path
import queue
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import uvicorn

try:
    from .handshake import PawSimulation, HandObservation
except ImportError:
    from handshake import PawSimulation, HandObservation


class Control(BaseModel):
    action: Literal['target','start','stop','withdraw','reset','push']
    position: list[float] | None = Field(default=None,min_length=3,max_length=3)
    leg: Literal['FL','FR'] | None = None


class Engine:
    def __init__(self,root):
        self.root=root;self.queue=queue.Queue(maxsize=32);self.lock=threading.Lock()
        self.state={'state':'INITIALIZING'};self.jpeg=b'';self.error=None
        self.last_client=time.monotonic();self.running=True
        threading.Thread(target=self.loop,daemon=True).start()

    def loop(self):
        try:
            s=PawSimulation(self.root);s.armed=False;s.requested_leg=1
            position=np.array([.37,-.13,.19]);visible=True;seq=0;push_until=-1.
            wall=time.monotonic();start_sim=float(s.data.time);next_render=0.;last_obs=-1.;last_push=0.
            while self.running:
                while not self.queue.empty():
                    cmd=self.queue.get_nowait()
                    if cmd.action=='reset':
                        s.close();s=PawSimulation(self.root);s.armed=False;s.requested_leg=1
                        position=np.array([.37,-.13,.19]);visible=True;seq=0;push_until=-1.
                        wall=time.monotonic();start_sim=0.;next_render=0.;last_obs=-1.
                    elif cmd.action=='target':
                        position=np.array(cmd.position);visible=True
                        if cmd.leg is not None:s.requested_leg=0 if cmd.leg=='FL' else 1
                    elif cmd.action=='start':
                        if s.state=='WAIT':
                            s.armed=True;s.hand_candidate_since=None;visible=True
                    elif cmd.action in ('stop','withdraw'):
                        s.armed=False
                        if cmd.action=='withdraw':visible=False
                    elif cmd.action=='push' and time.monotonic()-last_push>1:
                        push_until=float(s.data.time)+.16;last_push=time.monotonic()
                if time.monotonic()-self.last_client>2.5:
                    s.armed=False;visible=False
                # A browser drag controls a physical mocap hand. Only the hand is externally driven.
                now=float(s.data.time)
                if now-last_obs>=.025:
                    seq+=1;s.observe(HandObservation(position.tolist(),visible=visible,sequence=seq));last_obs=now
                s.data.xfrc_applied[s.base,:]=0
                if now<push_until:s.data.xfrc_applied[s.base,:3]=[0.,8.,0.]
                target_time=(time.monotonic()-wall)+start_sim
                if now<target_time:
                    s.step(min(5,max(1,int((target_time-now)/.01))))
                else:time.sleep(.002)
                if s.data.time>=next_render:
                    frame=s.render();image=Image.fromarray(frame);buf=io.BytesIO();image.save(buf,format='JPEG',quality=83)
                    camera=s.renderer.scene.camera
                    eye=(camera[0].pos+camera[1].pos)/2
                    forward=np.array(camera[0].forward);up=np.array(camera[0].up);right=np.cross(forward,up)
                    fy=640*camera[0].frustum_near/(camera[0].frustum_top-camera[0].frustum_bottom)
                    snap=s.snapshot();p=position
                    side_ok=(s.requested_leg==0 and p[1]>=.055) or (s.requested_leg==1 and p[1]<=-.055)
                    snap.update(target=position.tolist(),selected_leg='FL' if s.requested_leg==0 else 'FR',visible=visible,
                        workspace_valid=bool(.29<=p[0]<=.46 and .055<=abs(p[1])<=.23 and .12<=p[2]<=.27 and side_ok),
                        camera=dict(eye=eye.tolist(),forward=forward.tolist(),up=up.tolist(),right=right.tolist(),fy=float(fy),width=960,height=640),
                        events=s.events[-15:],hardware_enabled=False,simulation='MuJoCo 3.2.3 / free-base / rigid contacts')
                    with self.lock:self.state=snap;self.jpeg=buf.getvalue()
                    next_render=float(s.data.time)+.10
        except Exception as exc:
            import traceback
            traceback.print_exc();self.error=str(exc)


def create_app(root,assets):
    app=FastAPI(title='Go2 paw contact simulation')
    engine=Engine(root)

    @app.get('/')
    def page():return FileResponse(Path(assets)/'index.html',headers={'Cache-Control':'no-store'})

    @app.get('/api/state')
    def state():
        engine.last_client=time.monotonic()
        if engine.error:raise HTTPException(503,engine.error)
        with engine.lock:return dict(engine.state)

    @app.get('/api/frame.jpg')
    def frame():
        with engine.lock:payload=engine.jpeg
        if not payload:raise HTTPException(503,'Renderer warming up')
        return Response(payload,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    @app.post('/api/control',status_code=202)
    def control(cmd:Control):
        engine.last_client=time.monotonic()
        if cmd.action=='target':
            p=cmd.position
            if p is None or not np.isfinite(p).all() or not (.18<=p[0]<=.70 and -.35<=p[1]<=.35 and .06<=p[2]<=.35):
                raise HTTPException(422,'Invalid virtual hand position')
            with engine.lock:current=dict(engine.state)
            if cmd.leg and current.get('state') not in ('WAIT','SETTLE','COOLDOWN') and cmd.leg!=current.get('selected_leg'):
                raise HTTPException(409,'Retract before switching paws')
        if cmd.action=='start':
            with engine.lock:current=dict(engine.state)
            if current.get('state')!='WAIT':raise HTTPException(409,'Wait for a stable four-foot stance')
            if not current.get('workspace_valid'):raise HTTPException(422,'Hand is outside the selected paw workspace')
        try:engine.queue.put_nowait(cmd)
        except queue.Full:raise HTTPException(429,'Too many control updates')
        return {'accepted':True,'hardware_enabled':False}
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--assets',required=True);parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args();uvicorn.run(create_app(args.root,args.assets),host='127.0.0.1',port=args.port,log_level='warning')
