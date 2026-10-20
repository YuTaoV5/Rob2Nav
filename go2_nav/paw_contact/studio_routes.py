"""Same-origin Studio proxy to the isolated, loopback-only MuJoCo worker."""
import asyncio
from pathlib import Path
import urllib.error
import urllib.request

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response


def mount_handshake(app, root):
    @app.get('/handshake')
    def page():
        return FileResponse(Path(root)/'go2_nav/paw_contact/index.html',headers={'Cache-Control':'no-store'})

    @app.api_route('/handshake-api/{path:path}',methods=['GET','POST'])
    async def proxy(path:str, request:Request):
        methods={'api/state':'GET','api/frame.jpg':'GET','api/control':'POST'}
        if methods.get(path)!=request.method:raise HTTPException(404)
        body=await request.body()
        if len(body)>4096:raise HTTPException(413)
        def fetch():
            req=urllib.request.Request('http://127.0.0.1:8767/'+path,data=body if request.method=='POST' else None,
                method=request.method,headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=5) as r:return r.status,r.headers.get('Content-Type'),r.read()
            except urllib.error.HTTPError as e:return e.code,e.headers.get('Content-Type'),e.read()
        try:status,kind,payload=await asyncio.to_thread(fetch)
        except (OSError,urllib.error.URLError):raise HTTPException(503,'Paw simulation worker is unavailable')
        return Response(payload,status_code=status,media_type=kind,headers={'Cache-Control':'no-store'})
