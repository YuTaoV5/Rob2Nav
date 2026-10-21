"""Same-origin, allowlisted proxy to an isolated MuJoCo worker."""
import asyncio
from pathlib import Path
import urllib.error
import urllib.request
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response


def mount_navigation_live(app, root):
    @app.get('/scan-live')
    def scan_page():
        return FileResponse(Path(root)/'skills/navigation/scan.html',headers={'Cache-Control':'no-store'})

    @app.get('/navigation-live')
    def page():
        return FileResponse(Path(root)/'skills/navigation/index.html', headers={'Cache-Control': 'no-store'})

    @app.api_route('/scan-live-api/{path:path}', methods=['GET','POST'])
    @app.api_route('/navigation-live-api/{path:path}', methods=['GET', 'POST'])
    async def proxy(path: str, request: Request):
        if {'api/state': 'GET', 'api/frame.jpg': 'GET', 'api/control': 'POST'}.get(path) != request.method:
            raise HTTPException(404)
        body = await request.body()
        if len(body) > 4096:
            raise HTTPException(413)

        def fetch():
            port=8769 if request.url.path.startswith('/scan-live-api/') else 8768
            req = urllib.request.Request(f'http://127.0.0.1:{port}/'+path, data=body if request.method == 'POST' else None,
                method=request.method, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, r.headers.get('Content-Type'), r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get('Content-Type'), e.read()
        try:
            status, kind, payload = await asyncio.to_thread(fetch)
        except (OSError, urllib.error.URLError):
            raise HTTPException(503, 'Navigation physics worker is unavailable')
        return Response(payload, status_code=status, media_type=kind, headers={'Cache-Control': 'no-store'})
