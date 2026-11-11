"""Allowlisted same-origin access to the separate voice controller on loopback."""
import asyncio
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

ROUTES = {'api/status':'GET', 'api/places':'GET', 'api/map':'GET',
          'api/command':'POST', 'api/audio':'POST', 'api/tts':'POST',
          'api/place':'POST', 'api/delete-place':'POST', 'api/edges':'POST'}


def mount_voice_control(app, root):
    assets = Path(root)/'voice_agent/web'

    @app.get('/voice')
    def redirect():
        return RedirectResponse('/voice/')

    @app.get('/voice/')
    @app.get('/voice/{asset}')
    def page(asset='index.html'):
        if asset not in ('index.html', 'app.js', 'style.css'):
            raise HTTPException(404)
        return FileResponse(assets/asset, headers={'Cache-Control':'no-store'})

    @app.api_route('/voice-api/{path:path}', methods=['GET', 'POST'])
    async def proxy(path: str, request: Request):
        if ROUTES.get(path) != request.method:
            raise HTTPException(404)
        try:
            port = int(os.environ.get('VOICE_AGENT_PORT', '8096'))
            if not 1024 <= port <= 65535: raise ValueError()
        except ValueError:
            raise HTTPException(503, 'Invalid VOICE_AGENT_PORT')
        length = request.headers.get('content-length')
        if length and (not length.isdigit() or int(length) > 2*1024*1024):
            raise HTTPException(413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 2*1024*1024: raise HTTPException(413)
        headers = {k:request.headers[k] for k in ('Content-Type', 'X-Control-Generation', 'X-Command-Id') if k in request.headers}
        token = os.environ.get('VOICE_AGENT_TOKEN')
        if token: headers['X-Voice-Token'] = token
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def fetch():
            # Live forwarding requires an explicit server-side deployment setting.
            if request.method == 'POST' and path != 'api/tts' and os.environ.get('VOICE_AGENT_ALLOW_LIVE_PROXY') != '1':
                req = urllib.request.Request(f'http://127.0.0.1:{port}/api/status', headers=headers)
                with opener.open(req, timeout=2) as status:
                    if json.loads(status.read(65536)).get('mode') != 'demo':
                        return 403, 'application/json', b'{"error":"Set VOICE_AGENT_ALLOW_LIVE_PROXY=1 after commissioning"}', {}
            req = urllib.request.Request(f'http://127.0.0.1:{port}/'+path,
                data=bytes(body) if request.method == 'POST' else None, method=request.method, headers=headers)
            try:
                response = opener.open(req, timeout=30)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                payload = response.read(16*1024*1024+1)
                if len(payload) > 16*1024*1024: raise ValueError('Voice response too large')
                timing = {k:response.headers[k] for k in ('X-TTS-Ms','X-TTS-Cache') if k in response.headers}
                return response.code, response.headers.get('Content-Type','application/json'), payload, timing
        try:
            status, kind, payload, timing = await asyncio.to_thread(fetch)
        except (OSError, ValueError, urllib.error.URLError):
            raise HTTPException(503, 'Voice service unavailable; start python -m voice_agent')
        return Response(payload, status_code=status, media_type=kind,
                        headers={'Cache-Control':'no-store', **timing})
