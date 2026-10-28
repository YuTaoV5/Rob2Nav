"""Local dashboard/API; recognition never holds the robot controller lock."""
import argparse
import hmac
import json
import os
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .backends import MockBackend, ProcessBackend
from .controller import Controller
from .places import Places
from .speech import Speech

ROOT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, places, backend, speech=None, pcd=None, z_min=-float('inf'), z_max=float('inf'), audit=None):
        self.places, self.backend = places, backend
        self.speech = speech or Speech(); self.audit_lock = threading.Lock()
        self.audit_file = None
        if audit:
            path = Path(audit); path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_file = path.open('a', encoding='utf-8', buffering=1)
        self.controller = Controller(places, backend, audit=self.log)
        self.pcd = pcd; self.preview = dict(points=[], note='未加载 PCD；示例地点用于调度演示')
        if pcd:
            from .pcd import projection, file_sha256
            if places.data.get('pcd_sha256') != file_sha256(pcd):
                raise ValueError('PCD SHA256 与标注文件不同；先用 init_map.py 建立新地图标注文件')
            self.preview = projection(pcd, z_min, z_max)
        self.running = threading.Event(); self.thread = None

    def log(self, event):
        if self.audit_file:
            with self.audit_lock: self.audit_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')

    def start(self):
        self.running.set()
        def loop():
            while self.running.is_set():
                try: self.controller.tick()
                except Exception as exc: self.log({'kind': 'tick_error', 'error': str(exc)})
                time.sleep(.05)
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start()

    def close(self):
        self.running.clear()
        if self.thread: self.thread.join(timeout=2)
        self.backend.close()
        if self.audit_file: self.audit_file.close()


def make_handler(app, token=''):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass  # audio and tokens are never logged

        def reply(self, status, data, kind='application/json; charset=utf-8', headers=None):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if kind.startswith('application/json') else data
            self.send_response(status); self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in (headers or {}).items(): self.send_header(key, str(value))
            self.end_headers(); self.wfile.write(raw)

        def authorize(self):
            host = self.headers.get('Host', '')
            if not token and urlparse('//'+host).hostname not in ('localhost', '127.0.0.1', '::1'):
                raise PermissionError('非法 Host；远程部署必须配置访问令牌')
            if token and not hmac.compare_digest(self.headers.get('X-Voice-Token', ''), token):
                raise PermissionError('需要有效的 X-Voice-Token')
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != host: raise PermissionError('拒绝跨站控制请求')

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path in ('/', '/app.js', '/style.css'):
                    name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                    kind = {'/': 'text/html; charset=utf-8', '/app.js': 'text/javascript; charset=utf-8', '/style.css': 'text/css; charset=utf-8'}[path]
                    return self.reply(200, (ROOT/'voice_agent/web'/name).read_bytes(), kind)
                self.authorize()
                if path == '/api/status':
                    status = app.controller.snapshot()
                    return self.reply(200, dict(**status, speech=app.speech.info(),
                        robot_map_position=app.places.map_position(status['health'].get('position'))))
                if path == '/api/places': return self.reply(200, app.places.snapshot())
                if path == '/api/map': return self.reply(200, app.preview)
                self.reply(404, {'error': '未找到'})
            except PermissionError as exc: self.reply(403, {'error': str(exc)})
            except Exception as exc: self.reply(400, {'error': str(exc)})

        def do_POST(self):
            try:
                self.authorize(); path = urlparse(self.path).path
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2*1024*1024: raise ValueError('请求体需在 1–2 MiB 内')
                kind = self.headers.get('Content-Type', '').split(';')[0]
                if path == '/api/audio' and kind != 'audio/wav': raise ValueError('语音需要 audio/wav')
                if path != '/api/audio' and kind != 'application/json': raise ValueError('需要 application/json')
                body = self.rfile.read(length)
                if len(body) != length: raise ValueError('请求被截断')
                if path == '/api/audio':
                    generation = int(self.headers['X-Control-Generation'])
                    command_id = self.headers.get('X-Command-Id')
                    recognized = app.speech.transcribe(body)
                    started = time.perf_counter()
                    result = app.controller.submit(recognized['text'], command_id, generation, final=True)
                    return self.reply(200, dict(**recognized, result=result, dispatch_ms=(time.perf_counter()-started)*1000))
                data = json.loads(body)
                if not isinstance(data, dict): raise ValueError('需要 JSON 对象')
                if path == '/api/command':
                    return self.reply(200, app.controller.submit(data.get('text'), data.get('command_id'),
                        data.get('generation'), data.get('final', True)))
                if path == '/api/place':
                    # Freeze annotations during a task; a route snapshot is never silently retargeted.
                    with app.controller.lock:
                        if app.controller.state not in ('IDLE', 'DISABLED'): raise ValueError('停止任务后再编辑标注')
                        place = app.places.put(data); app.controller.generation += 1
                        return self.reply(200, place)
                if path == '/api/delete-place':
                    with app.controller.lock:
                        if app.controller.state not in ('IDLE', 'DISABLED'): raise ValueError('停止任务后再编辑标注')
                        app.places.remove(data['id']); app.controller.generation += 1
                        return self.reply(200, {'ok': True})
                if path == '/api/edges':
                    with app.controller.lock:
                        if app.controller.state not in ('IDLE', 'DISABLED'): raise ValueError('停止任务后再编辑路线')
                        app.places.set_edges(data['edges']); app.controller.generation += 1
                        return self.reply(200, {'ok': True})
                if path == '/api/tts':
                    audio, timing = app.speech.synthesize(data['text'])
                    return self.reply(200, audio, 'audio/wav', {'X-TTS-Ms': round(timing['tts_ms'], 2), 'X-TTS-Cache': str(timing['cache_hit']).lower()})
                self.reply(404, {'error': '未找到'})
            except PermissionError as exc: self.reply(403, {'error': str(exc)})
            except (ValueError, KeyError, RuntimeError, OSError) as exc: self.reply(400, {'error': str(exc)})
    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['demo', 'live'], default='demo')
    p.add_argument('--places', type=Path, default=ROOT/'data/voice/places.json')
    p.add_argument('--pcd', type=Path); p.add_argument('--z-min', type=float, default=-float('inf')); p.add_argument('--z-max', type=float, default=float('inf'))
    p.add_argument('--speech-config', type=Path); p.add_argument('--bridge-command', type=Path)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8096)
    args = p.parse_args()
    token = os.environ.get('VOICE_AGENT_TOKEN', '')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token: p.error('远程监听需要环境变量 VOICE_AGENT_TOKEN')
    initial = json.loads((ROOT/'configs/voice.places.demo.json').read_text(encoding='utf-8'))
    if args.mode == 'live' and not args.places.exists(): p.error('实机必须先通过 init_map.py 创建真实地图标注')
    places = Places(args.places, initial)
    if not args.places.exists(): places.save()
    # Model loading/prewarming must finish before the live worker heartbeat starts.
    speech = Speech(json.loads(args.speech_config.read_text(encoding='utf-8')) if args.speech_config else None, ROOT)
    replies=[prefix+place['name']+suffix for place in places.data['places'][:10]
             for prefix,suffix in [('准备前往','。'),('正在前往','。'),('已到达','并站稳。')]]
    replies += [prefix+name+'。' for name in ('后空翻','侧空翻','倒立') for prefix in ('准备执行','正在执行')]
    speech.prewarm(replies)
    if args.mode == 'live':
        if not args.bridge_command: p.error('实机需要 --bridge-command，且底层资格校验必须通过')
        if places.data['map_id'].startswith('demo'): p.error('实机不能使用示例地图')
        backend = ProcessBackend(json.loads(args.bridge_command.read_text(encoding='utf-8')))
    else: backend = MockBackend(places.data['map_id'])
    app = Application(places, backend, speech, args.pcd, args.z_min, args.z_max, ROOT/'data/voice/events.jsonl')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app, token)); server.daemon_threads = True
    app.start()
    print('Voice dashboard: http://%s:%s | mode=%s (demo does not run policies)' % (args.host, args.port, args.mode), flush=True)
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally: server.server_close(); app.close()


if __name__ == '__main__': main()
