import copy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_agent.backends import MockBackend
from voice_agent.places import Places
from voice_agent.server import Application,make_handler
from voice_agent.speech import encode_wav

ROOT=Path(__file__).resolve().parents[1]


class FakeSpeech:
    def __init__(self):self.started=threading.Event();self.release=threading.Event()
    def info(self):return {'asr':'fake-test','tts':'fake-test'}
    def transcribe(self,data):
        self.started.set();self.release.wait(5)
        return {'text':'去会议室','asr_ms':1,'audio_s':1}


class HttpTests(unittest.TestCase):
    def setUp(self):
        directory=ROOT/'generated/test-tmp';directory.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=directory);self.addCleanup(self.temp.cleanup)
        data=json.loads((ROOT/'configs/voice.places.demo.json').read_text(encoding='utf-8'))
        places=Places(Path(self.temp.name)/'places.json',data)
        self.fake=FakeSpeech();self.app=Application(places,MockBackend(data['map_id']),self.fake)
        self.http=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.app,token='test-session'))
        self.http.daemon_threads=True;self.url='http://127.0.0.1:'+str(self.http.server_address[1])
        self.thread=threading.Thread(target=self.http.serve_forever,kwargs={'poll_interval':.02},daemon=True);self.thread.start();self.app.start()
        self.addCleanup(self.cleanup)
        self.client=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def cleanup(self):
        self.fake.release.set();self.http.shutdown();self.http.server_close();self.app.close();self.thread.join(timeout=2)

    def request(self,path,data=None,headers=None):
        h={'X-Voice-Token':'test-session'};h.update(headers or {})
        if isinstance(data,dict):data=json.dumps(data).encode();h.setdefault('Content-Type','application/json')
        req=urllib.request.Request(self.url+path,data=data,headers=h)
        with self.client.open(req,timeout=5) as response:return json.load(response)

    def test_dashboard_assets_and_token(self):
        with self.client.open(self.url+'/') as response:self.assertIn('语音任务台',response.read().decode())
        with self.assertRaises(urllib.error.HTTPError) as error:self.request('/api/status',headers={'X-Voice-Token':'wrong'})
        self.assertEqual(error.exception.code,403)

    def test_place_edit_route_and_command_api(self):
        self.request('/api/command',{'text':'启用控制'})
        result=self.request('/api/place',dict(name='接待区',floor='一楼',xyz=[.3,.2,.4],yaw=0,aliases=['前台']))
        self.assertIn('id',result)
        self.assertEqual(self.request('/api/places')['places'][-1]['name'],'接待区')
        graph=self.request('/api/places')['edges']+[dict(**{'from':'home','to':result['id']},bidirectional=True)]
        self.request('/api/edges',{'edges':graph})
        result=self.request('/api/command',{'text':'去前台'})
        self.assertTrue(result['accepted']);self.assertEqual(result['state'],'NAV_PREPARING')

    def test_stop_not_blocked_by_asr_and_late_result_ignored(self):
        self.request('/api/command',{'text':'启用控制'})
        generation=self.request('/api/status')['generation'];result=[]
        def transcribe():
            result.append(self.request('/api/audio',encode_wav([0]*16000,16000),
                {'Content-Type':'audio/wav','X-Control-Generation':str(generation),'X-Command-Id':'old-audio'}))
        thread=threading.Thread(target=transcribe);thread.start();self.assertTrue(self.fake.started.wait(2))
        started=time.monotonic();stop=self.request('/api/command',{'text':'停止'})
        self.assertTrue(stop['accepted']);self.assertLess(time.monotonic()-started,1)
        self.fake.release.set();thread.join(timeout=3)
        self.assertFalse(result[0]['result']['accepted'])

    def test_cross_origin_and_plaintext_commands_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/command',{'text':'后空翻'},{'Origin':'https://untrusted.example'})
        self.assertEqual(error.exception.code,403)
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/command',b'{"text":"enable"}',{'Content-Type':'text/plain'})


if __name__=='__main__':unittest.main()
