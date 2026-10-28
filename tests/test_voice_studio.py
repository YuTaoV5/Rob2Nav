import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from studio.voice_routes import mount_voice_control


class Reply(io.BytesIO):
    def __init__(self,data):
        super().__init__(json.dumps(data).encode());self.code=200
        self.headers={'Content-Type':'application/json'}


class VoiceStudioTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();mount_voice_control(app,Path(__file__).resolve().parents[1]);self.client=TestClient(app)

    def test_assets_and_proxy_allowlist(self):
        self.assertIn('语音任务台',self.client.get('/voice/').text)
        self.assertIn('API_PREFIX',self.client.get('/voice/app.js').text)
        self.assertEqual(self.client.get('/voice-api/api/command').status_code,404)
        self.assertEqual(self.client.post('/voice-api/arbitrary').status_code,404)

    def test_proxy_forwards_token_and_generation_without_browser_secret(self):
        seen=[]
        class Opener:
            def open(self,req,timeout):
                seen.append(req)
                return Reply({'mode':'demo'} if req.full_url.endswith('/api/status') else {'accepted':True})
        with patch.dict(os.environ,{'VOICE_AGENT_TOKEN':'server-test-token','VOICE_AGENT_PORT':'8096'},clear=True),patch('studio.voice_routes.urllib.request.build_opener',return_value=Opener()):
            result=self.client.post('/voice-api/api/audio',content=b'audio',headers={'Content-Type':'audio/wav','X-Control-Generation':'7','X-Command-Id':'sample'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(seen[-1].get_header('X-voice-token'),'server-test-token')
        self.assertEqual(seen[-1].get_header('X-control-generation'),'7')
        self.assertEqual(seen[-1].data,b'audio')
        self.assertNotIn('server-test-token',result.text)

    def test_live_control_requires_server_opt_in(self):
        seen=[]
        class Opener:
            def open(self,req,timeout):seen.append(req);return Reply({'mode':'live'})
        with patch.dict(os.environ,{},clear=True),patch('studio.voice_routes.urllib.request.build_opener',return_value=Opener()):
            result=self.client.post('/voice-api/api/command',json={'text':'启用控制'})
        self.assertEqual(result.status_code,403);self.assertEqual(len(seen),1)

    def test_body_limit_and_port_validation(self):
        self.assertEqual(self.client.post('/voice-api/api/audio',content=b'x'*(2*1024*1024+1)).status_code,413)
        with patch.dict(os.environ,{'VOICE_AGENT_PORT':'22'}):
            self.assertEqual(self.client.get('/voice-api/api/status').status_code,503)


if __name__=='__main__':unittest.main()
