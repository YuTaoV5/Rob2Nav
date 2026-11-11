#!/usr/bin/env python3
"""Actual TTS -> WAV -> ASR -> demo FSM roundtrip; refuses a live backend."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8096')
    p.add_argument('--output',type=Path,default=Path('data/voice/roundtrip.json'));a=p.parse_args()
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path,data=None,extra=None,raw=False):
        headers={'X-Voice-Token':os.environ.get('VOICE_AGENT_TOKEN','')}
        if isinstance(data,dict):data=json.dumps(data).encode();headers['Content-Type']='application/json'
        headers.update(extra or {})
        with client.open(urllib.request.Request(a.url+path,data=data,headers=headers),timeout=30) as response:
            result=response.read()
        return result if raw else json.loads(result)
    status=request('/api/status')
    if status['mode']!='demo':p.error('This test only operates the synthetic demo backend')
    if status['state']!='DISABLED':p.error('Start a fresh demo service with a new places file')
    rows=[]
    def save():
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(scope='real ASR/TTS with synthetic task backend; no robot or policy inference',
            results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    for phrase in ('开始控制。','去会议室。','表演一个后空翻动作。','表演一个侧空翻动作。','表演一个倒立动作。','关闭控制。'):
        if request('/api/status')['mode']!='demo':raise RuntimeError('Backend changed')
        audio=request('/api/tts',{'text':phrase},raw=True)
        generation=request('/api/status')['generation']
        result=request('/api/audio',audio,{'Content-Type':'audio/wav',
            'X-Control-Generation':str(generation),'X-Command-Id':str(uuid.uuid4())})
        rows.append(dict(phrase=phrase,**result))
        save()
        if not result['result']['accepted']:raise RuntimeError('Rejected: '+json.dumps(result,ensure_ascii=False))
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            status=request('/api/status')
            if status['state'] in ('IDLE','DISABLED'):break
            if status['state']=='FAULT':raise RuntimeError(json.dumps(status,ensure_ascii=False))
            time.sleep(.1)
        else:raise TimeoutError(phrase)
        rows[-1]['settled_state']=status['state']
        save()
    print(json.dumps(rows,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
