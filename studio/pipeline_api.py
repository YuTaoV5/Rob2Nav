"""Tolerate short service restarts for reads; never blindly retry job creation."""
import json
import time
import urllib.error
import urllib.request


def request(path, data=None, base='http://127.0.0.1:8765/api/'):
    retryable=data is None or path=='preflight'
    delays=(1,2,4,8,15,15) if retryable else ()
    for attempt in range(len(delays)+1):
        req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type':'application/json','X-Studio-Request':'1'})
        try:
            with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (502,503,504) or attempt==len(delays):raise
        except (urllib.error.URLError,TimeoutError,ConnectionError):
            if attempt==len(delays):raise
        time.sleep(delays[attempt])
