"""Publish release data using existing Git credential storage, never token files.

Large uploads can run on the data host via SSH; token crosses only encrypted stdin.
"""
import argparse
import getpass
import hashlib
import http.client
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request,urlopen


def api(repo,token,path,body=None,method=None):
    req=Request('https://api.github.com/repos/'+repo+path,data=None if body is None else json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'Rob2Nav-release','X-GitHub-Api-Version':'2022-11-28'},method=method)
    try:
        with urlopen(req,timeout=90) as r:return json.load(r) if r.status!=204 else {}
    except HTTPError as e:
        if e.code==404:return None
        raise RuntimeError('GitHub API failed: HTTP '+str(e.code)) from None


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def upload(repo,token,rid,path):
    conn=http.client.HTTPSConnection('uploads.github.com',timeout=900)
    conn.putrequest('POST',f'/repos/{repo}/releases/{rid}/assets?name='+quote(path.name))
    for k,v in {'Authorization':'Bearer '+token,'Content-Type':'application/octet-stream','Content-Length':str(path.stat().st_size),'User-Agent':'Rob2Nav-release','Accept':'application/vnd.github+json'}.items():conn.putheader(k,v)
    conn.endheaders()
    try:
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''):conn.send(block)
        response=conn.getresponse();body=response.read()
        if response.status!=201:raise RuntimeError('Asset upload HTTP '+str(response.status))
        return json.loads(body)
    finally:conn.close()


def worker(repo,rid,directory,token):
    out=Path(directory);manifest=json.loads((out/'data-manifest.json').read_text(encoding='utf8'))
    names=[x['name'] for x in manifest['archives']]+['data-manifest.json','SHA256SUMS','release-summary.json','archive-verification.json']
    names += [p.name for p in sorted(out.glob('environment-*')) if p.is_file()]
    verified=[]
    for name in names:
        path=out/name;expected=sha(path)
        for attempt in range(4):
            existing={a['name']:a for a in api(repo,token,f'/releases/{rid}/assets?per_page=100')}
            asset=existing.get(name)
            if asset and asset.get('state')=='uploaded' and asset['size']==path.stat().st_size and asset.get('digest')=='sha256:'+expected:break
            if asset:
                # Only incomplete artifacts from this exact release upload are replaced.
                if asset.get('state')=='uploaded':raise RuntimeError('Existing asset differs; refusing replacement: '+name)
                api(repo,token,'/releases/assets/'+str(asset['id']),method='DELETE')
            print('UPLOADING',name,path.stat().st_size,flush=True)
            try:
                asset=upload(repo,token,rid,path)
                if asset['size']!=path.stat().st_size or asset.get('digest')!='sha256:'+expected:raise RuntimeError('Uploaded checksum differs: '+name)
                break
            except Exception as e:
                print('RETRY',name,attempt+1,type(e).__name__,flush=True)
                if attempt==3:raise
                time.sleep(3)
        verified.append({k:asset.get(k) for k in ('name','size','digest','browser_download_url','id')})
        print('VERIFIED',name,asset['size'],flush=True)
        (out/'upload-verification.json').write_text(json.dumps(verified,indent=2))
    print('ALL_UPLOADS_VERIFIED',len(verified),sum(a['size'] for a in verified),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='YuTaoV5/Rob2Nav');p.add_argument('--tag',default='full-archive-2026-10-19')
    p.add_argument('--create',action='store_true');p.add_argument('--publish',action='store_true');p.add_argument('--inspect',action='store_true')
    p.add_argument('--notes',type=Path);p.add_argument('--target',default='main');p.add_argument('--upload-remote',action='store_true')
    p.add_argument('--host');p.add_argument('--port',type=int,default=22);p.add_argument('--user',default='root');p.add_argument('--remote-python',default='/root/gpufree-data/envs/rob2nav-studio/bin/python')
    p.add_argument('--remote-script',default='/root/gpufree-data/Rob2Nav/scripts/publish_release.py');p.add_argument('--directory',default='/root/gpufree-data/Rob2Nav-release-20261019')
    p.add_argument('--worker',type=int);a=p.parse_args()
    if a.worker:
        token=json.loads(sys.stdin.readline())['token'];worker(a.repo,a.worker,a.directory,token);return
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',capture_output=True,text=True,check=True)
    values=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line);token=values.get('password')
    if not token:raise SystemExit('No GitHub credential available')
    release=api(a.repo,token,'/releases/tags/'+quote(a.tag))
    if release is None:
        release=next((r for r in api(a.repo,token,'/releases?per_page=100') if r['tag_name']==a.tag),None)
    if a.create and release is None:
        if not a.notes:raise SystemExit('--notes is required')
        release=api(a.repo,token,'/releases',{'tag_name':a.tag,'target_commitish':a.target,'name':'Rob2Nav 完整实验归档与迁移部署 · 2026-10-19','body':a.notes.read_text(encoding='utf-8'),'draft':True,'prerelease':False})
    if not release:raise SystemExit('Release does not exist')
    rid=release['id']
    if a.upload_remote:
        import paramiko
        password=getpass.getpass('SSH password (not saved): ')
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(a.host,port=a.port,username=a.user,password=password,timeout=30);password=None
        cmd=shlex.join([a.remote_python,a.remote_script,'--repo',a.repo,'--worker',str(rid),'--directory',a.directory])+' 2>&1'
        stdin,stdout,stderr=client.exec_command(cmd);stdin.write(json.dumps({'token':token})+'\n');stdin.flush();stdin.channel.shutdown_write()
        for line in stdout:print(line.rstrip(),flush=True)
        status=stdout.channel.recv_exit_status();client.close()
        if status:raise SystemExit('Remote uploader failed: '+str(status))
    assets=api(a.repo,token,f'/releases/{rid}/assets?per_page=100')
    if a.publish:
        if not assets or any(x['state']!='uploaded' for x in assets):raise SystemExit('Incomplete release assets')
        release=api(a.repo,token,f'/releases/{rid}',{'draft':False},method='PATCH')
    print(json.dumps({'id':rid,'url':release['html_url'],'draft':release['draft'],'assets':[{k:v.get(k) for k in ('name','size','digest','browser_download_url')} for v in assets]},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
