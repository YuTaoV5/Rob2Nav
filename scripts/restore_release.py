"""Download, SHA-256 verify and safely restore independently packaged release data."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
from urllib.parse import quote,urlparse
from urllib.request import Request,urlopen,HTTPRedirectHandler,build_opener


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        next_request=super().redirect_request(req,fp,code,msg,headers,newurl)
        if next_request and urlparse(newurl).hostname!='api.github.com':next_request.remove_header('Authorization')
        return next_request


def request(url,binary=False):
    headers={'User-Agent':'Rob2Nav-restore','Accept':'application/octet-stream' if binary else 'application/vnd.github+json'}
    if urlparse(url).hostname=='api.github.com' and os.environ.get('GITHUB_TOKEN'):headers['Authorization']='Bearer '+os.environ['GITHUB_TOKEN']
    return build_opener(SafeRedirect).open(Request(url,headers=headers),timeout=120)


def download(asset,folder,expected=None):
    name=asset['name']
    if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Invalid asset filename')
    path=folder/name
    digest=expected or str(asset.get('digest','')).removeprefix('sha256:')
    if path.is_file() and path.stat().st_size==asset['size'] and digest and sha(path)==digest:return path
    temp=path.with_name(path.name+'.download')
    url=asset['url'] if os.environ.get('GITHUB_TOKEN') else asset['browser_download_url']
    print('DOWNLOAD',name,asset['size'],flush=True)
    with request(url,binary=True) as r,temp.open('wb') as f:shutil.copyfileobj(r,f,4*1024**2)
    if temp.stat().st_size!=asset['size'] or not digest or sha(temp)!=digest:raise ValueError('Download checksum/size mismatch: '+name)
    temp.replace(path);return path


def extract_verified(archive,entries,destination,overwrite=False):
    expected={x['path']:x for x in entries};seen=set();destination=destination.resolve()
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            parts=PurePosixPath(member.name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in member.name or not member.isfile():raise ValueError('Unsafe archive member: '+member.name)
            if member.name not in expected or member.name in seen:raise ValueError('Unexpected/duplicate archive member: '+member.name)
            item=expected[member.name];target=(destination/member.name).resolve()
            if not target.is_relative_to(destination):raise ValueError('Archive escape')
            if member.size!=item['bytes']:raise ValueError('Archive member size mismatch')
            seen.add(member.name)
            if target.is_file() and sha(target)==item['sha256']:continue
            if target.exists() and not overwrite:raise ValueError('Existing file differs; use a clean destination or explicit --overwrite: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.restore-tmp')
            if temporary.is_symlink():raise ValueError('Unsafe temporary path')
            stream=t.extractfile(member)
            with temporary.open('wb') as f:shutil.copyfileobj(stream,f,4*1024**2)
            if sha(temporary)!=item['sha256']:raise ValueError('Archive member checksum mismatch: '+member.name)
            temporary.replace(target)
    if seen!=set(expected):raise ValueError('Archive has missing files')


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='YuTaoV5/Rob2Nav');p.add_argument('--tag',default='full-archive-2026-10-19')
    p.add_argument('--destination',type=Path,required=True,help='Parent of the Rob2Nav clone, not the clone itself')
    p.add_argument('--download-dir',type=Path);p.add_argument('--group',action='append');p.add_argument('--list',action='store_true')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--verify-restored',action='store_true');a=p.parse_args()
    with request('https://api.github.com/repos/'+a.repo+'/releases/tags/'+quote(a.tag)) as r:release=json.load(r)
    assets={x['name']:x for x in release['assets']}
    cache=a.download_dir or a.destination/'.rob2nav-downloads';cache.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(download(assets['data-manifest.json'],cache).read_text(encoding='utf8'))
    groups=set(a.group or manifest['groups'])
    if groups-set(manifest['groups']):raise SystemExit('Unknown data group')
    if a.list:
        print(json.dumps(manifest['groups'],ensure_ascii=False,indent=2));return
    files=[x for x in manifest['files'] if x['group'] in groups]
    if a.verify_restored:
        bad=[x['path'] for x in files if not (a.destination/x['path']).is_file() or sha(a.destination/x['path'])!=x['sha256']]
        if bad:raise SystemExit('Missing or changed files: '+str(bad[:20]))
        print('RESTORED_FILES_VERIFIED',len(files));return
    for part in manifest['archives']:
        if part['group'] not in groups:continue
        path=download(assets[part['name']],cache,part['sha256'])
        extract_verified(path,[x for x in files if x['archive']==part['name']],a.destination,a.overwrite)
        print('RESTORED',part['name'],flush=True)
    print('RESTORE_COMPLETE',len(files),'files',flush=True)


if __name__=='__main__':main()
