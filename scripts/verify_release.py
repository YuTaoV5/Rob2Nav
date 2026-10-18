"""Verify every compressed archive and member without unpacking the data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from package_release import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.directory/'data-manifest.json').read_text(encoding='utf8'));verified=[]
    for part in m['archives']:
        path=a.directory/part['name']
        if path.stat().st_size!=part['bytes'] or digest(path)!=part['sha256']:raise SystemExit('Archive differs: '+part['name'])
        members={f['path']:f for f in m['files'] if f['archive']==part['name']};seen=set()
        with tarfile.open(path,'r:gz') as t:
            for member in t:
                entry=members.get(member.name)
                if not member.isfile() or entry is None or member.name in seen or member.size!=entry['bytes']:raise SystemExit('Invalid member: '+member.name)
                h=hashlib.sha256()
                with t.extractfile(member) as f:
                    for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
                if h.hexdigest()!=entry['sha256']:raise SystemExit('Member differs: '+member.name)
                seen.add(member.name)
        if seen!=set(members):raise SystemExit('Missing members: '+part['name'])
        verified.append({'archive':part['name'],'files':len(seen),'sha256':part['sha256']});print('VERIFIED_MEMBERS',part['name'],len(seen),flush=True)
    report={'all_verified':True,'file_count':sum(x['files'] for x in verified),'archives':verified}
    (a.directory/'archive-verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('ALL_MEMBERS_VERIFIED',report['file_count'],flush=True)


if __name__=='__main__':main()
