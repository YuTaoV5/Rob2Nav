import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import pytest
from scripts.restore_release import extract_verified

ROOT=Path(__file__).resolve().parents[1]


def test_archive_round_trip_and_conflict(tmp_path):
    data=tmp_path/'data';p=data/'Rob2Nav/outputs/example/model.pt';p.parent.mkdir(parents=True);p.write_bytes(b'test weights')
    (p.parent/'iterations.jsonl').write_text('{"iteration":1}\n')
    (p.parent/'.env').write_text('PRIVATE=do not publish')
    out=tmp_path/'release'
    subprocess.run([sys.executable,str(ROOT/'scripts/package_release.py'),'--data-root',str(data),'--output',str(out)],check=True,capture_output=True)
    m=json.loads((out/'data-manifest.json').read_text());assert m['total_files']==2
    assert any(x['reason']=='credential file' for x in m['excluded'])
    target=tmp_path/'restored'
    for part in m['archives']:
        entries=[x for x in m['files'] if x['archive']==part['name']]
        extract_verified(out/part['name'],entries,target)
        extract_verified(out/part['name'],entries,target)
    assert (target/'Rob2Nav/outputs/example/model.pt').read_bytes()==b'test weights'
    (target/'Rob2Nav/outputs/example/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing file differs'):extract_verified(out/m['archives'][0]['name'],m['files'],target)


def test_archive_rejects_escape_and_changed_hash(tmp_path):
    tar=tmp_path/'unsafe.tar.gz'
    with tarfile.open(tar,'w:gz') as t:
        info=tarfile.TarInfo('../escape');info.size=1;t.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'):extract_verified(tar,[],tmp_path/'target')
    with tarfile.open(tar,'w:gz') as t:
        info=tarfile.TarInfo('Rob2Nav/a');info.size=1;t.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='checksum mismatch'):
        extract_verified(tar,[{'path':'Rob2Nav/a','bytes':1,'sha256':'0'*64}],tmp_path/'target')


def test_packager_refuses_secret_like_content(tmp_path):
    folder=tmp_path/'data/Rob2Nav/outputs';folder.mkdir(parents=True)
    (folder/'unsafe.txt').write_text('ghp_'+'a'*40)
    r=subprocess.run([sys.executable,str(ROOT/'scripts/package_release.py'),'--data-root',str(tmp_path/'data'),'--output',str(tmp_path/'release')],capture_output=True,text=True)
    assert r.returncode!=0 and 'Secret-like content detected' in r.stderr
