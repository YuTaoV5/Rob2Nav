import hashlib
import json
from unittest.mock import Mock
import pytest
from scripts import acrobatics_handoff as h


def test_preflight_does_not_create_or_submit(tmp_path,monkeypatch):
    p=tmp_path/'config.json';p.write_text(json.dumps({'name':'preview'}))
    request=Mock(return_value={'valid':True});monkeypatch.setattr(h,'request',request)
    monkeypatch.setattr('sys.argv',['handoff','--config',str(p)])
    h.main()
    assert request.call_count==1 and request.call_args.args[0]=='preflight'


def test_same_name_different_config_cannot_silently_reuse(tmp_path,monkeypatch):
    source=h.PACK/'新服务器任务配置/04_关节增益配置烟测.json'
    config=json.loads(source.read_text(encoding='utf-8'));other=json.loads(json.dumps(config));other['iterations']=3
    p=tmp_path/'config.json';p.write_text(json.dumps(config))
    request=Mock(side_effect=[{'valid':True},[dict(id='existing',state='queued',name=config['name'],config=other)]])
    monkeypatch.setattr(h,'request',request);monkeypatch.setattr('sys.argv',['handoff','--config',str(p),'--submit'])
    with pytest.raises(RuntimeError,match='different configuration'):h.main()
    assert request.call_count==2


def test_resource_corruption_is_rejected(tmp_path,monkeypatch):
    p=tmp_path/'input.bin';p.write_bytes(b'wrong')
    (tmp_path/'资源清单.json').write_text(json.dumps({'files':[dict(path='input.bin',bytes=5,sha256=hashlib.sha256(b'right').hexdigest())]}))
    monkeypatch.setattr(h,'ROOT',tmp_path);monkeypatch.setattr(h,'PACK',tmp_path)
    with pytest.raises(ValueError,match='hash mismatch'):h.verify()
