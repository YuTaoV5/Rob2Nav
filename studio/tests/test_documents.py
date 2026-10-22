import json
from pathlib import Path
import pytest
from studio.catalog import Catalog
from studio.documents import aliases,indexed_manual,indexed_path
from scripts.check_documentation import audit


def test_chinese_manual_is_indexed_once_and_old_bookmark_resolves(tmp_path):
    p=tmp_path/'skills/backflip';p.mkdir(parents=True)
    (p/'训练手册.md').write_text('# 后空翻训练手册\n正文',encoding='utf-8')
    (p/'TRAINING.md').write_text('# 兼容入口',encoding='utf-8')
    (tmp_path/'docs').mkdir();(tmp_path/'docs/文档路径映射.json').write_text(json.dumps({'skills/backflip/TRAINING.md':'skills/backflip/训练手册.md'}),encoding='utf-8')
    reports=Catalog(tmp_path).get()['reports']
    assert len(reports)==1
    assert reports[0]['aliases']==['skills/backflip/TRAINING.md']
    assert indexed_path(reports,'skills/backflip/TRAINING.md')=='skills/backflip/训练手册.md'
    assert indexed_manual(reports,'backflip','training')=='skills/backflip/训练手册.md'


def test_alias_cannot_escape_repository(tmp_path):
    (tmp_path/'docs').mkdir();(tmp_path/'docs/文档路径映射.json').write_text(json.dumps({'old.md':'../outside.md'}),encoding='utf-8')
    with pytest.raises(ValueError):aliases(tmp_path)


def test_audit_rejects_host_paths_missing_links_and_invalid_cli_options(tmp_path):
    p=tmp_path/'skills/example';p.mkdir(parents=True)
    (tmp_path/'skills/example/train.py').write_text("p.add_argument('--valid')\n",encoding='utf-8')
    (p/'训练手册.md').write_text('# 测试\n/root/server/project\n[坏链接](missing.md)\n```bash\n"$ISAAC_PY" -m skills.example.train --invented 1\n```\n',encoding='utf-8')
    report=audit(tmp_path)
    assert {'machine_path','missing_link','unknown_option'} <= {x['kind'] for x in report['issues']}
