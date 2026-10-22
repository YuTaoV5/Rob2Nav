"""Chinese canonical manuals with stable legacy deep links."""
import json
from pathlib import Path


def aliases(root):
    path=Path(root)/'docs/文档路径映射.json'
    if not path.is_file():return {}
    data=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data,dict):raise ValueError('Invalid documentation mapping')
    for old,new in data.items():
        for value in (old,new):
            p=Path(value)
            if p.is_absolute() or '..' in p.parts or p.suffix!='.md':raise ValueError('Invalid documentation alias')
    return data


def canonical(root,path):
    return aliases(root).get(path,path)


def manual(root,skill,kind):
    old=f'skills/{skill}/{kind.upper()}.md'
    new=canonical(root,old)
    return new if (Path(root)/new).is_file() else old


def indexed_manual(reports,skill,kind):
    basename={'training':'训练手册','simulation':'仿真手册','deployment':'部署手册'}[kind]
    new=f'skills/{skill}/{basename}.md'
    return new if any(r['path']==new for r in reports) else f'skills/{skill}/{kind.upper()}.md'


def indexed_path(reports,path):
    return next((r['path'] for r in reports if r['path']==path or path in r.get('aliases',[])),path)
