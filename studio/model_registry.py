"""Model inventory without deserializing executable model formats."""
from pathlib import Path
from urllib.parse import quote
from .catalog import ident, read_json

SKILLS=('stairs','platform','multiterrain','backflip','sideflip','upright_walk','navigation')
FORMATS={'.pt','.onnx','.zip'}

def asset_path(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    allowed=[root/'artifacts',root/'outputs',root/'skills',root/'vendor/RoboGauge/resources/models/go2']
    if (not any(p.is_relative_to(base.resolve()) for base in allowed) or
        any(part.startswith('.') for part in Path(relative).parts) or
        p.suffix.lower() not in FORMATS or not p.is_file()):
        raise ValueError('Not a published model asset')
    return p

def inventory(root,existing,evaluations):
    root=Path(root).resolve();rows={m['path']:dict(m) for m in existing}
    index=read_json(root/'artifacts/release_20261021/model_index.json',{}) or {}
    declared={m['path']:m for m in index.get('models',[])}
    declared.update({m['path']:m for m in (read_json(root/'studio/model_assets.json',{}) or {}).get('models',[])})
    paths=set(declared)
    for base in [root/'skills',root/'vendor/RoboGauge/resources/models/go2']:
        if base.exists():
            paths.update(p.relative_to(root).as_posix() for p in base.rglob('*')
                         if p.is_file() and p.suffix.lower() in FORMATS)
    for relative in paths:
        try:p=asset_path(root,relative)
        except ValueError:continue
        rows.setdefault(relative,dict(id=ident(p),name=p.name,group=p.parent.name,path=relative,bytes=p.stat().st_size))
    jobs={}
    from .jobs import Jobs
    from .telemetry import checkpoints
    # Reading an empty inventory must not create a queue database.
    if (root/'outputs/studio/studio.sqlite3').exists():
        for j in Jobs(root).all():
            w=checkpoints(root,j)
            for c in w['candidates']:
                jobs[c['path']]=(j,w,c)
                p=root/c['path'];rows.setdefault(c['path'],dict(id=ident(p.resolve()),name=p.name,group=j['name'],path=c['path'],bytes=c['bytes']))
    missing=[dict(path=rel,reason='模型索引列出的文件不存在或不在发布范围') for rel in declared if rel not in rows]
    for rel,m in rows.items():
        d=declared.get(rel,{})
        skill=d.get('skill')
        if skill not in SKILLS:skill=next((s for s in SKILLS if s in Path(rel).parts),None)
        role='archive';basis=d.get('selection_basis','历史检查点或导出文件；未自动选为最佳')
        if '/selected/' in rel or rel=='skills/navigation/policy/navigation.zip' :role='selected'
        elif '/manual_reproduction/' in rel:role='reproduction'
        elif '/representative_failed_baseline/' in rel:role='failed_baseline'
        elif rel.endswith('/go2_moe_cts_137k_0.6739.pt'):role='dependency';basis='实时运动策略的基础依赖，并非该技能训练成果'
        role=d.get('role',role)
        if rel in jobs:
            j,w,c=jobs[rel];skill=j['config'].get('capability_id') or skill
            m.update(job_id=j['id'],job_name=j['name'],iteration=c['iteration'])
            if w.get('selected',{}).get('path')==rel:role='queue_selected';basis=w['criterion']
        digest=d.get('sha256')
        evidence=[dict(label=e['name'],url='#simulation?evaluation='+e['id']) for e in evaluations
                  if digest and e.get('checkpoint')==digest]
        from .documents import manual
        manuals=[dict(label=label,url='#reports?path='+quote(manual(root,skill,kind),safe=''))
                 for label,kind in [('仿真手册','simulation'),('部署手册','deployment')]
                 if skill and (root/manual(root,skill,kind)).is_file()]
        m.update(skill=skill,format=Path(rel).suffix[1:].upper(),role=role,selection_basis=basis,
                 validation=d.get('validation','未关联到该文件哈希的独立评估；不自动认证'),
                 sha256=digest,checksum_source='发布索引记录' if digest else None,
                 hardware_qualified=False,evidence=evidence,manuals=manuals,
                 resources=[dict(x,url='/api/files/'+quote(x['path'],safe='/')) for x in d.get('resources',[])],
                 download_url='/api/model-assets/'+m['id']+'/download')
    return sorted(rows.values(),key=lambda m:(m['role'] not in ('selected','queue_selected','dependency','reproduction','failed_baseline'),m.get('skill') or '',m['path'])),missing
