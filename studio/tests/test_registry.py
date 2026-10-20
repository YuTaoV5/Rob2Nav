import json
import pytest
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.config import RunConfig,command
from studio.registry import registry,capabilities


def test_recipe_robot_algorithm_contract_and_external_gate(tmp_path):
    data=registry();native=next(r for r in data['recipes'] if r['id']=='moe_stairs')
    cfg=RunConfig(**native['defaults'])
    with pytest.raises(ValueError,match='Robot / algorithm'):
        RunConfig(**{**native['defaults'],'robot_id':'go1'})
    external=next(r for r in data['archived_recipes'] if r['id']=='wmp_parkour')
    with pytest.raises(ValueError,match='Unknown reviewed training recipe'):
        RunConfig(**external['defaults'])


def test_retention_recipe_preserves_observation_contract(tmp_path):
    r=next(r for r in registry()['recipes'] if r['id']=='moe_stairs')
    p=tmp_path/r['defaults']['checkpoint'];p.parent.mkdir(parents=True);p.write_bytes(b'weight')
    cfg=RunConfig(**r['defaults']);argv=command(cfg,tmp_path,tmp_path/'out',tmp_path/'config','python')
    assert '--navigation' not in argv and '--arrival-aware' not in argv
    assert cfg.warmup+cfg.fusion==cfg.iterations and cfg.mode=='student'
    with pytest.raises(ValueError,match='Terrain does not match'):
        command(RunConfig(**{**r['defaults'],'kind':'platform'}),tmp_path,tmp_path/'out',tmp_path/'config','python')


def test_capability_training_done_never_implies_success_and_diagnostics_not_scored():
    jobs=[dict(id='test',state='succeeded',name='completed',config={'capability_id':'stairs'})]
    ev={'id':'e','path':'outputs/studio/jobs/test/evaluation/stairs_0.50','kind':'stairs','diagnostic':False,
        'video':'outputs/studio/jobs/test/evaluation/stairs_0.50/episode_0.mp4','name':'stairs','successes':0,'total':32,'role':'感知学生','geometry':'标准'}
    diag={**ev,'id':'d','diagnostic':True}
    c=capabilities({'evaluations':[ev,diag],'videos':[]},jobs)
    skill=next(s for s in c['capabilities'] if s['id']=='stairs')
    assert len(skill['evaluations'])==1 and skill['evaluations'][0]['successes']==0
    assert skill['training_complete']==1 and skill['local_status']!='已通过'
    assert skill['local_videos'][0]['origin']=='local'


def test_registry_references_are_complete_and_pinned():
    d=registry();projects={p['id'] for p in d['projects']};recipes={r['id'] for r in d['recipes']};robots={r['id'] for r in d['robots']}
    for skill in d['capabilities']:
        assert set(skill['projects'])<=projects and skill['recipe'] in recipes
    for r in d['recipes']:assert r['robot'] in robots
    assert all(len(p['commit'])==40 for p in d['projects'])
    assert next(p for p in d['projects']+d['archived_projects'] if p['id']=='mid360')['contracts'][0].startswith('FAST-LIO')


def test_registry_and_schema_route_serve_without_starting_tasks(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        d=c.get('/api/capabilities').json()
        assert {'backflip','sideflip','upright_walk','navigation'} <= {s['id'] for s in d['capabilities']}
        assert not ({s['id'] for s in d['capabilities']} & {s['id'] for s in d['archived_capabilities']})
        assert c.get('/api/schema').json()['registry']['version']==1
        assert c.get('/api/jobs').json()==[]

def test_skill_manual_links_resolve_only_indexed_reports(tmp_path):
    folder=tmp_path/'skills/backflip';folder.mkdir(parents=True)
    (folder/'TRAINING.md').write_text('# Backflip training\nVerified command record.',encoding='utf-8')
    (folder/'DEPLOYMENT.md').write_text('# SDK2 deployment plan\nNo hardware validation.',encoding='utf-8')
    with TestClient(create_app(tmp_path)) as c:
        skill=next(s for s in c.get('/api/capabilities').json()['capabilities'] if s['id']=='backflip')
        assert {m['kind'] for m in skill['manuals']}=={'training','deployment'}
        reports=c.get('/api/catalog').json()['reports']
        for manual in skill['manuals']:
            report=next(r for r in reports if r['path']==manual['path'])
            assert c.get('/api/reports/'+report['id']).json()['path']==manual['path']
        assert c.get('/api/jobs').json()==[]


def test_retained_capabilities_have_three_real_manuals_and_no_archived_recipe():
    from pathlib import Path
    from studio.registry import REGISTRY
    root=REGISTRY.parent.parent
    data=registry()
    archived={r['id'] for r in data.get('archived_recipes',[])}
    for skill in data['capabilities']:
        assert skill['recipe'] not in archived
        folder=root/'skills'/skill['id']
        for name in ('TRAINING.md','SIMULATION.md','DEPLOYMENT.md'):
            source=(folder/name).read_text(encoding='utf-8')
            assert len(source)>200, (skill['id'],name)
