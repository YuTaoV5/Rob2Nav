from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.checkpoint_live import CheckpointLaunch,resolve_launch
from studio.config import RunConfig
from studio.jobs import Jobs

def training(root):
    jobs=Jobs(root);cfg=RunConfig(name='training',navigation=False,mode='student',arrival_aware=False,kind='flat').model_dump()
    job=jobs.create(cfg);folder=root/'outputs/studio/runs'/job['id'];folder.mkdir(parents=True)
    (folder/'model_10.pt').write_bytes(b'checkpoint10');(folder/'model_20.pt').write_bytes(b'checkpoint20')
    return jobs,job

def test_only_source_job_saved_candidates_and_contract(tmp_path):
    jobs,j=training(tmp_path)
    source,cfg,w=resolve_launch(tmp_path,jobs,CheckpointLaunch(job_id=j['id']))
    assert cfg.checkpoint.endswith('model_20.pt') and cfg.task=='simulation'
    assert cfg.navigation is False and cfg.mode=='student' and cfg.sensor_rig==j['config']['sensor_rig']
    assert w['validated_best'] is False
    for path in ['/tmp/untrusted.pt','../../shell','artifacts/other.pt']:
        with pytest.raises(ValueError):resolve_launch(tmp_path,jobs,CheckpointLaunch(job_id=j['id'],checkpoint=path))
    bad=dict(j['config'],algorithm_id='unknown');other=jobs.create(bad)
    with pytest.raises(ValueError):resolve_launch(tmp_path,jobs,CheckpointLaunch(job_id=other['id']))

def test_launch_header_gate_unknown_job_and_unavailable_runtime(tmp_path,monkeypatch):
    jobs,j=training(tmp_path);monkeypatch.setenv('ROB2NAV_ISAAC_PYTHON',str(tmp_path/'missing'))
    with TestClient(create_app(tmp_path,allow_training=False)) as c:
        assert c.post('/api/checkpoint-simulation',json={'job_id':j['id']}).status_code==403
        headers={'X-Studio-Request':'1'}
        assert c.post('/api/checkpoint-simulation',json={'job_id':'missing'},headers=headers).status_code==422
        assert c.post('/api/checkpoint-simulation',json={'job_id':j['id']},headers=headers).status_code==409
        assert c.post('/api/checkpoint-simulation',json={'job_id':j['id'],'module':'os'},headers=headers).status_code==422
        assert c.get('/api/resources').json()['training_enabled'] is False
    assert len(jobs.all())==1

def test_inventory_covers_export_formats_and_scopes_downloads(tmp_path):
    for rel in ['artifacts/a/model.pt','artifacts/a/policy.onnx','skills/navigation/policy/navigation.zip','vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt']:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(rel.encode())
    with TestClient(create_app(tmp_path)) as c:
        models=c.get('/api/catalog').json()['models']
        assert {m['format'] for m in models}=={'PT','ONNX','ZIP'}
        assert any(m['role']=='dependency' for m in models)
        for m in models:
            response=c.get(m['download_url'])
            assert response.status_code==200 and response.content==(tmp_path/m['path']).read_bytes()
        assert c.get('/api/model-assets/unregistered/download').status_code==404
