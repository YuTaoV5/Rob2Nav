import pytest
from fastapi.testclient import TestClient
from studio.api import create_app
from studio.live_skills import Camera
from pydantic import ValidationError


def test_live_skill_mutation_gate_and_no_arbitrary_execution(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get('/api/live-skills/state').json()['phase']=='IDLE'
        assert c.post('/api/live-skills/start',json={'skill':'sideflip'}).status_code==403
        headers={'X-Studio-Request':'1'}
        for payload in [{'skill':'../../shell'},{'skill':'sideflip','module':'os'},{'skill':'sideflip','policy':'/tmp/custom.pt'}]:
            assert c.post('/api/live-skills/start',json=payload,headers=headers).status_code==422
        assert c.post('/api/live-skills/camera',json={},headers=headers).status_code==409
        assert c.post('/api/live-skills/stop',json={},headers=headers).json()=={'stopped':True}


@pytest.mark.parametrize('payload',[{'distance':0},{'elevation':0},{'pan_x':11},{'azimuth':float('nan')},{'distance':float('inf')},{'qpos':[0]*19}])
def test_camera_bounds_and_pose_write_rejection(payload):
    with pytest.raises(ValidationError):Camera(**payload)


def test_composition_rejects_unapproved_or_unbounded_steps(tmp_path):
    folder=tmp_path/'deployment/skills';folder.mkdir(parents=True)
    (folder/'stand.json').write_text('{"id":"stand","atomic":false}')
    (folder/'walk_moe.json').write_text('{"id":"walk_moe","atomic":false}')
    (folder/'sideflip69.json').write_text('{"id":"sideflip69","atomic":true}')
    headers={'X-Studio-Request':'1'}
    valid=[{'skill':'stand'},{'skill':'walk_moe','velocity':[.18,0,0]},{'skill':'stand'}]
    with TestClient(create_app(tmp_path)) as c:
        route='/api/live-skills/composition/start'
        assert c.post(route,json={'steps':valid},headers={}).status_code==403
        for steps,extra in [
            ([{'skill':'walk_moe'},valid[-1]],{}),
            ([valid[0],{'skill':'sideflip69','trigger':True},valid[-1]],{}),
            ([valid[0],{'skill':'walk_moe','trigger':True},valid[-1]],{}),
            ([valid[0],{'skill':'walk_moe','velocity':[2,0,0]},valid[-1]],{}),
            (valid,{'cycles':5}),
            (valid,{'module':'os'}),
        ]:
            assert c.post(route,json={'steps':steps,**extra},headers=headers).status_code==422


@pytest.mark.parametrize('compact',[False,True])
def test_composition_triple_backflip_preserves_three_triggers(tmp_path,monkeypatch,compact):
    import json
    from types import SimpleNamespace
    folder=tmp_path/'deployment/skills';folder.mkdir(parents=True)
    (folder/'stand.json').write_text(json.dumps({'id':'stand','atomic':False}))
    flip=folder/'backflip60.json'
    flip.write_text(json.dumps({'id':'backflip60','atomic':True,'simulation_enabled':True}))
    process=SimpleNamespace(pid=999999,poll=lambda:0,wait=lambda **kw:0,terminate=lambda:None)
    commands=[]
    def launch(command,**kw):
        commands.append(command)
        return process
    monkeypatch.setattr('studio.live_skills.subprocess.Popen',launch)
    steps=[{'skill':'stand'}]+[{'skill':'backflip60','trigger':True} for _ in range(3)]+[{'skill':'stand'}]
    with TestClient(create_app(tmp_path)) as c:
        response=c.post('/api/live-skills/composition/start',json={'steps':steps,'compact_backflip_repeat':compact},headers={'X-Studio-Request':'1'})
        assert response.status_code==200,response.text
        assert ('--experimental-compact-backflip-repeat' in commands[-1]) is compact
        assert response.json()['compact_backflip_repeat'] is compact
        if compact:
            invalid=c.post('/api/live-skills/composition/start',json={'steps':[steps[0],steps[1],steps[-1]],'compact_backflip_repeat':True},headers={'X-Studio-Request':'1'})
            assert invalid.status_code==422
        saved=json.loads(next((tmp_path/'outputs/live_skills').glob('*/sequence.json')).read_text())
        assert sum(s['trigger'] for s in saved['steps'])==3
        assert [s['skill'] for s in saved['steps']]==[s['skill'] for s in steps]
        flip.write_text(json.dumps({'id':'backflip60','atomic':True,'simulation_enabled':False}))
        assert c.post('/api/live-skills/composition/start',json={'steps':steps},headers={'X-Studio-Request':'1'}).status_code==422
