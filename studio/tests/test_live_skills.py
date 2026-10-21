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
