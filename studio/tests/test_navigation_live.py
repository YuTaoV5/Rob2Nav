import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from studio.api import create_app
from skills.navigation.live import Control


def test_live_navigation_proxy_is_allowlisted_and_mutations_are_guarded(tmp_path):
    folder=tmp_path/'skills/navigation'
    folder.mkdir(parents=True)
    (folder/'index.html').write_text('<title>Live navigation</title>',encoding='utf-8')
    with TestClient(create_app(tmp_path)) as c:
        assert c.get('/navigation-live').status_code==200
        assert c.get('/navigation-live-api/not-allowlisted').status_code==404
        assert c.post('/navigation-live-api/api/control',json={'action':'start'}).status_code==403
        assert c.post('/navigation-live-api/api/control',json={'action':'start'},
            headers={'X-Studio-Request':'1','Origin':'https://untrusted.example'}).status_code==403
        assert c.post('/navigation-live-api/api/state',json={},headers={'X-Studio-Request':'1'}).status_code==404


@pytest.mark.parametrize('payload',[
    {'action':'goal','goal':[1]}, {'action':'goal','goal':[1,2,3]},
    {'action':'goal','goal':['NaN',0]}, {'action':'goal','goal':[0,'Infinity']},
    {'action':'teleport'}, {'action':'start','robot_qpos':[0]*19},
])
def test_navigation_controls_reject_nonfinite_and_pose_mutations(payload):
    with pytest.raises(ValidationError):
        Control.model_validate(payload)
