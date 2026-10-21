import json
from fastapi.testclient import TestClient
from studio.api import create_app


def test_switch_planner_is_readonly_allowlisted_and_guarded(tmp_path):
    folder=tmp_path/'deployment/skills';folder.mkdir(parents=True)
    for name,atomic,kp in [('walk',False,20),('flip',True,40)]:
        (folder/(name+'.json')).write_text(json.dumps(dict(id=name,joints=['j'+str(i) for i in range(12)],default=[0]*12,kp=[kp]*12,kd=[1]*12,atomic=atomic,hardware_qualified=False)))
    with TestClient(create_app(tmp_path)) as c:
        assert c.get('/api/deployment/unified').json()['motor_control_from_web'] is False
        assert c.post('/api/deployment/unified/plan',json={'source':'walk','target':'flip'}).status_code==403
        headers={'X-Studio-Request':'1'}
        result=c.post('/api/deployment/unified/plan',json={'source':'flip','target':'walk'},headers=headers)
        assert result.status_code==200 and result.json()['executed'] is False
        assert '原子动作' in result.json()['route'][1]
        assert c.post('/api/deployment/unified/plan',json={'source':'../../shell','target':'flip'},headers=headers).status_code==404
        assert c.post('/api/deployment/unified/plan',json={'source':'walk','target':'flip','enable_motors':True},headers=headers).status_code==422
