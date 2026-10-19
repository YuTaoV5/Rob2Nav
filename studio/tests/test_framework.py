import math
import pytest
from fastapi.testclient import TestClient
from studio.api import create_app
from rob2nav.runtime import DeploymentRuntime,VelocityCommand,Mode
from rob2nav.contracts import profiles,PolicyContract
from rob2nav.navigation import plan,collides,shield


def test_command_gate_fault_stops_transport_and_cannot_rearm():
    class Backend:
        def __init__(self):self.commands=[];self.stops=0
        def write(self,c):self.commands.append(c)
        def stop(self):self.stops+=1
    b=Backend();r=DeploymentRuntime(b,lambda:10.)
    with pytest.raises(ValueError):r.arm()
    r.arm(True);r.activate('walk')
    assert r.command(VelocityCommand(2,0,0,10)).vx==1
    with pytest.raises(ValueError,match='Stale'):r.command(VelocityCommand(1,0,0,9))
    assert r.mode==Mode.FAULT and b.stops==1 and len(b.commands)==1
    with pytest.raises(ValueError):r.activate('jump')


def test_profile_order_and_decimation_are_explicit():
    p=profiles()[0]
    assert p['joints'][0]=='FL_hip_joint' and p['physics_hz']==200 and p['policy_hz']==50
    with pytest.raises(ValueError):PolicyContract('bad','go2',('x','x'),'obs','act',50,200,(1,1),(1,1),'test','test').validate()


def test_geometry_path_segments_and_height_filter():
    r=plan('clutter');assert r['success'] and r['path_length_m']>10
    for a,b in zip(r['path'],r['path'][1:]):
        delta=(b[3]-a[3]+math.pi)%(2*math.pi)-math.pi
        for i in range(11):
            t=i/10
            assert not collides(a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[3]+delta*t,r['boxes'],r['length'],r['width'],r['margin'])
    assert not collides(4,4,0,[(3,5,2,6,.85,1.5)],.6,.3,.05)
    assert collides(4,4,0,[(3,5,2,6,0,1.5)],.6,.3,.05)


def test_cbf_projection_satisfies_constraint_and_overlap_rejected():
    p=(2.7,3.3);o=(3.4,3.3,.25);r=shield([1,.2],p,[o]);h=(p[0]-o[0])**2-(o[2]+.3)**2
    assert r['feasible'] and 2*(p[0]-o[0])*r['velocity'][0]>=-2*h-1e-7
    assert not shield([1,0],[3.4,3.3],[o])['feasible']


def test_framework_api_is_bounded_and_not_native_claim(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert not c.get('/api/deployment').json()['hardware_enabled']
        assert not c.get('/api/navigation').json()['native_connected']
        assert c.post('/api/navigation/probe',json={}).status_code==403
        headers={'X-Studio-Request':'1'}
        assert c.post('/api/navigation/probe',headers=headers,json={'length':50}).status_code==422
        r=c.post('/api/navigation/probe',headers=headers,json={'scene':'narrow'})
        assert r.status_code==200 and r.json()['success']
