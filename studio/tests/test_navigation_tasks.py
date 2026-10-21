import numpy as np
from skills.navigation.tasks import StaticRoute,follow_goal,pedestrian_guard

def test_static_route_uses_door_without_corner_cutting():
    boxes=[dict(x=3,y=-1.9,sx=.1,sy=1.1),dict(x=3,y=2.5,sx=.1,sy=.5)]
    r=StaticRoute(boxes);path=r.path(np.array([0.,-1.4]),np.array([6.,-1.4]))
    assert len(path)>2 and all(r.line(a,b) for a,b in zip(path,path[1:]))
    assert r.path(np.array([0.,0.]),np.array([3.,-1.4]))==[]

def test_follow_standoff_and_predictive_person_stop():
    goal,d=follow_goal([0,0],[3,0],[.3,0],1.2)
    assert np.allclose(goal,[1.8,0]) and d==3
    track=dict(position=[1.,.8],velocity=[0,-.6],visible=True)
    command,why=pedestrian_guard([.4,0,0],np.eye(3),[0,0],[track])
    assert why and 0<=command[0]<.4
    track['visible']=False
    command,why=pedestrian_guard([.4,0,0],np.eye(3),[0,0],[track])
    assert not why and command[0]==.4


def test_carrot_keeps_static_map_clearance():
    r=StaticRoute([dict(x=1.5,y=0.,sx=.1,sy=.8)])
    path=r.path(np.array([0.,0.]),np.array([3.,0.]))
    carrot=r.carrot(np.array([0.,0.]),path)
    assert r.line(np.array([0.,0.]),carrot)
