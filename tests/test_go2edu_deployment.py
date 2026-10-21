"""Offline math and actuation-gate tests. No ROS, DDS or robot is opened."""
import importlib.util,json,struct,sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'deployment/go2edu'
def load(name):
    spec=importlib.util.spec_from_file_location('go2edu_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prepare=load('prepare');Gate=load('velocity_gate').Gate

def healthy(g,t):
    for k in ('odom','cloud','deadman','robot'):g.feed(k,t)
    g.set_command((.2,.1,.3),t);g.publishers=1;g.pose=(0.,0.,.35)

def test_uncalibrated_and_left_handed_transforms_rejected(tmp_path):
    p=tmp_path/'c.json';p.write_text((ROOT/'calibration.template.json').read_text())
    with pytest.raises(ValueError):prepare.load_calibration(p)
    bad=np.eye(4);bad[0,0]=-1
    with pytest.raises(ValueError):prepare.transform(bad)

def test_measured_inverse_for_lio_body_offset():
    t=np.eye(4);t[:3,3]=[.2,0,.15];angle=.6
    t[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    assert np.allclose(prepare.transform(t)@np.linalg.inv(t),np.eye(4))
    # Translation of the inverse is -R.T*t, not just negating xyz.
    assert not np.allclose(np.linalg.inv(t)[:3,3],-t[:3,3])

@pytest.mark.parametrize('big',[False,True])
def test_cloud_z_shift_preserves_xy_fields_and_row_padding(big):
    order='>' if big else '<';point=struct.pack(order+'fffI',1.,2.,3.,123)
    raw=point+b'PAD!'+point+b'PAD!'
    result=prepare.shift_cloud_z(raw,1,2,16,20,8,.4,big)
    assert struct.unpack_from(order+'fffI',result)==pytest.approx((1.,2.,3.4,123))
    assert result[16:20]==b'PAD!' and result[36:40]==b'PAD!'

def test_gate_requires_new_goal_after_arm_and_never_auto_resumes():
    g=Gate(True);healthy(g,1.)
    assert g.step(1.)==(0,0,0) and g.arm(1.)[0]
    assert g.step(1.05)==(0,0,0)
    g.set_goal((2,0,.35),1.05)
    assert g.step(1.06)==(0,0,0)
    g.set_plan(1.07);healthy(g,1.10)
    assert g.step(1.10)[0]>0
    assert g.step(1.7)==(0,0,0) and not g.armed
    healthy(g,1.71)
    assert g.step(1.71)==(0,0,0)
    assert g.arm(1.71)[0] and g.goal is None

def test_gate_stops_on_bad_stamp_nan_duplicate_publishers_or_pose_jump():
    for fault in ('stamp','nan','publishers','jump','deadman','robot'):
        g=Gate(True);healthy(g,1.);assert g.arm(1.)[0];g.set_goal((2,0,.35),1.);g.set_plan(1.)
        if fault=='stamp':g.feed('cloud',1.1,stamp_age=10.)
        if fault=='nan':g.set_command((float('nan'),0,0),1.1)
        if fault=='publishers':g.publishers=2
        if fault=='jump':g.set_pose((1,0,.35),1.1,0)
        if fault=='deadman':g.received['deadman']=0
        if fault=='robot':g.received['robot']=0
        assert g.step(1.1)==(0,0,0) and not g.armed,fault

def test_gate_limits_acceleration_norm_and_stops_at_arrival():
    g=Gate();healthy(g,1.);g.arm(1.);g.set_goal((2,0,.35),1.);g.set_plan(1.);g.step(1.)
    for t in np.arange(1.05,3.,.05):
        healthy(g,t);v=g.step(t)
        assert np.linalg.norm(v[:2])<=.2+1e-9 and abs(v[2])<=.3
    # Exercise arrival independently of the jump detector tested above.
    for k in ('cmd','odom','cloud','deadman'):g.feed(k,3.)
    g.pose=(1.8,0,.35)
    assert g.step(3.)==(0,0,0) and g.reason=='arrived'

def test_retarget_holds_old_command_until_new_plan_and_command():
    g=Gate();healthy(g,1.);g.arm(1.);g.set_goal((2,0,.35),1.);g.set_plan(1.)
    healthy(g,1.1);assert g.step(1.1)==(0,0,0)
    healthy(g,1.2);assert g.step(1.2)[0]>0
    g.set_goal((0,2,.35),1.21)
    assert g.step(1.21)==(0,0,0)
    g.set_plan(1.22)
    assert g.step(1.23)==(0,0,0)
    healthy(g,1.24);assert g.step(1.24)[0]>0

def test_default_main_does_not_import_sdk_or_open_dds(monkeypatch):
    import builtins,types
    module=load('velocity_gate');published=[];done=[False]
    class Message:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    for package,names in {'geometry_msgs':('Twist','PoseStamped'),'nav_msgs':('Odometry',),'sensor_msgs':('PointCloud2',),'std_msgs':('Bool','String'),'scan_planner':('Bspline',),'std_srvs':('SetBool','SetBoolResponse')}.items():
        parent=types.ModuleType(package);suffix='srv' if package=='std_srvs' else 'msg';child=types.ModuleType(package+'.'+suffix)
        for name in names:setattr(child,name,Message)
        monkeypatch.setitem(sys.modules,package,parent);monkeypatch.setitem(sys.modules,package+'.'+suffix,child)
    class Publisher:
        def __init__(self,*a,**k):pass
        def publish(self,m):published.append(json.loads(m.data));done[0]=True
    class Subscriber:
        def __init__(self,*a,**k):pass
        def get_num_connections(self):return 1
    ros=types.ModuleType('rospy');ros.init_node=lambda *a,**k:None;ros.get_param=lambda *a:False
    ros.Publisher=Publisher;ros.Subscriber=Subscriber;ros.Service=lambda *a:None;ros.is_shutdown=lambda:done[0]
    monkeypatch.setitem(sys.modules,'rospy',ros);monkeypatch.setattr(sys,'argv',['velocity_gate.py'])
    original_import=builtins.__import__
    def guarded(name,*a,**k):
        if name.startswith('unitree_sdk2py'):raise AssertionError('Dry-run attempted SDK import')
        return original_import(name,*a,**k)
    monkeypatch.setattr(builtins,'__import__',guarded);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    module.main()
    assert published and published[0]['execute'] is False and published[0]['armed'] is False

def test_execute_and_replay_rejected_before_ros_import(monkeypatch):
    module=load('velocity_gate')
    monkeypatch.setattr(sys,'argv',['velocity_gate.py','--execute','--replay'])
    with pytest.raises(SystemExit) as exc:module.main()
    assert exc.value.code==2
