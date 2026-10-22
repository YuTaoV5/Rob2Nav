from types import SimpleNamespace
import numpy as np
import pytest
from skills.sideflip.gains import checked, configure, verify_export


def test_actual_training_and_export():
    a=SimpleNamespace(stiffness=40.,damping=1.)
    cfg=SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={'calf':a})))
    configure(cfg,40.,2.)
    env={'scene':{'robot':{'actuators':{'calf':vars(a)}}}}
    assert verify_export(env,{'kp':40.,'kd':2.})==(40.,2.)
    with pytest.raises(ValueError):verify_export(env,{'kp':40.,'kd':1.})


def test_unsaturated_motor_response():
    pytest.importorskip('mujoco')
    from rob2nav.mujoco_skill import motor_torque
    q=np.zeros(2); target=np.array([.1,-.1]);dq=np.array([1.,-1.])
    np.testing.assert_allclose(motor_torque(target,q,dq,np.ones(2)*50,np.ones(2)*30,'dc_linear_v1',40.,2.),[2.,-2.])
    np.testing.assert_array_equal(motor_torque(target,q,dq,np.ones(2)*50,np.ones(2)*30),40*(target-q)-dq)


@pytest.mark.parametrize('kp,kd',[(float('nan'),1),(40,float('inf')),(0,1),(40,-1),(81,1),(40,9),(True,1)])
def test_invalid_gains_rejected(kp,kd):
    with pytest.raises(ValueError):checked(kp,kd)
