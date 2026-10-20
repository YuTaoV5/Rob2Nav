"""A gyro observation must not rotate with the URDF inertia principal axes."""
import unittest
import numpy as np
import mujoco
from rob2nav.mujoco_skill import base_link_velocity,motor_torque

class StuntFrameTests(unittest.TestCase):
    def test_gyro_uses_link_axes_for_rotated_inertias_and_base_poses(self):
        for inertia in ['1 0 0 0','0.7071067812 0 0.7071067812 0']:
            for pose in [[1,0,0,0],[.7071067812,0,0,.7071067812]]:
                with self.subTest(inertia=inertia,pose=pose):
                    xml=f'<mujoco><worldbody><body name="base"><freejoint/><inertial mass="1" pos="0 0 0" quat="{inertia}" diaginertia=".1 .2 .3"/></body></worldbody></mujoco>'
                    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
                    data.qpos[3:7]=pose;data.qvel[3:6]=[.3,-.7,1.2]
                    mujoco.mj_forward(model,data)
                    result=base_link_velocity(model,data,model.body('base').id,np.zeros(6))
                    np.testing.assert_allclose(result[:3],[.3,-.7,1.2],atol=1e-9)

class MotorEnvelopeTests(unittest.TestCase):
    def test_no_accelerating_torque_beyond_free_speed_and_braking_is_bounded(self):
        effort=np.array([23.7,45.43]);speed=np.array([30.1,15.7]);q=np.zeros(2)
        for sign in [-1.,1.]:
            dq=sign*1.2*speed
            output=motor_torque(sign*np.ones(2)*100,q,dq,effort,speed,'dc_linear_v1')
            self.assertTrue(np.all(output*dq<=0))
            self.assertTrue(np.all(abs(output)<=effort))
            brake=motor_torque(-sign*np.ones(2)*100,q,dq,effort,speed,'dc_linear_v1')
            np.testing.assert_allclose(brake,-sign*effort)

if __name__=='__main__':unittest.main()
