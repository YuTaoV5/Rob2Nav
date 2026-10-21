from types import SimpleNamespace
import unittest
import numpy as np
from skills.backflip.control import JOINT_NAMES,control_contract,configure_playback


class BackflipControlTests(unittest.TestCase):
    def config(self):
        return dict(action_scale=.5,action_clip=8.,stiffness=40.,damping=1.,motor_velocity_x1=13.5,
            motor_torque_y1=20.2,motor_torque_y2=23.4,
            joint_velocity_limits=[30.1,30.1,15.7]*4,target_velocity_limits=[13.5]*12)

    def test_urdf_calf_limits_reach_native_motor_curve(self):
        module=SimpleNamespace(JOINT_NAMES=JOINT_NAMES)
        contract=configure_playback(module,self.config())
        np.testing.assert_equal(module.MOTOR_VELOCITY_X2[2::3],[15.7]*4)
        np.testing.assert_equal(module.TARGET_VELOCITY_LIMIT,[13.5]*12)
        self.assertEqual(module.ACTION_CLIP,8.)
        self.assertEqual(contract,control_contract(contract))

    def test_invalid_parameters_and_joint_order_fail(self):
        for update in ({'joint_names':list(reversed(JOINT_NAMES))},{'joint_velocity_limits':[30.]*11},
                       {'joint_velocity_limits':[13.5]*12},{'stiffness':float('nan')}):
            with self.assertRaises(ValueError):control_contract(dict(self.config(),**update))


if __name__=='__main__':unittest.main()
