import unittest
from types import SimpleNamespace
import numpy as np
from skills.frontflip.task import roll_rate_penalty
from skills.backflip.reward_options import parse_scales,apply_scales

class FrontRollRewardTests(unittest.TestCase):
    def test_only_roll_rate_is_penalized(self):
        env=SimpleNamespace(_robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=np.array([[0.,20.,5.],[-2.,0.,0.],[3.,-9.,0.]]))))
        np.testing.assert_array_equal(roll_rate_penalty(env),[0,2,3])
    def test_front_scale_is_explicit_and_back_schema_rejects_it(self):
        values=parse_scales('{"ang_vel_x":-1}')
        front=SimpleNamespace(rewards=SimpleNamespace(scales={'ang_vel_x':0.,'ang_vel_y':5.}))
        apply_scales(front,values)
        self.assertEqual(front.rewards.scales,{'ang_vel_x':-1,'ang_vel_y':5.})
        back=SimpleNamespace(rewards=SimpleNamespace(scales={'ang_vel_y':5.}))
        with self.assertRaises(ValueError):apply_scales(back,values)
if __name__=='__main__':unittest.main()
