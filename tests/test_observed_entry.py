import unittest
import numpy as np
from skills.backflip.initial_state import validate
from skills.backflip.control import JOINT_NAMES


class ObservedEntryTests(unittest.TestCase):
    def state(self):
        return dict(schema_version=1,joint_names=list(reversed(JOINT_NAMES)),joint_position=list(range(12)),
                    joint_velocity=list(range(12)),root_position=[0,0,.33],root_quaternion=[1,0,0,0],
                    root_linear_velocity_world=[0,0,0],root_angular_velocity_body=[0,0,0])

    def test_sdk_to_actor_mapping_is_named(self):
        d=validate(self.state(),JOINT_NAMES)
        np.testing.assert_equal(d['joint_position'],np.arange(12)[::-1])

    def test_bad_mapping_or_nonphysical_numbers_rejected(self):
        for key,value in [('joint_names',['same']*12),('root_quaternion',[2,0,0,0]),
                          ('joint_velocity',[float('nan')]*12),('root_position',[0,0])]:
            d=self.state();d[key]=value
            with self.assertRaises(ValueError):validate(d,JOINT_NAMES)


if __name__=='__main__':unittest.main()
