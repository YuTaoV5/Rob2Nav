import unittest
import torch
from skills.gait_objectives import alternating_rear_targets


class RearGaitTests(unittest.TestCase):
    def test_schedule_alternates_loaded_support_with_clearance(self):
        stance,height=alternating_rear_targets(torch.tensor([2.,2.125,2.25,2.375,2.5]))
        self.assertTrue(bool(stance.any(dim=-1).all()))
        self.assertTrue(bool(stance[1,0]) and not bool(stance[1,1]))
        self.assertFalse(bool(stance[3,0]));self.assertTrue(bool(stance[3,1]))
        self.assertGreater(float(height[1,1]),.05)
        self.assertGreater(float(height[3,0]),.05)
        torch.testing.assert_close(stance[0],stance[4])


if __name__=='__main__':unittest.main()
