import unittest
from deployment.patch_scan_goal_height import patched


class GoalHeightPatchTests(unittest.TestCase):
    def test_patch_is_opt_in_bounded_and_idempotent(self):
        source='navi_mode_ = load_parameter<int>(node_, "fsm.navi_mode", -1);\nend_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, rviz_goal_height_;'
        result=patched(source)
        self.assertIn('"fsm.rob2nav_use_goal_height", false',result)
        self.assertIn('std::isfinite(requested_height)',result)
        self.assertIn('std::max(0.20, std::min(1.00, requested_height)) : rviz_goal_height_',result)
        self.assertEqual(result,patched(result))

    def test_unknown_source_is_never_silently_patched(self):
        with self.assertRaises(ValueError):patched('unexpected upstream')


if __name__=='__main__':unittest.main()
