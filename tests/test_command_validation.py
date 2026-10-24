import unittest
import numpy as np
from rob2nav.deployment.command_validation import CommandRateValidation


class CommandValidationTests(unittest.TestCase):
    def test_detects_jump_despite_no_measured_velocity_input(self):
        v=CommandRateValidation(['hip','calf'],[30.1,15.7])
        v.sample(0,[0,0]);v.sample(.002,[0,.04])
        r=v.report(2)
        self.assertFalse(r['passed']);self.assertEqual(r['peak']['joint'],'calf')
        self.assertGreater(r['max_command_rate_ratio'],1)

    def test_uses_joint_specific_limits_and_real_interval(self):
        v=CommandRateValidation(['hip','calf'],[30.1,15.7])
        v.sample(0,[0,0]);v.sample(.004,[.08,.04])
        self.assertTrue(v.report(2)['passed'])
        self.assertAlmostEqual(v.report(2)['max_command_rate_ratio'],20/30.1)

    def test_rejects_missing_coverage_and_single_sample(self):
        v=CommandRateValidation(['a'],[1])
        v.sample(0,[0]);self.assertFalse(v.report(1)['passed'])
        v.sample(1,[.5]);self.assertFalse(v.report(3)['passed'])

    def test_rejects_nonfinite_targets_and_reversed_time(self):
        for t,q in [(1,[float('nan')]),(0,[0]),(-1,[0]),(1,[0,0])]:
            v=CommandRateValidation(['a'],[1]);v.sample(0,[0]);v.sample(t,q)
            self.assertFalse(v.report(2)['passed'])

    def test_copies_reused_output_buffer(self):
        v=CommandRateValidation(['a'],[1]);q=np.array([0.])
        v.sample(0,q);q[0]=2;v.sample(1,q)
        self.assertFalse(v.report(2)['passed'])

    def test_rejects_invalid_limits(self):
        for limits in [[0],[-1],[float('nan')],[1,2]]:
            with self.assertRaises(ValueError):CommandRateValidation(['a'],limits)

if __name__=='__main__':unittest.main()
