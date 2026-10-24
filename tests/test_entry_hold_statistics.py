import unittest
import numpy as np
from rob2nav.deployment.atomic_validation import EntryHoldStatistics

class EntryHoldTests(unittest.TestCase):
    def test_between_control_samples_peak_is_retained(self):
        stats=EntryHoldStatistics(.001)
        stats.sample(1e-11,np.zeros(12),np.full(12,99.),[0,99.,0],np.zeros(12))
        for i in range(1,1001):
            stats.sample(i*.001,np.zeros(12),np.full(12,7. if i==11 else .2),
                         [0,2. if i==11 else .1,0],np.zeros(12))
        r=stats.report()
        self.assertTrue(r['coverage_complete'])
        self.assertEqual(r['max_joint_speed_rad_s'],7.)
        self.assertEqual(r['max_body_angular_speed_rad_s'],2.)
        self.assertFalse(r['visual_twitch_free_verified'])

    def test_missing_substep_rejects_coverage_and_flip_is_excluded(self):
        stats=EntryHoldStatistics(.001)
        for i in range(2,1001):stats.sample(i*.001,np.zeros(12),np.ones(12),[0,0,0],np.zeros(12))
        stats.sample(1.1,np.ones(12)*99,np.ones(12)*99,[99,0,0],np.zeros(12))
        r=stats.report()
        self.assertFalse(r['coverage_complete'])
        self.assertEqual(r['samples'],999)
        self.assertEqual(r['max_joint_speed_rad_s'],1.)
