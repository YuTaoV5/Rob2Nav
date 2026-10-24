import unittest
from rob2nav.rebound_validation import ContinuousFlipAudit


class IndependentAuditTests(unittest.TestCase):
    def run_trace(self, gap=.1, rotations=(-6.28,-6.28,-6.28), body=False, skip=None):
        dt=.005;a=ContinuousFlipAudit(dt);tick=0
        def segment(seconds,contact,rate=0.):
            nonlocal tick
            for _ in range(round(seconds/dt)):
                tick+=1
                if tick!=skip:a.sample(tick*dt,contact,[0.,rate,0.],body)
        segment(1.,True)
        for i,rotation in enumerate(rotations):
            segment(1.,False,rotation)
            segment(gap if i<len(rotations)-1 else 1.,True)
        return a.report(tick)

    def test_complete_three_rebounds(self):
        r=self.run_trace()
        self.assertTrue(r['motion_pass'])
        self.assertEqual(r['observed_flights'],3)
        self.assertAlmostEqual(r['connections'][0]['confirmed_rebound_s'],.12)
        self.assertFalse(r['hardware_qualified'])

    def test_old_waiting_sequence_rejected(self):
        r=self.run_trace(gap=8.)
        self.assertTrue(all(f['motion_pass'] for f in r['flights']))
        self.assertFalse(r['motion_pass'])
        self.assertFalse(r['connections'][0]['passed'])

    def test_wrong_direction_extra_jump_and_partial_rotation_rejected(self):
        for rotations in [(-6.28,6.28,-6.28),(-6.28,-3.,-6.28),(-6.28,)*4,(-18.84,)]:
            with self.subTest(rotations=rotations):self.assertFalse(self.run_trace(rotations=rotations)['motion_pass'])

    def test_missing_physics_sample_or_body_contact_rejected(self):
        r=self.run_trace(skip=211)
        self.assertFalse(r['coverage_complete']);self.assertFalse(r['motion_pass'])
        self.assertFalse(self.run_trace(body=True)['motion_pass'])

    def test_unlanded_rotation_not_a_completed_flip(self):
        a=ContinuousFlipAudit(.005,cycles=2)
        for i in range(200):a.sample((i+1)*.005,False,[0.,-6.28,0.])
        r=a.report(200)
        self.assertEqual(r['observed_flights'],0)
        self.assertTrue(r['unfinished_flight']);self.assertFalse(r['motion_pass'])


if __name__=='__main__':unittest.main()
