import unittest
import torch
from skills.backflip.rebound_landing import LandingConfig, LandingPhases


class LandingCounterTests(unittest.TestCase):
    def setUp(self):
        self.r = LandingPhases(1,.01,LandingConfig(entry_seconds=.01))
        self.tick(True,0,1)

    def tick(self, contact, pitch, upright, count=1, body=False):
        for _ in range(count):
            self.r.step(torch.tensor([contact]),torch.tensor([[0.,float(pitch),0.]]),
                        torch.tensor([body]),torch.tensor([float(upright)]))

    def first(self):
        self.tick(True,-15,1,10)  # push 1.5 rad
        self.tick(False,-17.5,-1,20)  # airborne 3.5 rad
        self.tick(True,-10,.5,5)  # full angle sufficient, posture not yet recovered
        self.assertEqual(int(self.r.phase[0]),self.r.LANDING)
        self.assertEqual(int(self.r.completed[0]),0)
        self.tick(True,-10,.95,1)
        self.assertEqual(int(self.r.completed[0]),1)
        self.assertEqual(int(self.r.phase[0]),self.r.REBOUND)

    def test_absorption_counts_toward_rebound_deadline(self):
        self.first()
        self.assertAlmostEqual(float(self.r.rebound_time[0]),.06,places=5)
        self.tick(True,0,1,40)
        self.assertEqual(int(self.r.phase[0]),self.r.FAILED)

    def test_three_cycles_with_no_stand_hold(self):
        self.first()
        for cycle in (2,3):
            self.tick(True,-15,.95,10)
            self.tick(False,-17.5,-1,20)
            self.tick(True,-10,.5,5)
            self.tick(True,-10,.95)
            self.assertEqual(int(self.r.completed[0]),cycle)
        self.assertEqual(int(self.r.phase[0]),self.r.RECOVERY)
        self.assertAlmostEqual(float(self.r.last_rebound_seconds[0]),.18,places=5)

    def test_ground_spin_plus_hop_rejected(self):
        self.tick(True,-30,1,19)
        self.tick(False,-2,-1,10)
        self.tick(True,0,1,26)
        self.assertEqual(int(self.r.completed[0]),0)
        self.assertEqual(int(self.r.phase[0]),self.r.FAILED)

    def test_reverse_airborne_rotation_rejected(self):
        self.tick(False,20,-1,30)
        self.tick(True,0,1,26)
        self.assertEqual(int(self.r.completed[0]),0)
        self.assertEqual(int(self.r.phase[0]),self.r.FAILED)

    def test_never_inverted_rejected(self):
        self.tick(True,-15,1,10)
        self.tick(False,-17.5,.8,20)
        self.tick(True,-10,1,6)
        self.tick(True,0,1,20)
        self.assertEqual(int(self.r.completed[0]),0)

    def test_body_contact_failure_sticky(self):
        self.tick(True,0,1,body=True)
        self.tick(False,-20,-1,30)
        self.tick(True,0,1,10)
        self.assertEqual(int(self.r.phase[0]),self.r.FAILED)
        self.assertEqual(int(self.r.completed[0]),0)

    def test_short_hops_cannot_accumulate_airtime(self):
        for _ in range(10):
            self.tick(False,-20,-1)
            self.tick(True,-20,1)
        self.tick(True,0,1,30)
        self.assertEqual(int(self.r.completed[0]),0)

    def test_subset_reset(self):
        r=LandingPhases(2,.01)
        r.completed[:]=2;r.phase[:]=r.LANDING;r.rebound_time[:]=.15
        r.reset(torch.tensor([0]))
        self.assertEqual(r.completed.tolist(),[0,2])
        self.assertEqual(r.phase.tolist(),[r.ENTRY,r.LANDING])
        self.assertAlmostEqual(float(r.rebound_time[1]),.15,places=6)

    def test_nonfinite_fails(self):
        self.tick(False,float('nan'),1)
        self.assertEqual(int(self.r.phase[0]),self.r.FAILED)


if __name__ == '__main__':unittest.main()
