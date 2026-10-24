import unittest
import torch
from skills.backflip.rebound import ReboundPhases, ReboundConfig


class ReboundTests(unittest.TestCase):
    def setUp(self):
        self.s = ReboundPhases(2, .01, ReboundConfig(entry_seconds=.01))

    def sample(self, contact, pitch=0., body=False, n=1):
        for _ in range(n):
            self.s.step(torch.tensor([contact, True]),
                        torch.tensor([[0., pitch, 0.], [0., 0., 0.]]),
                        torch.tensor([body, False]))

    def flip(self):
        self.sample(False, -2 * 3.14159265359, n=100)
        self.sample(True)

    def test_three_distinct_cycles_without_entry_hold_between(self):
        self.sample(True)
        for i in range(3):
            self.flip()
            self.assertEqual(self.s.completed[0].item(), i + 1)
            self.assertEqual(self.s.phase[0].item(), self.s.RECOVERY if i == 2 else self.s.REBOUND)
            if i < 2:
                self.sample(True, n=10)
        self.assertEqual(self.s.completed[1].item(), 0)
        self.assertLess(self.s.last_rebound_seconds[0].item(), .45)

    def test_waiting_between_flips_rejected(self):
        self.sample(True)
        self.flip()
        self.sample(True, n=50)
        self.assertEqual(self.s.phase[0].item(), self.s.FAILED)

    def test_small_hop_and_wrong_direction_rejected(self):
        self.sample(True)
        self.sample(False, -1., n=10)
        self.sample(True)
        self.assertEqual(self.s.completed[0].item(), 0)
        self.assertEqual(self.s.phase[0].item(), self.s.FAILED)
        self.s.reset([0])
        self.sample(True)
        self.sample(False, 6.28, n=100)
        self.sample(True)
        self.assertEqual(self.s.phase[0].item(), self.s.FAILED)

    def test_body_contact_sticky_and_partial_reset(self):
        self.sample(True, body=True)
        self.flip()
        self.assertEqual(self.s.completed[0].item(), 0)
        self.assertEqual(self.s.phase[0].item(), self.s.FAILED)
        self.s.reset([0])
        self.assertEqual(self.s.phase[0].item(), self.s.ENTRY)
        self.assertEqual(self.s.phase[1].item(), self.s.PUSH)

    def test_missing_landing_does_not_count_three_rotations(self):
        self.sample(True)
        self.sample(False, -6.28, n=300)
        self.assertEqual(self.s.completed[0].item(), 0)
        self.sample(True)
        self.assertEqual(self.s.phase[0].item(), self.s.FAILED)


if __name__ == '__main__':
    unittest.main()
