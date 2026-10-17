"""Sensor-memory contracts and physical terrain invariants."""
import unittest
import torch
from perceptive_rl.elevation import ElevationMemory
from perceptive_rl.geometry import lane_boxes
from perceptive_rl.io import actuator_gains
from types import SimpleNamespace


class MeasuredElevationTests(unittest.TestCase):
    def setUp(self):
        self.memory = ElevationMemory(1, "cpu")
        self.position = torch.tensor([[0., 0., .4]])
        self.yaw = torch.zeros(1)
        self.reset = torch.ones(1, dtype=torch.bool)
        self.fresh = torch.ones(1, dtype=torch.bool)

    def observe(self, points, now=0., reset=None, fresh=None):
        return self.memory.update(torch.tensor([points], dtype=torch.float32), self.position, self.yaw, now,
            self.reset if reset is None else reset, self.fresh if fresh is None else fresh)

    def test_unknown_stays_unknown_and_finite(self):
        out = self.observe([[float("inf")]*3])
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(out.abs().sum()), 0.)

    def test_height_sign_top_surface_and_validity(self):
        out = self.observe([[.5, 0., 0.], [.5, 0., .2]])
        seen = out[0, 187:].bool()
        self.assertEqual(int(seen.sum()), 1)
        self.assertAlmostEqual(float(out[0, :187][seen][0]), -.3, places=5)

    def test_history_stays_in_world_instead_of_following_robot(self):
        first = self.observe([[.5, 0., .2]])
        self.position[:, 0] = .2
        after = self.observe([[float("inf")]*3], now=.2, reset=~self.reset, fresh=~self.fresh)
        before_bin = int(first[0, 187:].argmax())
        after_bin = int(after[0, 187:].argmax())
        self.assertEqual(before_bin-after_bin, 2)
        self.assertAlmostEqual(float(self.memory.world[0, after_bin, 0]), .5)

    def test_memory_expires_and_episode_reset_clears(self):
        self.observe([[.5, 0., .2]])
        after = self.observe([[float("inf")]*3], now=1.01, reset=~self.reset, fresh=~self.fresh)
        self.assertEqual(float(after[:, 187:].sum()), 0.)
        self.observe([[.5, 0., .2]], now=2.)
        after = self.observe([[float("inf")]*3], now=2.02)
        self.assertEqual(float(after[:, 187:].sum()), 0.)


class GeometryTests(unittest.TestCase):
    def test_export_uses_explicit_motor_gains(self):
        robot = SimpleNamespace(device="cpu", joint_names=["a", "b"],
            actuators={"motor": SimpleNamespace(joint_indices=slice(None),
                stiffness=torch.tensor([[25., 25.]]), damping=torch.tensor([[.5, .5]]))})
        self.assertEqual(actuator_gains(robot), ([25., 25.], [.5, .5]))

    def test_stones_have_real_gaps(self):
        boxes = lane_boxes(.5, "stones")
        x, y = 2.6+.46+.0875/2, 1.5-.2125
        surfaces = [top for x0, x1, y0, y1, top, bottom in boxes if x0 < x < x1 and y0 < y < y1]
        self.assertAlmostEqual(max(surfaces), -.55)

    def test_all_boxes_are_positive_volume(self):
        for kind in ("flat", "stairs", "platform", "stones"):
            for difficulty in (0., .5, 1.):
                for x0, x1, y0, y1, top, bottom in lane_boxes(difficulty, kind):
                    self.assertTrue(x1 > x0 and y1 > y0 and top > bottom)


if __name__ == "__main__":
    unittest.main()
