import unittest
import numpy as np
from go2_nav.env import Go2MaplessEnv
from go2_nav.planner import ReactivePlanner


class SimulationChecks(unittest.TestCase):
    def setUp(self):
        self.env = Go2MaplessEnv(scene="empty")

    def tearDown(self):
        self.env.close()

    def test_sensor_contract_and_control(self):
        obs, info = self.env.reset(seed=11)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(self.env.depth.shape, (24, 32))
        self.assertTrue(np.isfinite(self.env.depth).all())
        # The back wall at x=-2 must be visible to 360-degree sensing.
        self.assertLess(float(self.env.scan[0]), 2.5)
        start = self.env.data.qpos[0]
        for _ in range(30):
            obs, reward, done, truncated, info = self.env.step(np.array([.6, 0, 0]))
            self.assertFalse(done or truncated, info)
        self.assertGreater(self.env.data.qpos[0]-start, .3)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertLessEqual(np.max(np.abs(self.env.data.ctrl)-np.max(np.abs(self.env.model.actuator_ctrlrange), axis=1)), 1e-6)

    def test_blocked_controller_does_not_request_forward_collision(self):
        planner = ReactivePlanner(self.env.cfg)
        points = np.array([[.45, y, .3] for y in np.linspace(-1, 1, 40)])
        action = planner.act(points, np.array([5., 0]), np.zeros(3))
        self.assertEqual(action[0], 0.)

    def test_corridor_selection_turns_instead_of_stalling(self):
        planner = ReactivePlanner(self.env.cfg)
        points = np.array([[1.0, y, .3] for y in np.linspace(-.7, .7, 30)])
        action = planner.act(points, np.array([6., 0]), np.zeros(3))
        self.assertGreater(abs(action[2]), .1)


if __name__ == "__main__":
    unittest.main()
