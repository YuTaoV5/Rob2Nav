import unittest
import numpy as np
import mujoco
from go2_nav.terrain_env import TerrainEnv, terrain_config
from go2_nav.terrain_planner import TerrainNavigator


class TerrainChecks(unittest.TestCase):
    def test_pit_is_a_physical_hole_and_edge_memory_survives_blind_zone(self):
        env = TerrainEnv(terrain_config(), scene="gap")
        try:
            env.reset(seed=0)
            geom = np.zeros(1, np.int32)
            distance = mujoco.mj_ray(env.model, env.data, np.array([3.15, 0., 1.]),
                np.array([0., 0., -1.]), env.ray_groups, 1, -1, geom)
            self.assertAlmostEqual(distance, 1.8, places=3)
            env.data.qpos[0] = 2.2
            mujoco.mj_forward(env.model, env.data)
            env._sense()
            self.assertIsNotNone(env.gap_edge_world)
            env.data.qpos[0] = 2.75
            mujoco.mj_forward(env.model, env.data)
            env._sense()
            self.assertLess(env.gap_distance, .4)
            self.assertGreater(env.gap_distance, 0)
        finally:
            env.close()

    def test_water_uses_colour_returns_and_jump_targets_do_not_teleport(self):
        env = TerrainEnv(terrain_config(), scene="water")
        try:
            env.reset(seed=0)
            self.assertGreater(len(env.level_water), 15)
            nav = TerrainNavigator(env.cfg)
            nav.act(env.points, env.level_water, env.data.qpos[:2], 0., env.goal, env.command)
            self.assertEqual(len(nav.waypoints), 2)
            before = env.data.qpos.copy()
            env.skill.trigger(float(env.data.time))
            target, kp, kd = env.modify_targets(env.robot.default_dof_pos.copy(), env.robot.p_gains, env.robot.d_gains, env.command)
            np.testing.assert_array_equal(before, env.data.qpos)
            self.assertTrue(np.all(target >= env.model.jnt_range[env.joint_ids, 0]))
            self.assertTrue(np.all(target <= env.model.jnt_range[env.joint_ids, 1]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
