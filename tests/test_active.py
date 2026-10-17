import unittest
import mujoco
import numpy as np
from go2_nav.active_env import ActiveTerrainEnv
from go2_nav.terrain_env import terrain_config


class ActiveChecks(unittest.TestCase):
    def test_wide_pit_has_no_jump_authorization(self):
        cfg = terrain_config()
        cfg.update(gap_width=.7)
        env = ActiveTerrainEnv(cfg, scene="gap")
        try:
            env.reset(seed=200)
            env.data.qpos[0] = 1.8
            mujoco.mj_forward(env.model, env.data)
            env._sense()
            self.assertFalse(env.landing_confirmed)
            self.assertEqual(env.jump_count, 0)
        finally:
            env.close()

    def test_goal_is_user_supplied_and_flat_ground_does_not_request_lift(self):
        cfg = terrain_config()
        cfg.update(goal=[2., -1.], jump_backend="disabled")
        env = ActiveTerrainEnv(cfg, scene="empty")
        try:
            env.reset(seed=3)
            np.testing.assert_array_equal(env.goal, [2., -1.])
            for _ in range(8):
                env.step(np.array([.5, 0., 0.]))
            self.assertEqual(env.lift_count, 0)
        finally:
            env.close()

    def test_visible_step_changes_swing_targets_without_editing_robot_state(self):
        env = ActiveTerrainEnv(terrain_config(), scene="empty")
        try:
            env.reset(seed=3)
            # Airborne test fixture: swing corrections must not load stance feet.
            env.data.qpos[2] += .04
            mujoco.mj_forward(env.model, env.data)
            target = env.robot.default_dof_pos.copy()
            k = env.kinematics
            k.qpos[:] = env.data.qpos
            k.qpos[env.qids] = target
            mujoco.mj_forward(env.model, k)
            feet = k.geom_xpos[env.foot_ids].copy()
            env.last_nominal_feet = feet-np.array([0., 0., .01])
            patches = []
            for foot in env.data.geom_xpos[env.foot_ids]:
                xx, yy = np.meshgrid(np.linspace(.1, .3, 10), np.linspace(-.04, .04, 8))
                patches.append(np.c_[foot[0]+xx.ravel(), foot[1]+yy.ravel(), np.full(xx.size, env.support_z+.08)])
            env.measured_cloud = np.vstack(patches)
            before_pos, before_vel = env.data.qpos.copy(), env.data.qvel.copy()
            adjusted = env._lift_swing_feet(target)
            self.assertGreater(np.linalg.norm(adjusted-target), .01)
            np.testing.assert_array_equal(before_pos, env.data.qpos)
            np.testing.assert_array_equal(before_vel, env.data.qvel)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
