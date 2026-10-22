import unittest
import numpy as np
from skills.sideflip.reflection import joint_map,observation_map,motion
from rob2nav.fixstand_validation import rotation_matches

NAMES=[f'{leg}_{part}_joint' for part in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]


class ReflectionTests(unittest.TestCase):
    def test_opposite_flip_is_not_accepted_as_requested_direction(self):
        self.assertTrue(rotation_matches(6.25,'right'))
        self.assertFalse(rotation_matches(-6.25,'right'))
        self.assertTrue(rotation_matches(-6.25,'left'))
        self.assertFalse(rotation_matches(.1,'right'))
    def test_action_and_observation_reflections_are_involutions(self):
        for p,s in [joint_map(NAMES),observation_map(NAMES)]:
            x=np.random.default_rng(1).normal(size=(8,len(p)))
            np.testing.assert_allclose((x[:,p]*s)[:,p]*s,x)

    def test_axial_velocity_and_orientation_signs(self):
        p,s=observation_map(NAMES)
        np.testing.assert_equal(s[30:33],[-1,1,-1])
        np.testing.assert_equal(s[24:30],[1,-1,-1,1,1,-1])

    def test_motion_reflection_preserves_height_and_reverses_roll(self):
        bodies=['base','FL_foot','FR_foot','RL_foot','RR_foot'];rng=np.random.default_rng(4)
        data=dict(fps=np.array([50]),joint_pos=rng.normal(size=(3,12)),joint_vel=rng.normal(size=(3,12)),
                  body_pos_w=rng.normal(size=(3,5,3)),body_lin_vel_w=rng.normal(size=(3,5,3)),
                  body_ang_vel_w=rng.normal(size=(3,5,3)),body_quat_w=np.tile([.8,.6,0.,0.],(3,5,1)))
        reflected=motion(data,NAMES,bodies);twice=motion(reflected,NAMES,bodies)
        for k in data:np.testing.assert_allclose(twice[k],data[k])
        np.testing.assert_allclose(reflected['body_pos_w'][:,0,2],data['body_pos_w'][:,0,2])
        self.assertEqual(reflected['body_quat_w'][0,0,1],-.6)


if __name__=='__main__':unittest.main()
