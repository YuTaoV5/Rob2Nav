import unittest
import numpy as np
from moe_perceptive import sparse_course as course
from perceptive_rl import ame_terrain as original


class DeepCourseTest(unittest.TestCase):
    def test_real_pits_and_final_original_identity(self):
        for kind in ('ame_stakes','ame_alternate'):
            for d in (0.,.25,.5,.75,1.):
                h=course.heightfield(d,kind)
                self.assertTrue((h==-2.).any())
                self.assertTrue(np.all((h==-2.)|(h>=-.03)))
                self.assertEqual(course.boxes(d,kind)[0],(0.,8.,0.,8.,-2.,-2.2))
            np.testing.assert_array_equal(course.heightfield(1.,kind),original.heightfield(1.,kind))
            self.assertEqual(course.boxes(1.,kind),original.boxes(1.,kind))

    def test_precursor_really_widens_support(self):
        for kind in ('ame_stakes','ame_alternate'):
            self.assertGreater((course.heightfield(0.,kind)>-1).sum(),(original.heightfield(0.,kind)>-1).sum())


if __name__=='__main__':unittest.main()
