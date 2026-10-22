import unittest
from xml.etree import ElementTree as ET
from go2_nav.terrains import populate


class TerrainProtocolTests(unittest.TestCase):
    def test_platform_requires_real_high_surface_and_exit(self):
        world=ET.fromstring('<worldbody><geom name="floor"/></worldbody>')
        meta=populate(world,'platform',dict(terrain_height=.2,terrain_return=True))
        self.assertEqual(meta['required_top_height'],.2)
        self.assertEqual(meta['goal_height'],0.)
        self.assertEqual(meta['surfaces'],[[-2,3.,0],[3.,5.,.2],[5.,10.,0.]])

    def test_stairs_include_descent_and_ground_destination(self):
        world=ET.fromstring('<worldbody><geom name="floor"/></worldbody>')
        meta=populate(world,'stairs',dict(terrain_height=.08,terrain_return=True))
        self.assertEqual(len(world.findall("geom")),11)
        self.assertAlmostEqual(meta['required_top_height'],.32)
        self.assertEqual(meta['goal_height'],0.)
        self.assertEqual(meta['goal'],[9.,0.])


if __name__=='__main__':unittest.main()
