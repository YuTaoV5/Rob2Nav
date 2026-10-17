import unittest
import numpy as np
from perceptive_rl.ame_terrain import heightfield, parameters, boxes


class AMETerrainTests(unittest.TestCase):
    def test_original_narrow_square_gap_and_platform(self):
        h = heightfield(1., "ame_stakes")
        self.assertEqual(h.shape,(160,160))
        self.assertTrue(np.all(h[60:100,60:100] == 0.))
        square = h[25:29,75:79]
        self.assertTrue(np.all(square == square[0,0]))
        self.assertTrue(np.all(h[29:34,75:79] == -2.))
        self.assertTrue(np.all(h[25:29,79:81] == -2.))
        self.assertTrue(np.array_equal(h,heightfield(1.,"ame_stakes")))

    def test_boxes_match_every_cell_top(self):
        for kind in ("ame_stakes", "ame_alternate", "ame_stakes12"):
            h = heightfield(1.,kind); scale = parameters(1.,kind)["grid"]
            reconstructed = np.full_like(h,-np.inf)
            for x0,x1,y0,y1,top,bottom in boxes(1.,kind):
                self.assertTrue(x1>x0 and y1>y0 and top>bottom)
                a,b,c,d = [round(x/scale) for x in (x0,x1,y0,y1)]
                reconstructed[a:b,c:d] = np.maximum(reconstructed[a:b,c:d],top)
            np.testing.assert_array_equal(h,reconstructed)
        self.assertEqual(parameters(1.,"ame_stakes12")["side"],.12)
