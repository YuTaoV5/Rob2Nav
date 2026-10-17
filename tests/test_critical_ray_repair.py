import json
import sys
import types
import unittest
from unittest.mock import patch
import torch
from critical_ame.repair import install


class RayRepairTests(unittest.TestCase):
    def test_valid_values_preserved_and_unrecoverable_values_stay_invalid(self):
        class Sensor:
            meshes={'ground':object()}
            def _update_buffers_impl(self,ids):
                self._data.ray_hits_w=torch.tensor([[[0.,0.,.2],[float('inf')]*3,[float('inf')]*3]])
        sensor=Sensor()
        sensor.cfg=types.SimpleNamespace(prim_path='/Robot/base',mesh_prim_paths=['ground'],max_distance=100.)
        sensor._data=types.SimpleNamespace(ray_hits_w=torch.empty(1,3,3))
        sensor._ray_starts_w=torch.tensor([[[0.,0.,2.],[1.,0.,2.],[2.,0.,2.]]])
        sensor._ray_directions_w=torch.tensor([[[0.,0.,-1.]]*3])
        def raycast(starts,directions,**kwargs):
            hits=torch.full_like(starts,float('inf'))
            good=(starts[:,0]-1.00005).abs()<1e-7
            hits[good]=starts[good];hits[good,2]=.3
            return hits,None,None,None
        sensors=types.ModuleType('isaaclab.sensors');sensors.RayCaster=Sensor
        warp=types.ModuleType('isaaclab.utils.warp');warp.raycast_mesh=raycast
        modules={'isaaclab':types.ModuleType('isaaclab'),'isaaclab.utils':types.ModuleType('isaaclab.utils'),
                 'isaaclab.sensors':sensors,'isaaclab.utils.warp':warp}
        with patch.dict(sys.modules,modules),patch('critical_ame.repair.Path') as paths:
            stats=install('unused-test-output')
            sensor._update_buffers_impl(torch.tensor([0]))
            torch.testing.assert_close(sensor._data.ray_hits_w[0,0],torch.tensor([0.,0.,.2]),atol=0,rtol=0)
            self.assertTrue(torch.isfinite(sensor._data.ray_hits_w[0,1]).all())
            self.assertFalse(torch.isfinite(sensor._data.ray_hits_w[0,2]).any())
            self.assertEqual((stats['missing_rays'],stats['recovered_rays'],stats['unrecovered_rays']),(2,1,1))
            written=paths.return_value.__truediv__.return_value.write_text.call_args.args[0]
            self.assertEqual(json.loads(written)['unrecovered_rays'],1)
