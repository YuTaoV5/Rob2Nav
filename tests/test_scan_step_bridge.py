import json,unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from skills.navigation.scan_client import ScanClient


class ScanStepBridgeTests(unittest.TestCase):
    def test_static_step_course_without_people_keeps_free_rays_and_elevated_goal(self):
        client=ScanClient.__new__(ScanClient);client.sock=Mock();client.sock.recvfrom.side_effect=BlockingIOError
        client.session='test';client.telemetry={};client.received=0.
        env=NS(scene='stairs_route',step_navigation=True,people=[],points=np.array([[1.,.4,.4]]),lidar_free_dirs=np.array([[1.,0.,0.]]),
            cfg=dict(lidar_position=[0,0,.18],max_range=8.,max_vx=.65,max_vy=.25,max_wz=1.2),
            rotation=np.eye(3),base_id=0,data=NS(xpos=np.array([[0.,0.,.35]]),qpos=np.array([0.,0.,.35,1.,0.,0.,0.]),qvel=np.zeros(6),time=10.),ground=lambda x:.16)
        np.testing.assert_array_equal(client.act(env,np.array([3.,0.]),False),np.zeros(3))
        packet=json.loads(client.sock.sendto.call_args.args[0])
        self.assertEqual(len(packet['points']),2)
        self.assertFalse(packet['dynamic_task'])
        self.assertTrue(packet['low_step_navigation'])
        self.assertAlmostEqual(packet['goal_height'],.51)
        self.assertGreater(packet['points'][1][0],8.)
        self.assertEqual(packet['goal_epoch'],2)


if __name__=='__main__':unittest.main()
