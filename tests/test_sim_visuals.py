import unittest
import numpy as np
import mujoco
from studio.sim_visuals import reference_ground


class VisualReferenceTests(unittest.TestCase):
    def test_visuals_do_not_change_physics_or_sensors(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="0 0 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        before=(d.qpos.copy(),d.qvel.copy(),m.geom_rgba.copy(),m.geom_friction.copy(),m.body_mass.copy(),d.time,d.ncon)
        scn=mujoco.MjvScene(m,maxgeom=1000);cam=mujoco.MjvCamera();opt=mujoco.MjvOption();counts=[]
        for _ in range(3):
            mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL,scn)
            reference_ground(scn,m,d);counts.append(scn.ngeom)
        self.assertGreater(counts[0],100);self.assertEqual(len(set(counts)),1)
        for a,b in zip(before,(d.qpos,d.qvel,m.geom_rgba,m.geom_friction,m.body_mass,d.time,d.ncon)):np.testing.assert_array_equal(a,b)

    def test_small_scene_capacity_is_respected(self):
        m=mujoco.MjModel.from_xml_string('<mujoco/>');d=mujoco.MjData(m);scn=mujoco.MjvScene(m,maxgeom=5)
        reference_ground(scn,m,d);self.assertEqual(scn.ngeom,5)


if __name__=='__main__':unittest.main()
