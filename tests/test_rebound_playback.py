import unittest
import numpy as np
from tests.test_backflip_rebound_task import fake_class, config
from skills.backflip.rebound_task import configure
from skills.backflip.rebound_playback import ReboundPlaybackState
from skills.backflip.rebound_contract import VERSION, implementation_hashes


class PlaybackParityTests(unittest.TestCase):
    def test_every_actor_phase_matches_training_through_three_landings(self):
        cls=fake_class();cfg=config();configure(cfg,3,.45,cls);e=cls(cfg)
        c=dict(version=VERSION,phase_contract='contact_cycle_v1',settings=cfg.rebound,
               implementation_sha256=implementation_hashes(),execution_thresholds=dict(
                   takeoff_start=.5,recovery_contact_force=5.,unsafe_body_contact_force=5.))
        p=ReboundPlaybackState(c,.005)
        frames=[(True,0.)]*200
        for i in range(3):frames += [(False,-6.28)]*200+[(True,0.)]*(40 if i<2 else 200)
        for i,(contact,pitch) in enumerate(frames):
            if i%4==0:e._pre_physics_step(None)
            e._sim_step_counter+=1;e._apply_action()
            e._contact_sensor.data.net_forces_w[:,0,2]=10. if contact else 0.
            e._robot.data.root_ang_vel_b[:,1]=pitch
            p.sample((i+1)*.005,contact,[0.,pitch,0.])
            if i%4==3:
                e._get_dones()
                expected=np.array([x[0,0].item() for x in e._phase_features()])
                np.testing.assert_allclose(p.phase_features(),expected,rtol=0,atol=1e-7)
                self.assertEqual(int(p.state.phase[0]),int(e._rebound.phase[0]))
                self.assertEqual(int(p.state.completed[0]),int(e._rebound.completed[0]))
        self.assertEqual(int(p.state.completed[0]),3)
        self.assertEqual(int(p.state.phase[0]),p.state.RECOVERY)
        with self.assertRaises(ValueError):p.sample(len(frames)*.005,True,[0.,0.,0.])


if __name__=='__main__':unittest.main()
