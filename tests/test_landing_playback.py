import unittest
from tests.test_landing_contract import inputs
from skills.backflip.landing_contract import validate_landing_contract
from skills.backflip.landing_playback import LandingPlaybackState
from skills.backflip.simulate_landing import RawLandingAudit
from tests.test_landing_validation import trajectory


class LandingPlaybackTests(unittest.TestCase):
    def test_runtime_and_independent_audit_on_three_cycles(self):
        contract=validate_landing_contract(*inputs());runtime=LandingPlaybackState(contract,.01)
        audit=RawLandingAudit(.01,contract['settings']);rows=trajectory()
        for row in rows:
            runtime.upright=row['upright'];audit.upright=row['upright']
            args=(row['time'],row['foot_contact'],row['omega_body'],row['body_contact'])
            runtime.sample(*args);audit.sample(*args)
        self.assertEqual(int(runtime.state.completed[0]),3)
        self.assertTrue(audit.report(len(rows))['motion_pass'])
        self.assertEqual(float(runtime.clock[0]),2.)

    def test_training_hooks_and_playback_match_through_absorption_and_three_cycles(self):
        import numpy as np
        import torch
        from tests.test_backflip_rebound_landing_task import LandingTaskTests
        from skills.backflip.landing_contract import VERSION,implementation_hashes
        e=LandingTaskTests().make()
        c=dict(version=VERSION,phase_contract='landing_cycle_v2',settings=e.cfg.rebound,
               implementation_sha256=implementation_hashes(),execution_thresholds=dict(
                   takeoff_start=.5,recovery_contact_force=5.,unsafe_body_contact_force=5.))
        p=LandingPlaybackState(c,.005)
        frames=[(True,0.,1.)]*200
        for _ in range(3):
            frames += [(True,-15.,1.)]*20+[(False,-17.5,-1.)]*40
            frames += [(True,-10.,.5)]*10+[(True,-10.,.95)]*2
        frames += [(True,0.,1.)]*200
        visited=set()
        for i,(contact,pitch,upright) in enumerate(frames):
            if i%4==0:e._pre_physics_step(torch.zeros(2,12))
            e._sim_step_counter+=1;e._apply_action()
            # apply samples the completed previous physical substep; compare before
            # advancing the external physics inputs for the current substep.
            np.testing.assert_allclose(e._rebound_clock[0].item(),p.clock[0].item(),rtol=0,atol=1e-7)
            self.assertEqual(int(e._rebound.phase[0]),int(p.state.phase[0]))
            self.assertEqual(int(e._rebound.completed[0]),int(p.state.completed[0]))
            e._contact_sensor.data.net_forces_w[:,0,2]=10. if contact else 0.
            e._robot.data.root_ang_vel_b[:,1]=pitch
            e._robot.data.projected_gravity_b[:,2]=-upright
            p.upright=upright;p.sample((i+1)*.005,contact,[0.,pitch,0.])
            visited.add(int(p.state.phase[0]))
            if i%4==3:
                e._get_dones()
                expected=np.array([x[0,0].item() for x in e._phase_features()])
                np.testing.assert_allclose(p.phase_features(),expected,rtol=0,atol=1e-7)
                self.assertEqual(int(e._rebound.completed[0]),int(p.state.completed[0]))
        self.assertEqual(int(p.state.completed[0]),3)
        self.assertEqual(int(p.state.phase[0]),p.state.RECOVERY)
        self.assertIn(p.state.LANDING,visited)
        self.assertIn(p.state.REBOUND,visited)

    def test_orientation_and_source_provenance_required(self):
        c=validate_landing_contract(*inputs());r=LandingPlaybackState(c,.01)
        with self.assertRaises(ValueError):r.sample(.01,True,[0,0,0])
        c['implementation_sha256']={}
        with self.assertRaises(ValueError):LandingPlaybackState(c,.01)


if __name__=='__main__':unittest.main()
