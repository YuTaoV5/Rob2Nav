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

    def test_orientation_and_source_provenance_required(self):
        c=validate_landing_contract(*inputs());r=LandingPlaybackState(c,.01)
        with self.assertRaises(ValueError):r.sample(.01,True,[0,0,0])
        c['implementation_sha256']={}
        with self.assertRaises(ValueError):LandingPlaybackState(c,.01)


if __name__=='__main__':unittest.main()
