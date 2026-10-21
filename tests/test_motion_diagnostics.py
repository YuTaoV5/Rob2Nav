import unittest
from rob2nav.motion_diagnostics import MotionDiagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_substeps_peaks_contacts_and_axis_accounting(self):
        d=MotionDiagnostics(['hip','calf'],[30.,15.],0)
        d.record(.1,.1,[6.,-16.5],[3.,4.,0.],[('base',12.,-.001),('base',14.,-.002),('FL_foot',30.,-.003)],.4)
        d.record(.2,.1,[31.5,0.],[3.,0.,0.],[],.3)
        r=d.report()
        self.assertAlmostEqual(r['joint_speed_peaks']['calf']['max_ratio'],1.1)
        self.assertEqual(r['joint_speed_peaks']['hip']['peak_time_s'],.2)
        self.assertEqual(r['joint_speed_peaks']['calf']['overspeed_seconds'],.1)
        self.assertEqual(set(r['nonfoot_ground_contacts']),{'base'})
        self.assertEqual(r['nonfoot_ground_contacts']['base']['duration_s'],.1)
        self.assertEqual(r['nonfoot_ground_contacts']['base']['max_normal_force_N'],14.)
        self.assertAlmostEqual(r['time_bins'][0]['signed_axis_rad'],.6)
        self.assertAlmostEqual(r['time_bins'][0]['angular_travel_rad'],.8)
        self.assertAlmostEqual(r['time_bins'][0]['off_axis_travel_rad'],.4)


if __name__=='__main__':unittest.main()
