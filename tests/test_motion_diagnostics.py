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


def test_peak_context_is_copied_and_never_stale():
    import numpy as np
    d=MotionDiagnostics(['calf'],[15.7],0)
    target=np.array([-1.2]);context={'peak_target_position_rad':target,'peak_actuator_generalized_torque_Nm':np.array([8.])}
    d.record(.001,.001,[-16.9],[0,0,0],[],.4,context)
    target[0]=-99
    assert d.report()['joint_speed_peaks']['calf']['peak_target_position_rad']==-1.2
    d.record(.002,.001,[-17.],[0,0,0],[],.4)
    assert 'peak_target_position_rad' not in d.report()['joint_speed_peaks']['calf']


def test_sample_reads_joint_ordered_state_and_forces_without_mutation(monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    import sys
    monkeypatch.setitem(sys.modules,'mujoco',NS()) # No physics is simulated in this unit fixture.
    model=NS(joint=lambda n:NS(id=['hip','calf'].index(n)),jnt_dofadr=np.array([6,7]),jnt_qposadr=np.array([7,8]),opt=NS(timestep=.001))
    data=NS(contact=[],qpos=np.arange(9,dtype=float)/10,qvel=np.array([0,0,0,0,0,0,2.,-17.]),
        qfrc_actuator=np.arange(8,dtype=float),qfrc_constraint=np.arange(8,dtype=float)*2,qfrc_bias=np.arange(8,dtype=float)*3)
    before={k:v.copy() for k,v in vars(data).items() if isinstance(v,np.ndarray)}
    d=MotionDiagnostics(['hip','calf'],[30.,15.7],0)
    d.sample(model,data,.001,np.array([6,7]),np.zeros(3),target=np.array([.2,-1.2]))
    peak=d.report()['joint_speed_peaks']['calf']
    assert peak['peak_joint_position_rad']==.8 and peak['peak_target_position_rad']==-1.2
    assert peak['peak_actuator_generalized_torque_Nm']==7.
    assert peak['peak_constraint_generalized_torque_Nm']==14.
    assert peak['peak_bias_generalized_torque_Nm']==21.
    for k,v in before.items():np.testing.assert_array_equal(getattr(data,k),v)
