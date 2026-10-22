import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from skills.backflip.control import motor_torque,JOINT_NAMES
from rob2nav.deployment.core import Servo,NEUTRAL
from rob2nav.deployment.types import SDK_JOINTS,Snapshot,Skill
from rob2nav.deployment.atomic_validation import BackflipValidation


def contract():
    return dict(action_scale=.5,action_clip=8.,stiffness=40.,damping=1.,motor_velocity_x1=13.5,
                motor_torque_y1=20.2,motor_torque_y2=23.4,
                joint_velocity_limits=[30.1,30.1,15.7]*4,target_velocity_limits=[13.5]*12)


class UnifiedBackflipMotorTests(unittest.TestCase):
    def test_motoring_braking_and_zero_at_no_load_speed(self):
        c=contract();speed=np.zeros(12);speed[2]=14.6;raw=np.full(12,100.)
        self.assertAlmostEqual(motor_torque(raw,speed,c)[2],10.1)
        self.assertAlmostEqual(motor_torque(-raw,speed,c)[2],-11.7)
        speed[2]=15.7;self.assertEqual(motor_torque(raw,speed,c)[2],0.)
        self.assertEqual(motor_torque(raw,np.zeros(12),c)[0],23.4)
        self.assertEqual(motor_torque(-raw,np.zeros(12),c)[0],-23.4)

    def test_joint_order_and_sampled_target_slew(self):
        c=contract();speed=np.arange(12)*2.
        native=motor_torque(np.full(12,100.),speed,c)
        order=[JOINT_NAMES.index(n) for n in SDK_JOINTS]
        np.testing.assert_allclose(motor_torque(np.full(12,100.),speed[order],c,SDK_JOINTS),native[order])
        s=Snapshot(0,0,NEUTRAL.copy(),np.zeros(12),np.zeros(12),np.zeros(3),np.array([0,0,-1.]),np.ones(4))
        servo=Servo(s);cmd=servo.native(s,NEUTRAL+3,np.full(12,40.),np.ones(12),'ACTIVE',1,motor_control=c,dt=.002)
        np.testing.assert_allclose(cmd.qref-NEUTRAL,.027)

    def test_policy_contract_disagreement_rejected(self):
        s=Skill('backflip',SDK_JOINTS,tuple(NEUTRAL),(40.,)*12,(1.,)*12,'backflip60',50,.5,
                actuation='native_pd',action_parameterization='backflip_entry_v3',entry_hold_scale=.1,motor_control=contract()).checked()
        for update in [dict(scale=.25),dict(kp=(25.,)*12),dict(actuation='bounded_pd'),dict(action_parameterization='joint_offset_v1')]:
            with self.assertRaises(ValueError):replace(s,**update).checked()

    def test_entry_load_torque_is_continuous_then_removed(self):
        s=Snapshot(0,0,NEUTRAL.copy(),np.zeros(12),np.full(12,4.),np.zeros(3),np.array([0,0,-1.]),np.ones(4))
        servo=Servo(s);args=(s,NEUTRAL,np.full(12,40.),np.ones(12),'ACTIVE',1)
        first=servo.native(*args,motor_control=contract(),entry_blend_s=.2,entry_elapsed=0.)
        np.testing.assert_allclose(first.tau,4.)
        end=servo.native(*args,motor_control=contract(),entry_blend_s=.2,entry_elapsed=.2)
        np.testing.assert_allclose(end.tau,0.);np.testing.assert_allclose(end.feedforward,0.)

    def test_completed_switch_cannot_hide_incomplete_flip_or_bad_physics(self):
        # Synthetic measured summaries exercise the independent acceptance gate.
        check=BackflipValidation.__new__(BackflipValidation)
        check.skill=SimpleNamespace(id='backflip',atomic_duration_s=5.,motor_control=contract())
        check.start=10.;check.elapsed=5.;check.rotation=-2*np.pi;check.travel=2*np.pi
        check.max_air=.3;check.height=.6;check.minimum_up=-1.;check.bad=False;check.speed=.8;check.excess=0.
        check.trace=[dict(time=float(t),position=[0,0,.32],upright=1.,speed=0.,angular_speed=0.,
                          standard_pose_error=0.,max_joint_speed=0.,all_feet_ground=True) for t in np.arange(0,5.001,.02)]
        self.assertTrue(check.report()['passed'])
        for field,value in [('rotation',0.),('elapsed',2.),('bad',True),('speed',1.01),('excess',.01)]:
            original=getattr(check,field);setattr(check,field,value)
            self.assertFalse(check.report()['passed'],field);setattr(check,field,original)
        check.skill.motor_control=None;self.assertFalse(check.report()['passed'])


if __name__=='__main__':unittest.main()
