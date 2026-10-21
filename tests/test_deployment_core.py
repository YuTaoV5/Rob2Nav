import dataclasses
import unittest
import numpy as np
from rob2nav.deployment.types import Snapshot,Skill,Intent,SDK_JOINTS
from rob2nav.deployment.core import SwitchController,Servo,NEUTRAL
from rob2nav.deployment.policies import observation


def state(t=0.,sequence=0,**kw):
    d=dict(stamp=t,sequence=sequence,q=NEUTRAL.copy(),dq=np.zeros(12),tau=np.zeros(12),gyro=np.zeros(3),gravity=np.array([0.,0.,-1.]),feet=np.ones(4)*35,velocity=np.zeros(3),quaternion=np.array([1.,0,0,0]))
    d.update(kw);return Snapshot(**d)


def skill(id='stand',**kw):
    d=dict(id=id,joints=SDK_JOINTS,default=tuple(NEUTRAL),kp=(30.,)*12,kd=(1.5,)*12,observation='stand',policy_hz=50,scale=.25)
    d.update(kw);return Skill(**d)


class DeploymentCoreTests(unittest.TestCase):
    def test_mapping_is_by_name_and_requires_bijection(self):
        names=SDK_JOINTS[::-1];s=skill(joints=names)
        x=np.arange(12);np.testing.assert_array_equal(s.to_sdk(s.from_sdk(x)),x)
        with self.assertRaises(ValueError):skill(joints=(SDK_JOINTS[0],)*12).checked()
    def test_abrupt_pd_and_pose_requests_have_bounded_torque_and_gain_slew(self):
        s=state();servo=Servo(s);before=servo.tau.copy();oldkp=servo.kp.copy()
        for _ in range(2000):
            c=servo.step(s,NEUTRAL+.3,np.ones(12)*60,np.ones(12)*5,.002,'ENTRY',1)
            self.assertLessEqual(float(np.max(abs(c.tau-before))),.300000001)
            self.assertLessEqual(float(np.max(abs(c.kp-oldkp))),.030000001)
            self.assertLessEqual(float(np.max(abs(c.dqref))),2.000000001)
            before=c.tau.copy();oldkp=c.kp.copy()
    def test_stale_future_nan_state_and_nonmonotonic_clock_rejected(self):
        for bad in [state(stamp=-1),state(stamp=1),state(q=np.full(12,np.nan)),state(gravity=np.zeros(3))]:
            with self.assertRaises(ValueError):bad.checked(0)
        c=SwitchController();c.arm(state(),0)
        with self.assertRaises(ValueError):c.tick(state(),0)
        result=c.tick(state(.002,1,stamp=-1),.002)
        self.assertEqual(c.phase,'FAULT');self.assertIn('stale',result.fault)
    def test_old_policy_epoch_and_late_output_cannot_activate(self):
        for intent in [Intent(0,.002,NEUTRAL),Intent(1,-1,NEUTRAL)]:
            c=SwitchController();c.arm(state(),0);c.phase='ACTIVE';c.active=skill();c.epoch=1
            c.tick(state(.002,1),.002,intent)
            self.assertEqual(c.phase,'FAULT')
    def test_airborne_atomic_action_defers_switch(self):
        c=SwitchController();c.arm(state(),0);c.phase='ACTIVE';c.active=skill('atomic-test',atomic=True);c.epoch=1
        c.request(skill('next'),0)
        for i in range(1,201):
            t=i*.002;c.tick(state(t,i,feet=np.zeros(4)),t,Intent(1,t,NEUTRAL,finished=False))
        self.assertEqual(c.phase,'ACTIVE');self.assertEqual(c.active.id,'atomic-test')
        for i in range(201,401):
            t=i*.002;c.tick(state(t,i),t,Intent(1,t,NEUTRAL,finished=False))
        self.assertEqual(c.phase,'ACTIVE')
        t=.802;c.tick(state(t,401),t,Intent(1,t,NEUTRAL,finished=True))
        self.assertEqual(c.phase,'EXIT')
    def test_hardware_and_known_failed_skill_gates(self):
        for hardware,s in [(True,skill()),(False,skill(simulation_enabled=False))]:
            c=SwitchController(hardware);c.arm(state(),0);c.phase='READY'
            with self.assertRaises(ValueError):c.request(s,0)
        c=SwitchController();c.arm(state(),0);c.phase='READY'
        with self.assertRaises(ValueError):c.request(skill(entry='requires_exit_controller'),0)
    def test_body_velocity_required_and_observation_contracts_differ(self):
        s=state();zero=np.zeros(12)
        self.assertEqual(len(observation(skill(observation='robogauge45'),s,[0,0,0],zero)),45)
        self.assertEqual(len(observation(skill(observation='backflip60'),s,[0,0,0],zero)),60)
        self.assertEqual(len(observation(skill(observation='upright48'),s,[0,0,0],zero)),48)
        with self.assertRaises(ValueError):observation(skill(observation='upright48'),dataclasses.replace(s,velocity=None),[0,0,0],zero)
    def test_watchdogs_remain_latched(self):
        c=SwitchController();c.arm(state(),0);c.tick(state(.02,1),.02)
        self.assertEqual(c.phase,'FAULT')
        c.tick(state(.022,2),.022);self.assertEqual(c.phase,'FAULT')
        with self.assertRaises(ValueError):c.request(skill(),.022)


if __name__=='__main__':unittest.main()
