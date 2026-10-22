import dataclasses
import unittest
import tempfile,json,time
from pathlib import Path
import numpy as np
from rob2nav.deployment.types import Snapshot,Skill,Intent,SDK_JOINTS
from rob2nav.deployment.core import SwitchController,Servo,NEUTRAL
from rob2nav.deployment.policies import observation
from rob2nav.deployment.runner import UnifiedRuntime,Prepared
from rob2nav.deployment.sequence import SkillSequence,Step,load_steps
from unittest.mock import Mock
from rob2nav.deployment.unitree import Go2SdkTransport
from rob2nav.deployment.types import MotorCommand
from rob2nav.deployment.calibration import Calibration


def state(t=0.,sequence=0,**kw):
    d=dict(stamp=t,sequence=sequence,q=NEUTRAL.copy(),dq=np.zeros(12),tau=np.zeros(12),gyro=np.zeros(3),gravity=np.array([0.,0.,-1.]),feet=np.ones(4)*35,velocity=np.zeros(3),quaternion=np.array([1.,0,0,0]))
    d.update(kw);return Snapshot(**d)


def skill(id='stand',**kw):
    d=dict(id=id,joints=SDK_JOINTS,default=tuple(NEUTRAL),kp=(30.,)*12,kd=(1.5,)*12,observation='stand',policy_hz=50,scale=.25)
    d.update(kw);return Skill(**d)


class DeploymentCoreTests(unittest.TestCase):
    def test_compact_switch_enters_candidate_without_neutral_detour(self):
        c=SwitchController(compact_transitions=True);c.arm(state(),0)
        c.phase='ACTIVE';c.active=skill('stand');c.pending=skill('walk_moe',default=tuple(NEUTRAL+.03),actuation='native_pd')
        c.stable_since=0.;c.servo.kp[:]=30;c.servo.kd[:]=1.5
        c.tick(state(.4,1),.4,Intent(0,.4,NEUTRAL))
        self.assertEqual(c.phase,'FAULT')  # A compact route cannot bypass a control deadline.
        c=SwitchController(compact_transitions=True);c.arm(state(),0)
        c.phase='ACTIVE';c.active=skill('stand');c.pending=skill('walk_moe',default=tuple(NEUTRAL+.03),actuation='native_pd')
        c.stable_since=-1.;c.servo.kp[:]=30;c.servo.kd[:]=1.5
        c.tick(state(.002,1),.002,Intent(0,.002,NEUTRAL))
        self.assertEqual(c.phase,'ENTRY');self.assertLess(c.entry_duration,2.)
        self.assertTrue(any(e['event'].startswith('direct_entry:') for e in c.events))

    def test_entry_waits_for_actual_gain_convergence(self):
        c=SwitchController(compact_transitions=True);c.arm(state(),0)
        c.pending=skill('new',kp=(60.,)*12);c.phase='ENTRY';c.phase_at=-3
        c.start_pose=NEUTRAL.copy();c.stable_since=-1.;c.servo.kp[:]=30;c.servo.kd[:]=1.5
        c.tick(state(.002,1),.002)
        self.assertEqual(c.phase,'ENTRY')
        c.servo.kp[:]=60
        c.tick(state(.004,2),.004)
        self.assertEqual(c.phase,'WARMUP')

    def test_compact_routes_remain_disabled_for_hardware(self):
        self.assertFalse(SwitchController(hardware=True,compact_transitions=True).compact_transitions)

    def test_hardware_compact_route_requires_exact_qualified_pair(self):
        old=skill('stand');new=skill('walk_moe',default=tuple(NEUTRAL+.03))
        c=SwitchController(hardware=True,compact_transitions=True,qualified_compact_pairs=[(old.digest(),new.digest())])
        c.active=old;c.pending=new
        self.assertTrue(c.compact_entry_allowed())
        c.pending=skill('walk_moe',default=tuple(NEUTRAL+.04))
        self.assertFalse(c.compact_entry_allowed())

    def test_soft_stand_keeps_full_load_acquisition_entry(self):
        c=SwitchController(compact_transitions=True);c.arm(state(),0)
        c.pending=skill('stand_soft',kp=(15.,)*12,kd=(.5,)*12)
        c.begin_entry(.002)
        self.assertFalse(c.compact_entry_allowed());self.assertEqual(c.entry_duration,2.)

    def test_compact_entry_only_retries_neutral_once(self):
        c=SwitchController(compact_transitions=True);c.arm(state(),0)
        c.pending=skill('walk_moe',default=tuple(NEUTRAL+.3));c.start_pose=NEUTRAL.copy()
        c.phase='ENTRY';c.phase_at=-7;c.stable_since=-1
        c.tick(state(.002,1),.002)
        self.assertEqual(c.phase,'EXIT');self.assertTrue(c.entry_fallback)
        c.phase='ENTRY';c.phase_at=-7;c.stable_since=-1
        c.tick(state(.004,2),.004)
        self.assertEqual(c.phase,'FAULT')

    def test_preparation_bounds_metadata_and_closes_evicted_reference_handles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'deployment/skills';folder.mkdir(parents=True)
            (folder/'stand.json').write_text(json.dumps(dataclasses.asdict(skill())))
            r=UnifiedRuntime(root);old_reference=Mock()
            r.prepared=[Prepared(skill('old'),None,np.zeros(12),np.zeros(12),reference=old_reference)]
            try:
                r.stage_async('stand',state());r.future.result(timeout=2)
                self.assertEqual(len(r.prepared),1);old_reference.close.assert_called_once()
                self.assertEqual(r.model_history,{'stand':''})
            finally:r.close()

    def test_transport_rejects_sdk_false_return_without_opening_dds(self):
        transport=Go2SdkTransport('.')
        transport.publisher=Mock();transport.publisher.Write.return_value=False
        transport.codec=Mock();transport.read=Mock();transport.permit={'expires_unix':time.time()+100}
        command=MotorCommand(np.zeros(12),NEUTRAL,np.zeros(12),np.zeros(12),np.zeros(12),'READY',0,stamp=time.monotonic())
        with self.assertRaises(IOError):transport.write(command)
        transport.publisher.Write.return_value=True
        transport.write(dataclasses.replace(command,stamp=time.monotonic()))

    def test_joint_sign_zero_and_foot_unit_mapping_preserve_work(self):
        cal=Calibration(signs=tuple([1.,-1.]*6),offsets=(.05,)*12,foot_indices=(2,3,0,1),foot_scales=(2.,)*4,foot_offsets=(1.,)*4).checked()
        q=np.linspace(-.1,.2,12);dq=np.linspace(-1,1,12);tau=np.arange(12)*.1
        np.testing.assert_allclose(cal.position_from_sdk(cal.position_to_sdk(q)),q)
        self.assertAlmostEqual(float(tau@dq),float(cal.torque_to_sdk(tau)@cal.vector_from_sdk(dq)))
        np.testing.assert_array_equal(cal.forces_from_sdk([1,2,3,4]),[4,6,0,2])

    def test_reference_yaw_is_latched_at_entry_after_preload_turn(self):
        r=UnifiedRuntime('.')
        try:
            c=r.controller;c.arm(state(),0);c.phase='WARMUP';c.epoch=1
            contract=skill(observation='mimic69',atomic=True);c.pending=contract
            prepared=Prepared(contract,None,np.zeros(12),np.zeros(12),reference={'body_quat_w':np.array([[[1.,0,0,0]]])})
            r.next=prepared
            r.step(state(.002,1,quaternion=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])),.002)
            self.assertAlmostEqual(prepared.reference_yaw,np.pi/2)
            self.assertEqual(prepared.entry_epoch,1)
        finally:
            if r.next is not None:r.next.reference=None
            r.close()

    def test_failed_candidate_does_not_replace_or_stop_current_stand(self):
        with tempfile.TemporaryDirectory() as temp:
            r=UnifiedRuntime(temp)
            try:
                r.arm(state(),0);r.controller.phase='ACTIVE';r.controller.epoch=1
                current=skill();r.controller.active=current;r.active=Prepared(current,None,np.zeros(12),np.zeros(12))
                r.stage_async('missing',state())
                with self.assertRaises(FileNotFoundError):r.future.result(timeout=2)
                command=r.step(state(.002,1),.002)
                self.assertIs(r.controller.active,current);self.assertEqual(command.phase,'ACTIVE')
                self.assertIsNotNone(r.stage_error)
            finally:r.close()

    def test_sequence_queues_return_but_does_not_interrupt_atomic_action(self):
        c=SwitchController();c.arm(state(),0);c.phase='ACTIVE';c.epoch=1
        gait=skill('walk');flip=skill('flip',atomic=True);stand=skill()
        contracts={'walk':gait,'flip':flip,'stand':stand}
        r=Mock();r.controller=c;r.profile=lambda name:(None,contracts[name]);r.stage_error=None;r.future=None;r.next=None;r.queued=None
        seq=SkillSequence(r,[Step('walk',seconds=.01),Step('flip',trigger=True),Step('walk',seconds=.01)])
        c.active=gait;seq.tick(state(),0);seq.tick(state(.02,1),.02)
        self.assertEqual(seq.index,1)
        seq.tick(state(.022,2),.022);r.stage_async.assert_called_once()
        c.active=flip;c.phase='ARMED';seq.tick(state(.024,3),.024)
        self.assertTrue(c.trigger_requested)
        c.phase='ACTIVE';seq.tick(state(.026,4),.026);seq.tick(state(.028,5),.028)
        self.assertEqual(r.stage_async.call_args.args[0],'walk')
        self.assertIs(c.active,flip);self.assertEqual(c.phase,'ACTIVE')
        with self.assertRaises(ValueError):SkillSequence(r,[Step('flip')])
        with self.assertRaises(ValueError):Step('walk',velocity=(3,0,0)).checked()

    def test_long_sequence_keeps_stair_gait_until_level_handover(self):
        c=SwitchController();c.arm(state(),0);c.phase='ACTIVE';c.active=skill('walk_moe')
        contracts={'walk_moe':c.active,'stand':skill()}
        r=Mock();r.controller=c;r.profile=lambda name:(None,contracts[name]);r.stage_error=None;r.future=None;r.next=None;r.queued=None
        seq=SkillSequence(r,[Step('walk_moe',seconds=.01,velocity=(.2,0,0),handover_extension_s=1.),Step('stand')])
        tilted=state(.02,1,gravity=np.array([.34,0,-np.sqrt(1-.34**2)]))
        seq.tick(tilted,0);seq.tick(tilted,.02)
        self.assertEqual(seq.index,0);self.assertAlmostEqual(c.requested_command[0],.2)
        seq.tick(state(.04,2),.04)
        self.assertEqual(seq.index,1)
        seq.tick(state(.06,3),.06)
        r.stage_async.assert_called_once()

    def test_flat_to_stairs_same_moe_keeps_model_and_history(self):
        c=SwitchController();c.arm(state(),0);c.phase='ACTIVE';c.active=skill('walk_moe')
        r=Mock();r.controller=c;r.profile=lambda name:(None,c.active if name=='walk_moe' else skill())
        r.stage_error=None;r.future=None;r.next=None;r.queued=None
        seq=SkillSequence(r,[Step('walk_moe',seconds=.01,velocity=(.2,0,0)),
            Step('walk_moe',seconds=.1,velocity=(.18,0,0)),Step('stand')])
        tilted=state(.02,1,gravity=np.array([.34,0,-np.sqrt(1-.34**2)]))
        seq.tick(tilted,0);seq.tick(tilted,.02);seq.tick(tilted,.04)
        self.assertEqual(seq.index,1)
        self.assertAlmostEqual(c.requested_command[0],.18)
        r.stage_async.assert_not_called()

    def test_supported_integral_does_not_leak_to_inaccurate_nominal_load(self):
        s=state(support_torque=np.full(12,-5.))
        servo=Servo(s);servo.bias=np.full(12,2.)
        servo.step(s,NEUTRAL,np.full(12,15.),np.full(12,.5),.002,'ENTRY',0,integrate=True)
        np.testing.assert_array_equal(servo.bias,np.full(12,2.))
        servo.q=NEUTRAL+.01
        servo.step(s,NEUTRAL+.01,np.full(12,15.),np.full(12,.5),.002,'ENTRY',0,integrate=True)
        self.assertTrue(np.all(servo.bias>2.))

    def test_atomic_requires_explicit_trigger_and_stable_entry(self):
        c=SwitchController();c.arm(state(),0)
        c.active=skill('flip',atomic=True);c.phase='ARMED';c.epoch=1
        for i in range(1,301):
            t=i*.002;c.tick(state(t,i),t,Intent(1,t,NEUTRAL))
        self.assertEqual(c.phase,'ARMED')
        c.trigger()
        c.tick(state(.602,301,feet=np.zeros(4)),.602,Intent(1,.602,NEUTRAL))
        self.assertEqual(c.phase,'ARMED')
        for i in range(302,503):
            t=i*.002;c.tick(state(t,i),t,Intent(1,t,NEUTRAL))
        self.assertEqual(c.phase,'ACTIVE')

    def test_velocity_intent_expires_and_ramps_down(self):
        c=SwitchController();c.arm(state(),0);c.active=skill();c.phase='ACTIVE';c.epoch=1
        c.set_velocity([.2,0,0],0)
        for i in range(1,101):
            t=i*.002;c.tick(state(t,i),t,Intent(1,t,NEUTRAL))
        self.assertGreater(c.command[0],.09)
        for i in range(101,301):
            t=i*.002;c.tick(state(t,i),t,Intent(1,t,NEUTRAL))
        np.testing.assert_allclose(c.command,0,atol=1e-12)

    def test_asynchronous_stand_preparation_finishes_without_touching_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'deployment/skills';folder.mkdir(parents=True)
            (folder/'stand.json').write_text(json.dumps(dataclasses.asdict(skill())))
            r=UnifiedRuntime(root)
            try:
                r.stage_async('stand',state());prepared=r.future.result(timeout=2)
                self.assertEqual(prepared.skill.id,'stand');self.assertEqual(r.controller.phase,'OFF')
            finally:r.close()
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
