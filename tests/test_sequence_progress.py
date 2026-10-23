"""Timing must describe the existing controller without shortening its gates."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from rob2nav.deployment.sequence import SkillSequence,Step


class SequenceProgressTests(unittest.TestCase):
    def setup_sequence(self,steps):
        contracts={s.skill:NS(id=s.skill,atomic=s.trigger) for s in steps}
        c=NS(phase='ENTRY',active=None,set_velocity=Mock(),trigger_requested=False,epoch=0)
        r=NS(controller=c,profile=lambda n:(None,contracts[n]),stage_error=None,
            future=None,next=None,queued=None,stage_async=Mock())
        seq=SkillSequence(r,steps)
        self.snapshot=NS(gravity=np.array([0.,0.,-1.]),gyro=np.zeros(3),contacts=lambda:np.ones(4))
        return seq,c,contracts

    def test_preparation_does_not_spend_dwell(self):
        seq,c,contracts=self.setup_sequence([Step('stand',2)])
        seq.tick(self.snapshot,3)
        p=seq.progress(3)
        self.assertEqual((p['dwell_elapsed_s'],p['extra_elapsed_s'],p['status']),(0,3,'transition'))
        c.phase='ACTIVE';c.active=contracts['stand'];seq.tick(self.snapshot,4)
        self.assertEqual(seq.progress(5)['dwell_elapsed_s'],1)
        seq.tick(self.snapshot,6)
        p=seq.progress(6)
        self.assertTrue(seq.done)
        self.assertEqual((p['dwell_elapsed_s'],p['extra_elapsed_s']),(2,4))
        seq.tick(self.snapshot,7)
        self.assertEqual(seq.progress(7)['dwell_elapsed_s'],2)

    def test_two_steps_keep_separate_dwell_across_switch(self):
        seq,c,contracts=self.setup_sequence([Step('walk',2),Step('stand',3)])
        c.phase='ACTIVE';c.active=contracts['walk'];seq.tick(self.snapshot,1);seq.tick(self.snapshot,3)
        c.phase='ENTRY';seq.tick(self.snapshot,5)
        self.assertEqual(seq.progress(5)['dwell_elapsed_s'],2)
        c.phase='ACTIVE';c.active=contracts['stand'];seq.tick(self.snapshot,6)
        p=seq.progress(7)
        self.assertEqual((p['step_index'],p['step_dwell_s'],p['dwell_elapsed_s']),(1,1,3))
        self.assertAlmostEqual(p['dwell_elapsed_s']+p['extra_elapsed_s'],p['execution_elapsed_s'])

    def test_same_skill_starts_a_new_dwell(self):
        seq,c,contracts=self.setup_sequence([Step('stand',1),Step('stand',2)])
        c.phase='ACTIVE';c.active=contracts['stand'];seq.tick(self.snapshot,0);seq.tick(self.snapshot,1)
        seq.tick(self.snapshot,1.01)
        self.assertAlmostEqual(seq.progress(2.01)['dwell_elapsed_s'],2)

    def test_repeated_atomic_requires_a_new_epoch(self):
        seq,c,contracts=self.setup_sequence([Step('flip',trigger=True),Step('flip',trigger=True),Step('stand')])
        c.phase='ACTIVE';c.active=contracts['flip'];c.epoch=7
        seq.tick(self.snapshot,1)
        self.assertEqual(seq.index,1)
        seq.tick(self.snapshot,1.02)
        seq.runtime.stage_async.assert_called_once_with('flip',self.snapshot)
        seq.tick(self.snapshot,2)
        self.assertEqual(seq.index,1)
        c.phase='ENTRY';seq.tick(self.snapshot,3)
        self.assertEqual(seq.index,1)
        c.phase='ACTIVE';c.epoch=8;seq.tick(self.snapshot,4)
        self.assertEqual(seq.index,2)
        self.assertEqual(seq.consumed_atomic_epoch,8)

    def test_unstable_handover_is_extra_time(self):
        seq,c,contracts=self.setup_sequence([Step('walk',1),Step('stand',2)])
        c.phase='ACTIVE';c.active=contracts['walk'];seq.tick(self.snapshot,0)
        self.snapshot.gravity=np.array([1.,0.,0.]);seq.tick(self.snapshot,2)
        p=seq.progress(2)
        self.assertEqual((p['status'],p['dwell_elapsed_s'],p['extra_elapsed_s']),('handover',1,1))
        self.assertEqual(seq.index,0)

    def test_atomic_duration_is_not_a_dwell_timer(self):
        seq,c,contracts=self.setup_sequence([Step('flip',seconds=99,trigger=True),Step('stand',2)])
        c.phase='ACTIVE';c.active=contracts['flip'];seq.tick(self.snapshot,1)
        p=seq.progress(4)
        self.assertEqual((p['status'],p['step_index'],p['planned_dwell_s'],p['dwell_elapsed_s']),('atomic',0,2,0))
        self.assertIsNone(p['step_target_s'])
        self.assertEqual(seq.index,1)  # Return is queued; the motion still owns the controller.

    def test_fault_keeps_partial_dwell_and_reports_failure(self):
        seq,c,contracts=self.setup_sequence([Step('stand',2)])
        c.phase='ACTIVE';c.active=contracts['stand'];seq.tick(self.snapshot,0)
        c.phase='FAULT';c.fault='test';seq.tick(self.snapshot,1)
        self.assertEqual(seq.progress(1)['status'],'failed')


if __name__=='__main__':unittest.main()
