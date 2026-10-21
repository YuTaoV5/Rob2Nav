"""Offline numerical checks against the FIXSTAND training schedules."""
from dataclasses import replace
import unittest
import numpy as np

from rob2nav.deployment.core import NEUTRAL
from rob2nav.deployment.keys import key_request
from rob2nav.deployment.policies import action_target,observation,reference_action_frame,history_action
from rob2nav.deployment.types import Skill,SDK_JOINTS,Snapshot


def skill(mode,observation_name):
    return Skill(id='test',joints=SDK_JOINTS,default=tuple(NEUTRAL),kp=(40.,)*12,kd=(1.,)*12,
        observation=observation_name,policy_hz=50,scale=.5,action_parameterization=mode,
        reference='motion.npz' if mode.startswith('reference_residual_') else '',
        residual_limit_rad=.2 if mode.startswith('reference_residual_') else 0.,
        bridge_residual_limit_rad=.05 if mode.startswith('reference_residual_') else 0.,
        terminal_hip_taper_frames=50 if mode=='reference_residual_v3' else 0).checked()


class FixstandDeploymentTests(unittest.TestCase):
    def test_backflip_delayed_entry_and_phase(self):
        s=skill('backflip_entry_v2','backflip60');raw=np.ones(12)
        np.testing.assert_allclose(action_target(s,raw,0.),NEUTRAL+.2*np.tanh(raw))
        np.testing.assert_allclose(action_target(s,raw,1.2),NEUTRAL+.5*raw)
        snapshot=Snapshot(0,0,NEUTRAL,np.zeros(12),np.zeros(12),np.zeros(3),np.array([0,0,-1]),np.ones(4))
        obs=observation(s,snapshot,[0,0,0],np.zeros(12),phase=0.)
        self.assertAlmostEqual(float(obs[-6]),-1.)
        quiet=replace(s,action_parameterization='backflip_entry_v3',entry_hold_scale=.1).checked()
        np.testing.assert_allclose(action_target(quiet,raw,0.),NEUTRAL+.05*np.tanh(raw))
        np.testing.assert_allclose(history_action(quiet,raw,0.),.1*np.tanh(raw))
        np.testing.assert_allclose(action_target(quiet,np.ones(12)*20,2.),NEUTRAL+4.)

    def test_actor_history_is_converted_to_sdk_order_once(self):
        s=replace(skill('upright_endpoint_v3','upright52'),joints=tuple(reversed(SDK_JOINTS))).checked()
        raw=np.arange(12,dtype=float)
        np.testing.assert_equal(s.from_sdk(history_action(s,raw,3.)),raw)

    def test_upright_stance_action_and_phase(self):
        s=skill('upright_endpoint_v2','upright52');raw=np.ones(12)
        np.testing.assert_allclose(action_target(s,raw,0.),NEUTRAL+.2*np.tanh(raw))
        np.testing.assert_allclose(action_target(s,raw,3.),NEUTRAL+.5)
        np.testing.assert_allclose(action_target(s,raw,7.),NEUTRAL+.2*np.tanh(raw))
        snapshot=Snapshot(0,0,NEUTRAL,np.zeros(12),np.zeros(12),np.zeros(3),np.array([0,0,-1]),np.ones(4),velocity=np.zeros(3))
        np.testing.assert_allclose(observation(s,snapshot,[0,0,0],np.zeros(12),phase=5.5)[-4:],[0,0,1,5.5/8])
        tight=replace(s,action_parameterization='upright_endpoint_v3',endpoint_residual_limit_rad=.05).checked()
        np.testing.assert_allclose(action_target(tight,raw,0.),NEUTRAL+.05*np.tanh(raw))

    def test_sideflip_reference_boundary_and_raw_middle(self):
        s=skill('reference_residual_v2','mimic69');raw=np.ones(12)
        reference={'fps':np.array([50.]),'joint_pos':np.tile(NEUTRAL+.1,(300,1))}
        np.testing.assert_allclose(action_target(s,raw,0.,reference),NEUTRAL+.1+.2*np.tanh(raw))
        self.assertLess(np.max(abs(action_target(s,raw,1.78,reference)-(NEUTRAL+.1+.05*np.tanh(raw)))),.001)
        np.testing.assert_allclose(action_target(s,raw,3.,reference),NEUTRAL+.5)
        frame=reference_action_frame(reference,0.)
        np.testing.assert_allclose(action_target(s,raw,0.,frame),action_target(s,raw,0.,reference))

    def test_sideflip_terminal_taper_changes_only_hips(self):
        s=skill('reference_residual_v3','mimic69');raw=np.ones(12)
        reference={'fps':np.array([50.]),'joint_pos':np.tile(NEUTRAL+.1,(300,1))}
        v2=skill('reference_residual_v2','mimic69')
        target=action_target(s,raw,5.98,reference)
        original=action_target(v2,raw,5.98,reference)
        hip=np.array([name.endswith('_hip_joint') for name in SDK_JOINTS])
        np.testing.assert_allclose(target[hip],(NEUTRAL+.1)[hip])
        np.testing.assert_allclose(target[~hip],original[~hip])

    def test_invalid_contracts_rejected(self):
        with self.assertRaises(ValueError):skill('upright_endpoint_v2','upright48')
        with self.assertRaises(ValueError):replace(skill('joint_offset_v1','backflip60'),residual_limit_rad=.2).checked()
        with self.assertRaises(ValueError):replace(skill('reference_residual_v2','mimic69'),terminal_hip_taper_frames=50).checked()
        with self.assertRaises(ValueError):replace(skill('reference_residual_v3','mimic69'),boundary_blend_frames=10).checked()

    def test_continuous_reference_boundary_matches_training_and_keeps_midair_raw(self):
        from skills.fixstand import sideflip_native_weight,sideflip_boundary_limit,terminal_hip_weight
        s=replace(skill('reference_residual_v3','mimic69'),action_parameterization='reference_residual_v4',boundary_blend_frames=10).checked()
        raw=np.linspace(-1.,1.,12)
        hip=np.array([n.endswith('_hip_joint') for n in s.joints])
        for frame in [0,79,80,89,90,91,100,150,200,209,210,211,220,299]:
            reference=dict(frame=frame,count=300,joint_pos=(NEUTRAL+.1).tolist())
            weight=float(sideflip_native_weight(frame,300,10))
            res=float(sideflip_boundary_limit(frame,300,.2,.05))*np.tanh(raw)
            res[hip]*=float(terminal_hip_weight(frame,300,50))
            expected=(1.-weight)*(NEUTRAL+.1+res)+weight*(NEUTRAL+.5*raw)
            np.testing.assert_allclose(action_target(s,raw,frame/50,reference),expected,atol=1e-7)
        self.assertEqual(float(sideflip_native_weight(150,300,10)),1.)
        self.assertEqual(float(sideflip_native_weight(0,300,10)),0.)

    def test_keyboard_commands(self):
        self.assertEqual(key_request('3'),{'switch':'sideflip69'})
        self.assertEqual(key_request('5'),{'switch':'upright52'})
        self.assertEqual(key_request('T'),{'trigger':True})
        self.assertEqual(key_request('\x1b'),{'stop':True})


if __name__=='__main__':unittest.main()
