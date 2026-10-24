import unittest
import torch
from types import SimpleNamespace as NS
from tests.test_backflip_rebound_task import fake_class,config
from skills.backflip.rebound_landing_task import configure
from studio.stability_options import defaults,validate,arguments
from skills.backflip.launcher_patch import configure_fixstand_source


class LandingTaskTests(unittest.TestCase):
    def make(self,weight=5.):
        parent=fake_class()
        class Env(parent):
            def __init__(self,cfg):
                super().__init__(cfg)
                self._actions=torch.zeros(2,12)
                self._joint_velocity_limits=torch.full((1,12),10.)
            def _pre_physics_step(self,actions):
                self._actions.copy_(actions.clamp(-8,8))
        cfg=config();cfg.control=NS(action_scale=.5)
        configure(cfg,3,env_class=Env,target_rate_weight=weight)
        return Env(cfg)

    def test_penalty_uses_mapped_actions_and_preserves_commands(self):
        e=self.make();e._rebound.phase[:]=e._rebound.REBOUND
        action=torch.ones(2,12)
        e._pre_physics_step(action)
        self.assertTrue(torch.equal(e._actions,action))
        self.assertTrue(torch.allclose(e._rebound_target_ratio,torch.full((2,),2.5)))
        self.assertTrue(torch.allclose(e._get_rewards(),torch.full((2,),-5*(2.5-.8)**2*.02)))
        e._pre_physics_step(action)
        self.assertTrue(torch.equal(e._rebound_target_cost,torch.zeros(2)))

    def test_reset_clears_only_selected_previous_targets(self):
        e=self.make();e._pre_physics_step(torch.ones(2,12))
        e._reset_idx(torch.tensor([0]))
        self.assertEqual(e._rebound_previous_action[:,0].tolist(),[0,1])
        self.assertEqual(e.cfg.rebound_phase_version,'landing_v2')
        self.assertEqual(set(e.cfg.rebound_implementation_sha256),{'rebound_landing.py','rebound_landing_task.py'})

    def test_physics_tick_and_upright_input(self):
        e=self.make()
        e._pre_physics_step(torch.zeros(2,12))
        for _ in range(4):e._sim_step_counter+=1;e._apply_action()
        e._get_dones()
        self.assertAlmostEqual(float(e._rebound.elapsed[0]),.02,places=6)
        e._get_dones()
        self.assertAlmostEqual(float(e._rebound.elapsed[0]),.02,places=6)

    def test_web_version_parameters_and_launcher_match(self):
        o=defaults('backflip_native_ppo')
        o.update(fixstand_roundtrip=True,delayed_entry=True,rebound_cycles=3,
                 rebound_phase_version='landing_v2',rebound_target_rate_weight=5.)
        validate('backflip_native_ppo',o)
        a=arguments('backflip_native_ppo',o)
        self.assertEqual(a[a.index('--rebound-target-rate-weight')+1],'5.0')
        source=configure_fixstand_source('env_cfg = Go2BackflipEnvCfg()', '.',delayed_entry=True,
            rebound_cycles=3,rebound_phase_version='landing_v2',rebound_landing_options={'target_rate_weight':5.})
        self.assertIn('from skills.backflip.rebound_landing_task import configure',source)
        self.assertIn("'target_rate_weight': 5.0",source)
        o['rebound_phase_version']='legacy_v1'
        with self.assertRaises(ValueError):validate('backflip_native_ppo',o)
        o['rebound_phase_version']='landing_v2';o['rebound_max_seconds']=.1
        with self.assertRaises(ValueError):validate('backflip_native_ppo',o)
        self.assertNotIn('rebound_phase_version',defaults('frontflip_native_ppo'))


if __name__=='__main__':unittest.main()
