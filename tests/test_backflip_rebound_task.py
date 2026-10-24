from types import SimpleNamespace as NS
import unittest
import torch
from skills.backflip.rebound_task import configure
from skills.backflip.control import validate_entry_contract
from studio.stability_options import defaults, validate, arguments


def config():
    return NS(episode_length_s=5., entry_delay_seconds=1., sim=NS(dt=.005),
              contact_sensor=NS(update_period=.02), rewards=NS(takeoff_start=.5,
              recovery_contact_force=5., unsafe_body_contact_force=5.,
              recovery_upright_cos=.95,recovery_min_height=.25,recovery_max_pitch_rate=.5,
              success_max_joint_speed=1.,recovery_pose_error=.1,success_hold_time_s=1.,
              target_height=.32,scales={'default_pose':-1.}))


def fake_class():
    class Env:
        def __init__(self, cfg):
            self.cfg=cfg; self.device='cpu'; self.num_envs=2; self.step_dt=.02
            self._sim_step_counter=0
            self._feet_contact_ids=[0]; self._undesired_contact_sensor_ids=[1]
            force=torch.zeros(2,2,3);force[:,0,2]=10.
            self._contact_sensor=NS(data=NS(net_forces_w=force))
            from skills.fixstand import POSE
            q=torch.tensor([list(POSE.values())]*2)
            self._robot=NS(device='cpu',joint_names=list(POSE),data=NS(root_lin_vel_w=torch.zeros(2,3),root_ang_vel_b=torch.zeros(2,3),
                projected_gravity_b=torch.tensor([[0.,0.,-1.]]*2),root_link_pos_w=torch.tensor([[0.,0.,.32]]*2),
                joint_pos=q,default_joint_pos=q.clone(),joint_vel=torch.zeros(2,12)))
            for name in ('_unsafe_episode','_just_completed','_flip_completed','_landing_detected',
                         '_was_airborne','_just_succeeded','_flip_success'):
                setattr(self,name,torch.zeros(2,dtype=torch.bool))
            for name in ('_rotation_progress_step','_flip_angle','_max_flip_angle'):
                setattr(self,name,torch.zeros(2))
            self._reward_functions={'default_pose': self._reward_default_pose}
            self._episode_sums={'default_pose':torch.zeros(2)}; self.extras={}
        def _reset_idx(self,ids):pass
        def _pre_physics_step(self,actions):pass
        def _apply_action(self):pass
        def _update_flip_state(self):pass
        def _get_dones(self):
            self._update_flip_state()
            return torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool)
        def _reward_default_pose(self):return torch.zeros(2)
    return Env


class ReboundTaskTests(unittest.TestCase):
    def make(self):
        cls=fake_class();cfg=config();configure(cfg,3,.45,cls)
        return cls(cfg)

    def test_substeps_once_with_final_sample_and_subset_reset(self):
        e=self.make()
        for _ in range(2):
            e._pre_physics_step(torch.zeros(2,12))
            for _ in range(4):
                e._sim_step_counter+=1;e._apply_action()
            e._get_dones()
        self.assertAlmostEqual(e._rebound.elapsed[0].item(),.04,places=6)
        e._get_dones()  # same physical state does not advance time again
        self.assertAlmostEqual(e._rebound.elapsed[0].item(),.04,places=6)
        e._reset_idx(torch.tensor([0]))
        self.assertEqual(e._rebound.elapsed[0].item(),0.)
        self.assertAlmostEqual(e._rebound.elapsed[1].item(),.04,places=6)
        self.assertEqual(e._rebound_tick,8)
        e._sim_step_counter+=2
        with self.assertRaises(RuntimeError):e._get_dones()

    def test_only_final_recovery_gets_stand_rewards(self):
        e=self.make();r=e._rebound
        r.phase[:]=r.REBOUND
        self.assertFalse(e._recovery_active().any())
        self.assertTrue(torch.equal(e._get_rewards(),torch.zeros(2)))
        r.phase[0]=r.RECOVERY
        self.assertEqual(e._recovery_active().tolist(),[True,False])
        self.assertGreater(e._get_rewards()[0].item(),0.)
        self.assertEqual(e._get_rewards()[1].item(),0.)

    def test_config_and_legacy_export_guard(self):
        e=self.make()
        self.assertEqual(e.cfg.episode_length_s,9.)
        self.assertEqual(e.cfg.contact_sensor.update_period,.005)
        self.assertEqual(len(e._phase_features()),6)
        with self.assertRaises(ValueError):
            validate_entry_contract({}, {'action_parameterization':'backflip_rebound_v1'})

    def test_web_parameters_and_default_preservation(self):
        o=defaults('backflip_native_ppo')
        self.assertFalse(any('rebound' in x for x in arguments('backflip_native_ppo',o)))
        o.update(rebound_cycles=3,fixstand_roundtrip=True,delayed_entry=True)
        validate('backflip_native_ppo',o)
        a=arguments('backflip_native_ppo',o)
        self.assertEqual(a[a.index('--rebound-cycles')+1],'3')
        self.assertEqual(a[a.index('--rebound-max-seconds')+1],'0.45')
        o['delayed_entry']=False
        with self.assertRaises(ValueError):validate('backflip_native_ppo',o)
        self.assertNotIn('rebound_cycles',defaults('frontflip_native_ppo'))


if __name__=='__main__':unittest.main()
