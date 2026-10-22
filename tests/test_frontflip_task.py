import unittest
from types import SimpleNamespace
import torch
from skills.frontflip.task import adapt_method
from skills.backflip.launcher_patch import configure_fixstand_source


class FrontflipTaskTests(unittest.TestCase):
    def test_positive_pitch_is_rewarded_and_counted(self):
        data=SimpleNamespace(root_ang_vel_b=torch.tensor([[0.,2.,0.],[0.,-2.,0.]]))
        obj=SimpleNamespace(_robot=SimpleNamespace(data=data))
        for name in ('_update_flip_state','_reward_ang_vel_y'):
            text=f'def {name}(self):\n    return -self._robot.data.root_ang_vel_b[:, 1]\n'
            scope={};exec(adapt_method(name,text),scope)
            torch.testing.assert_close(scope[name](obj),torch.tensor([2.,-2.]))

    def test_orientation_target_is_positive_pitch(self):
        text='def _reward_orientation_control(self):\n    phase=.125\n    desired_angle = -4.0 * torch.pi * phase\n    return desired_angle\n'
        scope={'torch':torch};exec(adapt_method('_reward_orientation_control',text),scope)
        self.assertAlmostEqual(scope['_reward_orientation_control'](None),torch.pi/2)

    def test_unknown_or_changed_source_fails_closed(self):
        for name in ('_reward_ang_vel_y','unknown'):
            with self.assertRaises(ValueError):adapt_method(name,'def x(): return 0')

    def test_front_configuration_follows_delayed_entry(self):
        text='    env_cfg = Go2BackflipEnvCfg()\n'
        result=configure_fixstand_source(text,'/pinned',True,True,True,.1,'front')
        self.assertLess(result.index('configure_backflip_delayed_entry(env_cfg'),result.index('configure_frontflip(env_cfg'))
        self.assertNotIn('configure_frontflip',configure_fixstand_source(text,'/pinned'))


if __name__=='__main__':unittest.main()
