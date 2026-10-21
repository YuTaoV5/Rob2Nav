from types import SimpleNamespace
from unittest.mock import patch
import unittest
from skills.backflip.launcher_patch import configure_fixstand_source
from skills.backflip.control import validate_entry_contract


class BackflipLauncherTests(unittest.TestCase):
    def test_parameterized_base_config_still_applies_delayed_entry(self):
        source='def main():\n    env_cfg = Go2BackflipEnvCfg()\n    return env_cfg\n'
        result=configure_fixstand_source(source,'/pinned',True,True,True,.1)
        def delayed(cfg,scale):cfg.episode_length_s=5.;cfg.entry_delay_seconds=1.
        with patch('skills.fixstand.configure_backflip') as base,patch('skills.fixstand.configure_backflip_delayed_entry',side_effect=delayed) as entry:
            scope={'Go2BackflipEnvCfg':lambda:SimpleNamespace(episode_length_s=4.)}
            exec(compile(result,'<test-pinned-entry>','exec'),scope)
            cfg=scope['main']()
            base.assert_called_once_with(cfg,'/pinned',True)
            entry.assert_called_once_with(cfg,.1)
        validate_entry_contract(vars(cfg),{'action_parameterization':'backflip_entry_v3','entry_hold_scale':.1})

    def test_missing_injection_or_mislabeled_export_is_rejected(self):
        with self.assertRaises(RuntimeError):configure_fixstand_source('changed entry','/pinned')
        with self.assertRaises(ValueError):validate_entry_contract({'episode_length_s':4.},
            {'action_parameterization':'backflip_entry_v3'})
        with self.assertRaises(ValueError):validate_entry_contract({'episode_length_s':5.,'entry_delay_seconds':1.,'entry_hold_scale':.4},
            {'action_parameterization':'backflip_entry_v3','entry_hold_scale':.1})


if __name__=='__main__':unittest.main()
