import unittest
import importlib.util
from copy import deepcopy
from dataclasses import asdict
from tests.test_rebound_contract import inputs as legacy_inputs
from skills.backflip.rebound_landing import LandingConfig
from skills.backflip.landing_contract import validate_landing_contract,implementation_hashes
from skills.backflip.rebound_contract import validate_rebound_contract
from skills.backflip.control import validate_entry_contract


def inputs():
    e,c=legacy_inputs()
    e.update(rebound=asdict(LandingConfig()),rebound_phase_version='landing_v2',
        rebound_target_rate_weight=5.,rebound_target_rate_margin=.8,
        rebound_implementation_sha256=implementation_hashes())
    c.update(action_parameterization='backflip_rebound_landing_v2',phase_contract='landing_cycle_v2',
        rebound_landing_options=dict(landing_max_seconds=.25,min_air_rotation=3.141592653589793,
            landing_upright_cos=.9,target_rate_weight=5.,target_rate_margin=.8))
    return e,c


class LandingContractTests(unittest.TestCase):
    def test_actual_configuration_complete(self):
        e,c=inputs();r=validate_landing_contract(e,c)
        self.assertEqual(r['target_rate_objective']['weight'],5.)
        self.assertFalse(r['capability_verified'])
        for key in e:
            broken=deepcopy(e);del broken[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_landing_contract(broken,c)

    def test_changed_reward_or_hash_cannot_be_hidden(self):
        e,c=inputs()
        for key,value in [('rebound_target_rate_weight',0.),('rebound_target_rate_margin',1.),
                          ('rebound_phase_version','legacy_v1'),('rebound_implementation_sha256',{})]:
            x=deepcopy(e);x[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_landing_contract(x,c)
        x=deepcopy(c);x['rebound_landing_options']['landing_max_seconds']=.2
        with self.assertRaises(ValueError):validate_landing_contract(e,x)

    def test_wrong_exporter_cannot_accept_v2_or_legacy(self):
        e,c=inputs()
        with self.assertRaises(ValueError):validate_entry_contract(e,c)
        with self.assertRaises(ValueError):validate_rebound_contract(e,c)
        with self.assertRaises(ValueError):validate_landing_contract(*legacy_inputs())

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),
                         'Run this integration test in the server export environment')
    def test_real_export_path_with_synthetic_checkpoint(self):
        import json,tempfile
        from pathlib import Path
        import torch,yaml
        from skills.backflip.export_cpu import export
        from skills.backflip.control import JOINT_NAMES
        e,c=inputs()
        e['rewards']['scales']={}
        e['control'].update(joint_names=list(JOINT_NAMES),action_scale=.5,stiffness=40.,damping=1.,
            motor_velocity_x1=13.5,motor_torque_y1=20.2,motor_torque_y2=23.4,
            joint_velocity_limits=[30.1,30.1,15.7]*4,target_velocity_limits=[13.5]*12)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'params').mkdir()
            (root/'params/agent.yaml').write_text(yaml.safe_dump({'policy':{'activation':'elu'}}))
            (root/'params/env.yaml').write_text(yaml.safe_dump(e))
            (root/'fixstand.json').write_text(json.dumps(c))
            actor=torch.nn.Sequential(torch.nn.Linear(60,16),torch.nn.ELU(),torch.nn.Linear(16,12))
            checkpoint=root/'synthetic_test_only.pt'
            torch.save({'model_state_dict':{'actor.'+k:v for k,v in actor.state_dict().items()}},checkpoint)
            export(checkpoint,root/'exported',landing=True)
            meta=json.loads((root/'exported/export.json').read_text())
            self.assertLess(meta['onnx_max_error'],1e-5)
            self.assertEqual(meta['rebound_contract']['version'],'backflip_rebound_landing_v2')
            self.assertEqual(meta['rebound_contract']['target_rate_objective']['weight'],5.)
            self.assertFalse(meta['capability_verified'])
            with self.assertRaises(ValueError):export(checkpoint,root/'wrong_single')
            with self.assertRaises(ValueError):export(checkpoint,root/'wrong_v1',rebound=True)
            self.assertFalse((root/'wrong_single').exists())
            self.assertFalse((root/'wrong_v1').exists())

    def test_missing_orientation_and_timing_rejected(self):
        e,c=inputs()
        for key in ('inverted_cos','landing_upright_cos','landing_max_seconds'):
            x=deepcopy(e);del x['rebound'][key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_landing_contract(x,c)
        e['contact_sensor']['update_period']=.02
        with self.assertRaises(ValueError):validate_landing_contract(e,c)


if __name__=='__main__':unittest.main()
