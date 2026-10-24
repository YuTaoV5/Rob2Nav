from copy import deepcopy
from dataclasses import asdict
import unittest
from skills.backflip.rebound import ReboundConfig
from skills.backflip.rebound_contract import validate_rebound_contract, implementation_hashes


def inputs():
    env=dict(rebound=asdict(ReboundConfig()), episode_length_s=9., entry_delay_seconds=1.,
             entry_hold_scale=.15, sim={'dt':.005}, decimation=4,
             contact_sensor={'update_period':.005}, control={'action_clip':8.},
             rewards={'takeoff_start':.5,'recovery_contact_force':5.,'unsafe_body_contact_force':5.},
             rebound_implementation_sha256=implementation_hashes())
    contract=dict(action_parameterization='backflip_rebound_v1', phase_contract='contact_cycle_v1',
                  actor_observations=60, rebound_cycles=3, rebound_max_seconds=.45,
                  episode_seconds=9., entry_seconds=1., entry_hold_scale=.15, release_seconds=[1.,1.2])
    return env,contract


class ContractTests(unittest.TestCase):
    def test_complete_actual_config_is_required(self):
        e,c=inputs()
        result=validate_rebound_contract(e,c)
        self.assertEqual(result['settings']['cycles'],3)
        self.assertFalse(result['capability_verified'])
        for key in e:
            broken=deepcopy(e);del broken[key]
            with self.subTest(key=key),self.assertRaises(ValueError):validate_rebound_contract(broken,c)

    def test_single_flip_cannot_be_relabelled(self):
        e,c=inputs()
        for key,value in [('episode_length_s',5.),('entry_delay_seconds',0.),('entry_hold_scale',.1),('decimation',10),('rebound_implementation_sha256',{})]:
            broken=deepcopy(e);broken[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_rebound_contract(broken,c)
        e['contact_sensor']['update_period']=.02
        with self.assertRaises(ValueError):validate_rebound_contract(e,c)

    def test_contract_mismatch_and_missing_thresholds_rejected(self):
        e,c=inputs()
        for key,value in [('rebound_cycles',2),('rebound_max_seconds',.6),('actor_observations',69),('phase_contract','single'),('release_seconds',[0.,.2]),('flip_direction','front')]:
            broken=deepcopy(c);broken[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_rebound_contract(e,broken)
        del e['rebound']['min_axis_dominance']
        with self.assertRaises(ValueError):validate_rebound_contract(e,c)


if __name__=='__main__':unittest.main()
