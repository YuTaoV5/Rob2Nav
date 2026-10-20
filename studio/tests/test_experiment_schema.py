import hashlib
import json
import pytest
from studio.config import RunConfig
from rob2nav.experiment_schema import reward_schema
from moe_perceptive.studio_overrides import load


def test_mode_defaults_and_complete_catalog():
    nav=reward_schema(RunConfig())
    loco=reward_schema(RunConfig(navigation=False,mode='student',arrival_aware=False,course='original'))
    assert nav['forward_progress']['effective_weight']==5
    assert loco['forward_progress']['effective_weight']==1
    assert loco['lane_center']['effective_weight']==-.5
    assert not nav['track_lin_vel_xy_exp']['available']
    assert loco['track_lin_vel_xy_exp']['effective_weight']==1.5
    assert 'dof_torques_l2' in nav and 'undesired_contacts' in nav


def test_optional_targets_and_legacy_digest(tmp_path):
    old=RunConfig().overrides()
    assert 'motion' not in old
    path=tmp_path/'overrides.json';path.write_text(json.dumps(old))
    parsed,digest=load(path)
    assert digest==hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert parsed==old
    cfg=RunConfig(body_height_target_m=.32,swing_clearance_target_m=.12)
    assert reward_schema(cfg)['body_height_error']['available']
    path.write_text(json.dumps(cfg.overrides()))
    assert load(path)[0]['motion']['swing_clearance_target_m']==.12


def test_invalid_targets_and_unavailable_rewards():
    for kwargs in ({'body_height_target_m':.8},{'swing_clearance_target_m':.01},
                   {'rewards':{'body_height_error':-1}},
                   {'task':'simulation','body_height_target_m':.32}):
        with pytest.raises(ValueError):RunConfig(**kwargs)


def test_stance_motion_is_opt_in():
    assert 'stance_foot_motion' not in RunConfig().overrides()['rewards']
    cfg=RunConfig(rewards={'stance_foot_motion':-.5})
    assert reward_schema(cfg)['stance_foot_motion']['effective_weight']==-.5
