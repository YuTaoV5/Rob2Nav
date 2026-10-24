import pytest
import torch
from skills.backflip.landing_shaping import shaping_terms, metadata
from studio.stability_options import defaults, validate


def test_only_absorption_and_no_inverted_reward():
    phase=torch.tensor([0,1,2,3,4,5,6,6])
    up,h=shaping_terms(phase,torch.tensor([1.]*7+[-1.]),torch.full((8,),.3),.3)
    assert up[:6].sum()==0 and h[:6].sum()==0
    assert up[6]==1 and up[7]<.000001
    assert h[6]==1


def test_invalid_weights():
    for value in (-1,float('nan'),101):
        with pytest.raises(ValueError):metadata(value,0)


def test_web_requires_v2():
    options=defaults('backflip_native_ppo')
    options['rebound_absorption_upright_weight']=5
    with pytest.raises(ValueError):validate('backflip_native_ppo',options)
    options.update(rebound_phase_version='landing_v2',rebound_cycles=3,fixstand_roundtrip=True,delayed_entry=True)
    validate('backflip_native_ppo',options)



def test_wrapper_scales_by_dt_and_preserves_other_phases():
    from types import SimpleNamespace as N
    from skills.backflip.landing_shaping import configure
    class Env:
        def _get_rewards(self):return torch.ones(3)
    cfg=N(rebound_phase_version='landing_v2')
    configure(cfg,5.,2.,Env)
    e=Env();e._rebound=N(phase=torch.tensor([6,3,4]));e.step_dt=.02;e.extras={}
    e.cfg=N(rewards=N(target_height=.3))
    e._robot=N(data=N(projected_gravity_b=torch.tensor([[0.,0.,-1.]]*3),root_link_pos_w=torch.tensor([[0.,0.,.3]]*3)))
    assert torch.allclose(e._get_rewards(),torch.tensor([1.14,1.,1.]))
    with pytest.raises(RuntimeError):configure(cfg,5.,2.,Env)


def test_export_detects_shaping_mismatch():
    from tests.test_landing_contract import inputs
    from skills.backflip.landing_contract import validate_landing_contract
    e,c=inputs(); e['rebound_landing_shaping']=metadata(5.,2.)
    with pytest.raises(ValueError):validate_landing_contract(e,c)
    c['rebound_landing_options'].update(absorption_upright_weight=5.,absorption_height_weight=2.)
    assert validate_landing_contract(e,c)['landing_shaping']==metadata(5.,2.)
    e['rebound_landing_shaping']['source_sha256']='wrong'
    with pytest.raises(ValueError):validate_landing_contract(e,c)
