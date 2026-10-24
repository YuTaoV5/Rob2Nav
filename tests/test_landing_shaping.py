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

