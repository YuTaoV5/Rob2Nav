from dataclasses import replace
import numpy as np
import pytest
import torch
from skills.fixstand import terminal_hip_weight
from rob2nav.deployment.types import Skill,SDK_JOINTS
from rob2nav.deployment.policies import action_target


def contract():
    return Skill('left',SDK_JOINTS,(0.,)*12,(40.,)*12,(1.,)*12,'mimic69',50.,.5,
        action_parameterization='reference_residual_v4',reference='reference.npz',
        residual_limit_rad=.2,bridge_residual_limit_rad=.05,boundary_blend_frames=10)


def test_zero_taper_already_supported_by_training_is_accepted_by_unified_contract():
    s=contract().checked()
    assert s.terminal_hip_taper_frames==0
    ref=dict(frame=297,count=298,joint_pos=[0.]*12)
    raw=np.ones(12)*.7
    learned=s.from_sdk(action_target(s,raw,reference=ref))
    tapered=s.from_sdk(action_target(replace(s,terminal_hip_taper_frames=50).checked(),raw,reference=ref))
    hip=np.array([n.endswith('_hip_joint') for n in s.joints])
    assert np.all(learned[hip]>.12) # Bounded compensation remains controllable.
    assert np.all(tapered[hip]==0.)
    np.testing.assert_allclose(learned[~hip],tapered[~hip])
    for invalid in [True,1,-1,51]:
        with pytest.raises(ValueError):replace(s,terminal_hip_taper_frames=invalid).checked()
    with pytest.raises(ValueError):replace(s,action_parameterization='reference_residual_v3',boundary_blend_frames=0).checked()


def test_terminal_zero_taper_eliminates_hip_actor_influence_not_physical_error():
    raw=torch.tensor([.7],requires_grad=True)
    target=.2*torch.tanh(raw)*terminal_hip_weight(297,298,50)
    target.sum().backward()
    assert raw.grad.item()==0.
    # Disabling the existing optional taper keeps a bounded learned offset.
    raw2=torch.tensor([.7],requires_grad=True)
    (.2*torch.tanh(raw2)).sum().backward()
    assert raw2.grad.item()>0.
