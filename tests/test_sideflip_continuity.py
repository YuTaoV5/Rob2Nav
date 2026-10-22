from types import SimpleNamespace
import pytest
import torch
from skills.sideflip.continuity import penalty, validate


def test_penalty_uses_mapped_setpoint_actual_dt_and_joint_order():
    delta=torch.tensor([[.1,.2],[.01,.01]])
    term=SimpleNamespace(target_step_delta=delta)
    env=SimpleNamespace(step_dt=.02,action_manager=SimpleNamespace(get_term=lambda _:term))
    before=delta.clone()
    value=penalty(env,['slow','fast'],{'fast':20.,'slow':5.},.8)
    torch.testing.assert_close(value,torch.tensor([.04,0.]))
    torch.testing.assert_close(delta,before)
    env.step_dt=.01
    torch.testing.assert_close(penalty(env,['slow','fast'],{'fast':20.,'slow':5.},.8),torch.tensor([1.44,0.]))


@pytest.mark.parametrize('weight,margin',[(float('nan'),.8),(-1,.8),(101,.8),(1,0),(1,1.1)])
def test_invalid_controls(weight,margin):
    with pytest.raises(ValueError):validate(weight,margin)
