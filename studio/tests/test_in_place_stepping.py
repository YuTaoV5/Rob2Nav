import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys


def load():
    path = Path(__file__).parents[2]/'skills/upright_walk/in_place.py'
    spec = importlib.util.spec_from_file_location('in_place_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_zero_commands_preserve_alternating_reward(monkeypatch):
    managers = NS(RewardTermCfg=lambda **kw:NS(**kw))
    monkeypatch.setitem(sys.modules, 'isaaclab', NS())
    monkeypatch.setitem(sys.modules, 'isaaclab.managers', managers)
    reward = object()
    cfg = NS(commands=NS(base_velocity=NS(ranges=NS(), rel_standing_envs=0.)),
             rewards=NS(rear_alternation=reward))
    load().configure(cfg)
    ranges = cfg.commands.base_velocity.ranges
    assert ranges.lin_vel_x == ranges.lin_vel_y == ranges.ang_vel_z == (0., 0.)
    assert cfg.rewards.rear_alternation is reward
    assert cfg.rewards.in_place_drift.weight < 0
    assert not hasattr(cfg.rewards, 'stationary_balance')


def test_drift_penalty_never_changes_physical_state(monkeypatch):
    import torch
    module = load()
    times = torch.tensor([1.9, 1.9])
    monkeypatch.setitem(sys.modules, 'skills.fixstand', NS(phase_time=lambda env:times))
    pos = torch.tensor([[0., 0., .48], [1., 0., .48]])
    velocity = torch.zeros((2, 3))
    env = NS(scene={'robot':NS(data=NS(root_pos_w=pos, root_lin_vel_w=velocity))})
    assert torch.equal(module.drift_cost(env), torch.zeros(2))
    times[:] = 3.
    pos[0, 0] += .1
    before = pos.clone()
    cost = module.drift_cost(env)
    assert cost[0] > 0 and cost[1] == 0
    assert torch.equal(pos, before) and torch.equal(velocity, torch.zeros((2, 3)))
    # A reset episode starts with a fresh anchor before the next hold window.
    times[0] = .1
    assert module.drift_cost(env)[0] == 0
    times[0] = 3.
    assert module.drift_cost(env)[0] == 0
