import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skills.backflip.teacher_retention import load_reference, mapped_targets, RetentionUpdater, install

REFERENCE = Path(__file__).resolve().parents[1] / "artifacts/backflip_rebound_training_20261024/teacher_reference"


def read_reference():
    return load_reference(REFERENCE, hashlib.sha256((REFERENCE / "manifest.json").read_bytes()).hexdigest())


def test_archived_reference():
    data, meta = read_reference()
    assert data["observations"].shape == (408, 60)
    assert len(meta["sources"]) == 4
    assert data["policy_time"].max() == pytest.approx(2.02)


def test_manifest_tampering_rejected():
    with pytest.raises(ValueError, match="manifest hash"):
        load_reference(REFERENCE, "0" * 64)


@pytest.mark.parametrize("corruption", ["data_hash", "terminal_time", "shape", "seed", "coverage", "cutoff", "absorption"])
def test_invalid_data_rejected(tmp_path, corruption):
    data, meta = read_reference()
    if corruption == "terminal_time": data["policy_time"][-1] = 2.5
    if corruption == "shape": data["observations"] = data["observations"][:, :-1]
    if corruption == "seed": data["seed"][0] = -1
    if corruption == "cutoff": meta["sources"][0]["cutoff_s"] += .01
    if corruption == "absorption": meta["absorption_seconds"] = .1
    if corruption == "coverage": data["policy_time"][1] += .001
    np.savez(tmp_path / "samples.npz", **data)
    meta["samples_sha256"] = hashlib.sha256((tmp_path / "samples.npz").read_bytes()).hexdigest()
    if corruption == "data_hash": meta["samples_sha256"] = "0" * 64
    payload = json.dumps(meta).encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    with pytest.raises(ValueError):
        load_reference(tmp_path, hashlib.sha256(payload).hexdigest())


def test_entry_mapping_and_clip():
    raw = torch.full((3, 12), 20.)
    actual = mapped_targets(raw, torch.tensor([0., 1.1, 1.2]), .15, .5, 8.)
    assert torch.allclose(actual[0], torch.full((12,), .075))
    assert torch.allclose(actual[1:], torch.full((2, 12), 4.))
    raw.fill_(2.)
    actual = mapped_targets(raw, torch.tensor([0., 1.1, 1.2]), .15, .5, 8.)
    assert actual[1, 0].item() == pytest.approx((2 + .15 * np.tanh(2)) * .25, abs=1e-6)
    assert actual[2, 0].item() == pytest.approx(1.)


def test_actor_only_update_preserves_rng_gradients_optimizer_and_other_parameters():
    actor = torch.nn.Linear(60, 12)
    critic = torch.nn.Linear(60, 1)
    std = torch.nn.Parameter(torch.ones(12))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [std])
    for p in actor.parameters(): p.grad = torch.ones_like(p)
    optimizer.step()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_grad = [p.grad.clone() for p in actor.parameters()]
    saved_critic = copy.deepcopy(critic.state_dict())
    saved_std = std.detach().clone()
    data = dict(observations=np.ones((8, 60)), raw_actions=np.zeros((8, 12)), policy_time=np.full(8, 1.5))
    rng = torch.random.get_rng_state().clone()
    updater = RetentionUpdater(actor, data, weight=2., learning_rate=.001, batch_size=8)
    before = actor(torch.ones(8, 60)).square().mean().item()
    for _ in range(10): updater.step()
    assert actor(torch.ones(8, 60)).square().mean().item() < before
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(std, saved_std)
    for key, value in critic.state_dict().items(): assert torch.equal(value, saved_critic[key])
    for p, g in zip(actor.parameters(), saved_grad): assert torch.equal(p.grad, g)
    current = optimizer.state_dict()
    assert current["param_groups"] == saved_optimizer["param_groups"]
    for key, state in current["state"].items():
        for name, value in state.items(): assert torch.equal(value, saved_optimizer["state"][key][name])


def test_install_preserves_original_update_and_rejects_duplicate():
    actor = torch.nn.Linear(60, 12)
    data = dict(observations=np.zeros((2, 60)), raw_actions=np.zeros((2, 12)), policy_time=np.ones(2))
    updater = RetentionUpdater(actor, data, weight=1.)
    calls = []
    def original():
        calls.append(1)
        return {"ppo": 7.}
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor), update=original))
    install(runner, updater)
    losses = runner.alg.update()
    assert calls == [1] and losses["ppo"] == 7. and updater.steps == 1
    assert "teacher_retention_mse" in losses
    with pytest.raises(ValueError, match="already installed"): install(runner, updater)
