"""Mechanism tests: complete cycles, shared differentiable reward and target mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from go2_supermimic.supermimic import state as st
import rob2nav_gait_v2 as gait
from rob2nav_gait_metrics import complete_cycles


def test_complete_cycle_not_window_occupancy():
    contact = np.tile(np.r_[np.ones(60), np.zeros(40)], 4).astype(bool)
    cycles = complete_cycles(contact, np.ones(len(contact), dtype=bool), .005)
    assert len(cycles) == 2
    assert all(abs(c["duty_factor"]-.6) < 1e-6 for c in cycles)
    invalid = np.ones(len(contact), dtype=bool);invalid[150] = False
    assert len(complete_cycles(contact, invalid, .005)) == 1


def test_collision_chatter_is_not_swing():
    contact = np.ones(400, dtype=bool);contact[[90, 190, 290, 390]] = False
    assert complete_cycles(contact, np.ones(400, dtype=bool), .005) == []


def test_reward_gradients_and_mapped_rate():
    n = 3
    s0 = torch.zeros(n, st.STATE_DIM);s0[:, st.Z] = .24;s0[:, 3] = -1.
    s0[:, st.FEET] = torch.tensor([.2, -.14, -.218, .2, .14, -.218, -.2, -.14, -.218, -.2, .14, -.218])
    s1 = s0.clone().requires_grad_()
    action = torch.zeros(n, 12, requires_grad=True)
    previous = torch.zeros(n, 12)
    scale = torch.ones(n, 12)*.5;limits = torch.ones(12)*20.
    soft = torch.stack([torch.full((n,12), -3.), torch.full((n,12), 3.)], -1)
    args = [s0, s1, torch.zeros(n,3), torch.zeros(n,3), torch.zeros(n,3), action, previous, soft, torch.zeros(n,12), scale, limits, .02, gait.GaitSettings()]
    value, terms = gait.reward(*args)
    assert terms["mapped_target_rate"].max() == 0
    (-value.mean()).backward()
    assert torch.isfinite(s1.grad).all() and s1.grad[:, st.Z].abs().sum() > 0
    jumped = torch.ones(n,12)
    args[5] = jumped
    _, terms = gait.reward(*args)
    assert torch.all(terms["mapped_target_rate"] > 0)


def test_peak_reduction_does_not_dilute_one_joint_outlier():
    from dataclasses import replace
    n = 1
    s = torch.zeros(n, st.STATE_DIM);s[:, st.Z] = .29;s[:, 3] = -1.
    action = torch.zeros(n, 12);action[:, 0] = 1.
    soft = torch.stack([torch.full((n,12), -3.), torch.full((n,12), 3.)], -1)
    args = [s, s, torch.zeros(n,3), torch.zeros(n,3), torch.zeros(n,3), action,
            torch.zeros(n,12), soft, torch.zeros(n,12), torch.ones(n,12)*.5, torch.ones(12)*20., .02]
    settings = gait.GaitSettings()
    _, mean = gait.reward(*args, settings)
    _, maximum = gait.reward(*args, replace(settings, mapped_target_rate_reduction="max"))
    assert torch.allclose(maximum["mapped_target_rate"], mean["mapped_target_rate"]*12)
