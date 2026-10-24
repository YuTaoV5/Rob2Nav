"""Batched contact-driven cycle bookkeeping for a NEW rebound task.

No robot state writes or controller outputs. Not wired into the single-flip task.
Call once per physics step with current contacts (not history-window peaks).
This counter is not a complete motion/safety/endpoint validator.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ReboundConfig:
    cycles: int = 3
    entry_seconds: float = 1.
    max_rebound_seconds: float = .45
    takeoff_confirm_seconds: float = .02
    landing_confirm_seconds: float = .01
    min_airtime_seconds: float = .08
    min_rotation: float = 5.5
    max_rotation: float = 7.1
    min_axis_dominance: float = .75

    def __post_init__(self):
        if self.cycles not in (2, 3):
            raise ValueError('Rebound task supports two or three cycles')
        values = (self.entry_seconds, self.max_rebound_seconds,
                  self.takeoff_confirm_seconds, self.landing_confirm_seconds,
                  self.min_airtime_seconds, self.min_rotation, self.max_rotation,
                  self.min_axis_dominance)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('Finite positive thresholds required')
        if self.min_rotation >= self.max_rotation or self.min_axis_dominance > 1:
            raise ValueError('Invalid rotation thresholds')
        if self.max_rebound_seconds <= self.landing_confirm_seconds:
            raise ValueError('Rebound budget must include landing confirmation')


class ReboundPhases:
    ENTRY, PUSH, FLIGHT, REBOUND, RECOVERY, FAILED = range(6)

    def __init__(self, count, dt, config=None, device='cpu'):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Positive physics step required')
        self.cfg = config or ReboundConfig()
        self.dt = dt
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.completed = torch.zeros_like(self.phase)
        self.just_landed = torch.zeros(count, dtype=torch.bool, device=device)
        self.just_took_off = torch.zeros_like(self.just_landed)
        for name in ('elapsed', 'air', 'max_air', 'contact', 'angle', 'travel', 'rebound_time'):
            setattr(self, name, torch.zeros(count, device=device))
        self.last_rebound_seconds = torch.full((count,), float('nan'), device=device)

    def reset(self, ids):
        # Bookkeeping only; caller controls actual episode resets.
        self.phase[ids] = self.ENTRY
        self.completed[ids] = 0
        self.just_landed[ids] = False
        self.just_took_off[ids] = False
        for name in ('elapsed', 'air', 'max_air', 'contact', 'angle', 'travel', 'rebound_time'):
            getattr(self, name)[ids] = 0
        self.last_rebound_seconds[ids] = float('nan')

    def step(self, foot_contact, body_omega, body_contact):
        """Body-frame omega; negative pitch is backflip. Contact inputs are booleans."""
        if foot_contact.shape != self.phase.shape or body_contact.shape != self.phase.shape:
            raise ValueError('Contact batch shape mismatch')
        if body_omega.shape != (*self.phase.shape, 3):
            raise ValueError('Angular velocity batch shape mismatch')
        if foot_contact.dtype != torch.bool or body_contact.dtype != torch.bool:
            raise ValueError('Contact masks must be boolean')
        self.just_landed.zero_()
        self.just_took_off.zero_()
        self.phase[body_contact | ~torch.isfinite(body_omega).all(dim=-1)] = self.FAILED
        self.elapsed += self.dt
        self.phase[(self.phase == self.ENTRY) &
                   (self.elapsed + 1e-6 >= self.cfg.entry_seconds)] = self.PUSH
        moving = ((self.phase == self.PUSH) | (self.phase == self.FLIGHT) |
                  (self.phase == self.REBOUND))
        airborne = moving & ~foot_contact
        self.air = torch.where(airborne, self.air + self.dt, 0.)
        self.max_air = torch.maximum(self.max_air, self.air)
        # Count rotation in flight only; grounded turning cannot earn a flip.
        unconfirmed_contact = ((self.phase == self.PUSH) | (self.phase == self.REBOUND)) & foot_contact
        self.angle[unconfirmed_contact] = 0.
        self.travel[unconfirmed_contact] = 0.
        integrate = airborne
        self.angle += torch.where(integrate, -body_omega[:, 1] * self.dt, 0.)
        self.travel += torch.where(integrate, torch.linalg.vector_norm(body_omega, dim=-1) * self.dt, 0.)
        # Start the rebound budget at first landing contact, including confirmation.
        landing_sample = (self.phase == self.FLIGHT) & foot_contact
        self.contact = torch.where(landing_sample, self.contact + self.dt, 0.)
        is_rebound = self.phase == self.REBOUND
        self.rebound_time += torch.where(is_rebound | landing_sample, self.dt, 0.)
        self.phase[is_rebound & (self.rebound_time > self.cfg.max_rebound_seconds + 1e-6)] = self.FAILED
        takeoff = (((self.phase == self.PUSH) | (self.phase == self.REBOUND)) &
                   (self.air + 1e-6 >= self.cfg.takeoff_confirm_seconds))
        self.last_rebound_seconds[takeoff & is_rebound] = self.rebound_time[takeoff & is_rebound]
        self.phase[takeoff] = self.FLIGHT
        self.just_took_off[:] = takeoff
        landed = ((self.phase == self.FLIGHT) &
                  (self.contact + 1e-6 >= self.cfg.landing_confirm_seconds))
        valid = ((self.angle >= self.cfg.min_rotation) & (self.angle <= self.cfg.max_rotation) &
                 (self.angle / self.travel.clamp_min(1e-8) >= self.cfg.min_axis_dominance) &
                 (self.max_air + 1e-6 >= self.cfg.min_airtime_seconds))
        self.phase[landed & ~valid] = self.FAILED
        accepted = landed & valid
        self.completed += accepted.long()
        self.just_landed[:] = accepted
        self.phase[accepted] = self.REBOUND
        self.phase[accepted & (self.completed == self.cfg.cycles)] = self.RECOVERY
        confirmed_contact_duration = self.contact.clone()
        for name in ('angle', 'travel', 'air', 'max_air', 'contact'):
            getattr(self, name)[accepted] = 0.
        # Retain just the current contact-confirmation interval as next rebound budget.
        self.rebound_time[accepted] = confirmed_contact_duration[accepted]
        return self.phase
