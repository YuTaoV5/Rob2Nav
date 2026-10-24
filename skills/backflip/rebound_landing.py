"""Candidate v2 cycle counter, NOT installed into training or deployment.

Tracks full-cycle rotation separately from airborne evidence. Inputs are measured
physics state; never writes robot state. Existing v1 contracts remain unchanged.
"""
from dataclasses import dataclass
import math
import torch
from .rebound import ReboundConfig


@dataclass(frozen=True)
class LandingConfig(ReboundConfig):
    landing_max_seconds: float = .25
    min_air_rotation: float = math.pi
    inverted_cos: float = -.5
    landing_upright_cos: float = .9

    def __post_init__(self):
        super().__post_init__()
        if not (math.isfinite(self.landing_max_seconds) and
                self.landing_confirm_seconds < self.landing_max_seconds <= self.max_rebound_seconds):
            raise ValueError('Landing absorption must fit inside rebound budget')
        if not (math.isfinite(self.min_air_rotation) and 0 < self.min_air_rotation <= self.min_rotation):
            raise ValueError('Invalid minimum aerial rotation')
        if not (-1 <= self.inverted_cos < 0 < self.landing_upright_cos <= 1):
            raise ValueError('Invalid orientation thresholds')


class LandingPhases:
    ENTRY, PUSH, FLIGHT, REBOUND, RECOVERY, FAILED, LANDING = range(7)

    def __init__(self, count, dt, config=None, device='cpu'):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid physics timestep')
        self.cfg = config or LandingConfig()
        self.dt = dt
        self.phase = torch.zeros(count, dtype=torch.long, device=device)
        self.completed = torch.zeros_like(self.phase)
        for name in ('elapsed', 'angle', 'travel', 'air', 'air_angle', 'air_travel',
                     'max_air', 'contact', 'rebound_time'):
            setattr(self, name, torch.zeros(count, device=device))
        for name in ('inverted', 'just_landed', 'just_took_off'):
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        self.last_rebound_seconds = torch.full((count,), float('nan'), device=device)

    def reset(self, ids):
        self.phase[ids] = self.ENTRY
        self.completed[ids] = 0
        for name in ('elapsed', 'angle', 'travel', 'air', 'air_angle', 'air_travel',
                     'max_air', 'contact', 'rebound_time'):
            getattr(self, name)[ids] = 0
        for name in ('inverted', 'just_landed', 'just_took_off'):
            getattr(self, name)[ids] = False
        self.last_rebound_seconds[ids] = float('nan')

    def step(self, foot_contact, body_omega, body_contact, upright):
        if any(x.shape != self.phase.shape for x in (foot_contact, body_contact, upright)) or body_omega.shape != (*self.phase.shape, 3):
            raise ValueError('Batch shape mismatch')
        if foot_contact.dtype != torch.bool or body_contact.dtype != torch.bool:
            raise ValueError('Boolean contact masks required')
        self.just_landed.zero_(); self.just_took_off.zero_()
        bad = body_contact | ~torch.isfinite(body_omega).all(-1) | ~torch.isfinite(upright) | (upright.abs() > 1.00001)
        self.phase[bad] = self.FAILED
        self.elapsed += self.dt
        self.phase[(self.phase == self.ENTRY) & (self.elapsed + 1e-6 >= self.cfg.entry_seconds)] = self.PUSH
        active = (self.phase == self.PUSH) | (self.phase == self.FLIGHT) | (self.phase == self.LANDING) | (self.phase == self.REBOUND)
        self.angle += torch.where(active, -body_omega[:, 1]*self.dt, 0.)
        self.travel += torch.where(active, torch.linalg.vector_norm(body_omega, dim=-1)*self.dt, 0.)
        unconfirmed = (self.phase == self.PUSH) | (self.phase == self.REBOUND)
        # Discard brief pre-takeoff contact chatter from aerial evidence only.
        clear = unconfirmed & foot_contact
        for name in ('air', 'air_angle', 'air_travel', 'max_air'):
            getattr(self, name)[clear] = 0
        self.inverted[clear] = False
        aerial = (unconfirmed | (self.phase == self.FLIGHT)) & ~foot_contact
        self.air = torch.where(aerial, self.air+self.dt, 0.)
        self.max_air = torch.maximum(self.max_air, self.air)
        self.air_angle += torch.where(aerial, -body_omega[:, 1]*self.dt, 0.)
        self.air_travel += torch.where(aerial, torch.linalg.vector_norm(body_omega, dim=-1)*self.dt, 0.)
        self.inverted |= aerial & (upright <= self.cfg.inverted_cos)
        previous_rebound = self.phase == self.REBOUND
        first_contact = (self.phase == self.FLIGHT) & foot_contact
        self.phase[first_contact] = self.LANDING
        self.rebound_time[first_contact] = 0.
        landing = self.phase == self.LANDING
        # Includes first contact, all absorption time and subsequent launch confirmation.
        self.rebound_time += torch.where(landing | previous_rebound, self.dt, 0.)
        self.contact = torch.where(landing & foot_contact, self.contact+self.dt, 0.)
        expired = ((landing & (self.rebound_time > self.cfg.landing_max_seconds+1e-6)) |
                   (previous_rebound & (self.rebound_time > self.cfg.max_rebound_seconds+1e-6)))
        self.phase[expired | (active & (self.angle > self.cfg.max_rotation))] = self.FAILED
        takeoff = ((self.phase == self.PUSH) | (self.phase == self.REBOUND)) & (self.air+1e-6 >= self.cfg.takeoff_confirm_seconds)
        self.last_rebound_seconds[takeoff & previous_rebound] = self.rebound_time[takeoff & previous_rebound]
        self.phase[takeoff] = self.FLIGHT
        self.just_took_off[:] = takeoff
        valid = ((self.angle >= self.cfg.min_rotation) & (self.angle <= self.cfg.max_rotation) &
                 (self.angle/self.travel.clamp_min(1e-8) >= self.cfg.min_axis_dominance) &
                 (self.air_angle >= self.cfg.min_air_rotation) &
                 (self.air_angle/self.air_travel.clamp_min(1e-8) >= self.cfg.min_axis_dominance) &
                 (self.max_air+1e-6 >= self.cfg.min_airtime_seconds) & self.inverted &
                 (upright >= self.cfg.landing_upright_cos) & foot_contact &
                 (self.contact+1e-6 >= self.cfg.landing_confirm_seconds))
        accepted = (self.phase == self.LANDING) & valid
        self.completed += accepted.long()
        self.just_landed[:] = accepted
        self.phase[accepted] = self.REBOUND
        self.phase[accepted & (self.completed == self.cfg.cycles)] = self.RECOVERY
        for name in ('angle', 'travel', 'air', 'air_angle', 'air_travel', 'max_air', 'contact'):
            getattr(self, name)[accepted] = 0.
        self.inverted[accepted] = False
        # Never reset rebound_time at acceptance: absorption consumes the same budget.
        return self.phase
