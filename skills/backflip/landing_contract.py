"""Strict actual-training-config validation for contact-cycle actors."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from .rebound_landing import LandingConfig as ReboundConfig

VERSION = 'backflip_rebound_landing_v2'


def implementation_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for name in ('rebound_landing.py', 'rebound_landing_task.py')}


def validate_landing_contract(env, contract):
    if not isinstance(contract, dict) or contract.get('action_parameterization') != VERSION:
        raise ValueError('An explicit rebound training contract is required')
    if contract.get('phase_contract') != 'landing_cycle_v2' or contract.get('actor_observations') != 60:
        raise ValueError('Unsupported rebound observation semantics')
    if contract.get('flip_direction', 'back') != 'back':
        raise ValueError('Rebound contract is for backflips only')
    raw = env.get('rebound')
    if not isinstance(raw, dict):
        raise ValueError('Actual env.yaml lacks rebound settings')
    try:
        settings = ReboundConfig(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid actual rebound settings') from exc
    if asdict(settings) != raw or type(raw['cycles']) is not int:
        raise ValueError('Actual YAML must include every rebound threshold')
    if contract.get('rebound_cycles') != settings.cycles or contract.get('rebound_max_seconds') != settings.max_rebound_seconds:
        raise ValueError('Exported rebound settings disagree with actual training')
    if env.get('rebound_phase_version') != 'landing_v2':
        raise ValueError('Actual training did not select landing v2')
    weight=env.get('rebound_target_rate_weight')
    margin=env.get('rebound_target_rate_margin')
    if (type(weight) not in (int,float) or not math.isfinite(weight) or not 0<=weight<=100 or
        type(margin) not in (int,float) or not math.isfinite(margin) or not .1<=margin<=1):
        raise ValueError('Actual target-rate objective is missing or invalid')
    options=dict(landing_max_seconds=settings.landing_max_seconds,
        min_air_rotation=settings.min_air_rotation,landing_upright_cos=settings.landing_upright_cos,
        target_rate_weight=weight,target_rate_margin=margin)
    if contract.get('rebound_landing_options') != options:
        raise ValueError('Landing/objective metadata disagrees with actual training')
    duration = 3. + 2. * settings.cycles
    if env.get('episode_length_s') != duration or contract.get('episode_seconds') != duration:
        raise ValueError('Rebound episode duration mismatch')
    if env.get('entry_delay_seconds') != 1. or contract.get('entry_seconds') != 1. or settings.entry_seconds != 1.:
        raise ValueError('Rebound entry must occur once for one second')
    hold = env.get('entry_hold_scale')
    if type(hold) not in (int, float) or not math.isfinite(hold) or not 0 <= hold <= .4 or contract.get('entry_hold_scale') != hold:
        raise ValueError('Entry hold scale mismatch')
    if contract.get('release_seconds') != [1., 1.2]:
        raise ValueError('Entry release contract mismatch')
    dt = env.get('sim', {}).get('dt')
    decimation = env.get('decimation')
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0 or type(decimation) is not int or decimation < 1:
        raise ValueError('Missing actual physics/control timing')
    if abs(dt * decimation - .02) > 1e-9:
        raise ValueError('Rebound actor requires 50Hz control')
    if env.get('contact_sensor', {}).get('update_period') != dt:
        raise ValueError('Rebound contacts must be sampled every physics step')
    if env.get('control', {}).get('action_clip') != 8.:
        raise ValueError('Rebound parent action clip must remain 8')
    rewards=env.get('rewards',{})
    timing={key:rewards.get(key) for key in ('takeoff_start','recovery_contact_force','unsafe_body_contact_force')}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in timing.values()):
        raise ValueError('Actual contact thresholds and cycle launch phase required')
    hashes = implementation_hashes()
    if env.get('rebound_implementation_sha256') != hashes:
        raise ValueError('Rebound implementation differs from recorded training code')
    return dict(version=VERSION, settings=raw, episode_seconds=duration,
                entry_hold_scale=hold, phase_contract='landing_cycle_v2',
                training_physics_dt=dt, policy_dt=.02, execution_thresholds=timing, implementation_sha256=hashes,
                target_rate_objective=dict(weight=weight,margin=margin,space="mapped_clipped_sdk_actions"),
                capability_verified=False)
