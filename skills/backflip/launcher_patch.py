"""Reviewed environment injection for the pinned upstream training entry."""


def configure_fixstand_source(text,source,urdf_speed_limits=False,fast_safety_curriculum=False,
                              delayed_entry=False,entry_hold_scale=.4,flip_direction='back',peak_contact_forces=False,reward_scales=None,rebound_cycles=1,rebound_max_seconds=.45):
    if flip_direction not in ('back','front'):raise ValueError('Unknown flip direction')
    needle='env_cfg = Go2BackflipEnvCfg()'
    if text.count(needle)!=1:raise RuntimeError('Pinned environment config entry changed')
    block=[needle,'from skills.fixstand import configure_backflip',
        f'configure_backflip(env_cfg,{str(source)!r} if {urdf_speed_limits!r} else None,{fast_safety_curriculum!r})']
    if peak_contact_forces:
        block.extend(['env_cfg.rewards.use_peak_contact_forces=True',
                      'env_cfg.contact_sensor.update_period=env_cfg.sim.dt',
                      'env_cfg.contact_sensor.history_length=max(env_cfg.contact_sensor.history_length,env_cfg.decimation)'])
    if delayed_entry:
        block.extend(['from skills.fixstand import configure_backflip_delayed_entry',
            f'configure_backflip_delayed_entry(env_cfg,{entry_hold_scale!r})',
            'if env_cfg.episode_length_s!=5. or getattr(env_cfg,"entry_delay_seconds",None)!=1.:',
            '    raise RuntimeError("Delayed-entry training configuration was not applied")'])
    if flip_direction=='front':
        block.extend(['from skills.frontflip.task import configure as configure_frontflip',
                      f'configure_frontflip(env_cfg,{str(source)!r})'])
    if rebound_cycles != 1:
        if flip_direction != 'back' or not delayed_entry:
            raise ValueError('Rebound requires backflip delayed entry')
        block.extend(['from skills.backflip.rebound_task import configure as configure_rebound',
                      f'configure_rebound(env_cfg,{rebound_cycles!r},{rebound_max_seconds!r})'])
    if reward_scales:
        from .reward_options import parse_scales
        import json
        values=parse_scales(json.dumps(reward_scales))
        block.extend(['from skills.backflip.reward_options import apply_scales',f'apply_scales(env_cfg,{values!r})'])
    return text.replace(needle,'\n    '.join(block))
