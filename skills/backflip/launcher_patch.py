"""Reviewed environment injection for the pinned upstream training entry."""


def configure_fixstand_source(text,source,urdf_speed_limits=False,fast_safety_curriculum=False,
                              delayed_entry=False,entry_hold_scale=.4):
    needle='env_cfg = Go2BackflipEnvCfg()'
    if text.count(needle)!=1:raise RuntimeError('Pinned environment config entry changed')
    block=[needle,'from skills.fixstand import configure_backflip',
        f'configure_backflip(env_cfg,{str(source)!r} if {urdf_speed_limits!r} else None,{fast_safety_curriculum!r})']
    if delayed_entry:
        block.extend(['from skills.fixstand import configure_backflip_delayed_entry',
            f'configure_backflip_delayed_entry(env_cfg,{entry_hold_scale!r})',
            'if env_cfg.episode_length_s!=5. or getattr(env_cfg,"entry_delay_seconds",None)!=1.:',
            '    raise RuntimeError("Delayed-entry training configuration was not applied")'])
    return text.replace(needle,'\n    '.join(block))
