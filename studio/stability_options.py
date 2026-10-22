"""Typed stability controls shared by the web form and command builder."""
import math


def field(label, default, low=None, high=None, choices=None):
    return dict(label=label, default=default, min=low, max=high, choices=choices,
                type='boolean' if type(default) is bool else 'number' if type(default) in (int, float) else 'string',
                step=1 if type(default) is int else 'any')


COMMON = {
    'fixstand_roundtrip': field('四足稳定起止', False),
    'finetune_learning_rate': field('微调学习率（0沿用训练器）', 0., 0., .001),
}
SCHEMAS = {
    'sideflip_native_ppo': {**COMMON,
        'sideflip_direction': field('侧翻方向', 'left', choices=['left','right']),
        'motion': field('项目内参考动作NPZ（空白使用原参考）', ''),
        'boundary_residual': field('边界残差控制', False),
        'boundary_residual_limit': field('边界残差上限 rad', .05, .01, .3),
        'boundary_bridge_limit': field('衔接残差上限（0沿用训练器）', 0., 0., .3),
        'boundary_terminal_hip_taper_frames': field('末段髋关节渐收帧数', 0, 0, 50),
        'boundary_blend_frames': field('边界连续混合帧数', 0, 0, 20),
        'quiet_preparation': field('准备段消抖', False),
        'peak_endpoint_objective': field('起止峰值约束', False),
        'landing_constraints': field('落地约束', False),
        'substep_objective': field('逐物理子步峰值约束', False),
        'reset_std': field('重设策略噪声（0保留）', 0., 0., 1.),
    },
    'upright_walk_ppo': {**COMMON,
        'stepping_gait': field('踏步动平衡奖励', False),
        'alternating_rear_gait': field('后足交替踏步', False),
        'stationary_upright': field('静态直立（与踏步互斥）', False),
        'in_place_stepping': field('原地踏步（零位移指令）', False),
        'endpoint_residual': field('起止残差控制', False),
        'endpoint_residual_limit': field('起止残差上限 rad', .2, .01, .2),
        'upright_reset_fraction': field('直立阶段初始化比例', 0., 0., .5),
        'reset_std': field('重设策略噪声（0保留）', 0., 0., 1.),
    },
    'backflip_native_ppo': {**COMMON,
        'peak_contact_forces': field('物理子步接触峰值', False),
        'flip_direction': field('空翻方向', 'back', choices=['back', 'front']),
        'delayed_entry': field('稳定承重后进入动作', False),
        'entry_hold_scale': field('入口残差幅度', .4, 0., .4),
        'urdf_speed_limits': field('URDF关节速度约束', False),
        'fast_safety_curriculum': field('快速约束课程', False),
        'safety_start': field('约束课程起点（-1沿用）', -1., -1., 1.),
        'initial_std': field('初始策略噪声（0沿用）', 0., 0., 1.),
        'reset_optimizer': field('续训重置优化器', False),
        'min_action_std': field('策略噪声下限（0沿用）', 0., 0., 1.),
        'max_action_std': field('策略噪声上限（0沿用）', 0., 0., 1.),
    },
}
SCHEMAS['frontflip_native_ppo'] = {**SCHEMAS['backflip_native_ppo'],
    'peak_contact_forces': field('物理子步接触峰值', True),
    'flip_direction': field('空翻方向', 'front', choices=['front'])}
SCHEMAS['backflip_native_ppo']['flip_direction'] = field('空翻方向', 'back', choices=['back'])


def defaults(rid):
    return {k: v['default'] for k, v in SCHEMAS.get(rid, {}).items()}


def validate(rid, options):
    schema = SCHEMAS.get(rid, {})
    for key, value in options.items():
        if key not in schema:
            continue
        f = schema[key]
        if f['type'] == 'number':
            if type(value) not in (int, float) or not math.isfinite(value) or not f['min'] <= value <= f['max'] or (f['step'] == 1 and type(value) is not int):
                raise ValueError('Invalid numeric stability option: ' + key)
        elif type(value) is not type(f['default']) or (f['choices'] and value not in f['choices']):
            raise ValueError('Invalid stability option: ' + key)
    o = {**defaults(rid), **options}
    def requires(flag, condition):
        if o.get(flag) and not condition:
            raise ValueError('Incompatible stability option: ' + flag)
    requires('delayed_entry', o.get('fixstand_roundtrip'))
    requires('peak_contact_forces', o.get('fixstand_roundtrip'))
    requires('fast_safety_curriculum', o.get('urdf_speed_limits'))
    requires('boundary_residual', o.get('fixstand_roundtrip'))
    requires('peak_endpoint_objective', o.get('fixstand_roundtrip'))
    requires('quiet_preparation', o.get('boundary_blend_frames'))
    requires('landing_constraints', o.get('peak_endpoint_objective'))
    requires('substep_objective', o.get('landing_constraints'))
    requires('stepping_gait', o.get('transfer_motors'))
    requires('alternating_rear_gait', o.get('stepping_gait') and o.get('fixstand_roundtrip'))
    requires('stationary_upright', o.get('transfer_motors') and o.get('fixstand_roundtrip') and not o.get('stepping_gait') and not o.get('alternating_rear_gait'))
    requires('in_place_stepping', o.get('transfer_motors') and o.get('fixstand_roundtrip') and o.get('stepping_gait') and o.get('alternating_rear_gait') and not o.get('stationary_upright'))
    requires('endpoint_residual', o.get('fixstand_roundtrip') and (o.get('stepping_gait') or o.get('stationary_upright')))
    requires('upright_reset_fraction', o.get('fixstand_roundtrip'))
    for k, high in [('boundary_blend_frames', 20), ('boundary_terminal_hip_taper_frames', 50)]:
        requires(k, o.get('boundary_residual') and 2 <= o.get(k, 0) <= high)
    requires('boundary_bridge_limit', o.get('boundary_residual') and .01 <= o.get('boundary_bridge_limit', 0) <= o.get('boundary_residual_limit', 0))
    for key in ('reset_std', 'initial_std', 'min_action_std', 'max_action_std'):
        requires(key, o.get(key, 0) >= .01)
    if rid in ('backflip_native_ppo', 'frontflip_native_ppo'):
        if o['flip_direction'] == 'front' and not (o['fixstand_roundtrip'] and o['delayed_entry']):
            raise ValueError('Frontflip requires FIXSTAND and delayed entry')
        if o['entry_hold_scale'] != .4 and not o['delayed_entry']:
            raise ValueError('Entry hold scale requires delayed entry')
        if o['safety_start'] != -1 and not (0 <= o['safety_start'] <= 1 and o['fast_safety_curriculum']):
            raise ValueError('Safety start requires fast safety curriculum')
        if o['min_action_std'] and o['max_action_std'] and o['min_action_std'] > o['max_action_std']:
            raise ValueError('Action noise minimum exceeds maximum')
        for key in ('finetune_learning_rate', 'initial_std'):
            requires(key, o['fixstand_roundtrip'])
    if rid == 'upright_walk_ppo' and o['endpoint_residual_limit'] != .2 and not o['endpoint_residual']:
        raise ValueError('Endpoint limit requires endpoint residual')


def arguments(rid, options):
    result = []
    for key, f in SCHEMAS.get(rid, {}).items():
        value = options[key]
        if key == 'motion':
            continue  # resolved against the trusted project roots separately
        if key in ('finetune_learning_rate', 'reset_std', 'initial_std', 'min_action_std', 'max_action_std', 'boundary_bridge_limit') and value == 0:
            continue
        if key == 'safety_start' and value == -1:
            continue
        flag = '--' + (key if key in ('reset_optimizer', 'min_action_std', 'max_action_std') else key.replace('_', '-'))
        if type(value) is bool:
            if value: result.append(flag)
        else:
            result += [flag, str(value)]
    return result
