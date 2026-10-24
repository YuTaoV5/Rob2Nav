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
        'joint_kp': field('关节位置增益 Kp', 40., 1., 80.),
        'joint_kd': field('关节阻尼增益 Kd', 1., 0., 8.),
        'target_rate_weight': field('全动作目标变化率惩罚权重（0关闭）', 0., 0., 100.),
        'target_rate_margin': field('目标变化率相对URDF速度的起罚比例', .8, .1, 1.),
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
        'in_place_airtime': field('零指令真实抬脚落地奖励', False),
        'in_place_airtime_weight': field('真实抬脚落地奖励权重', 60., 0., 200.),
        'endpoint_residual': field('起止残差控制', False),
        'endpoint_residual_limit': field('起止残差上限 rad', .2, .01, .2),
        'upright_reset_fraction': field('直立阶段初始化比例', 0., 0., .5),
        'reset_std': field('重设策略噪声（0保留）', 0., 0., 1.),
    },
    'backflip_native_ppo': {**COMMON,
        'reward_scales_json': field('前后翻奖励权重JSON（前翻可用 ang_vel_x 抑制横滚，默认0）', '{}'),
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
SCHEMAS['backflip_native_ppo']['rebound_cycles'] = field('连续后翻次数（1为原单翻；2/3为接续训练）',1,1,3)
SCHEMAS['backflip_native_ppo']['rebound_max_seconds'] = field('中间落地至再次离地确认时限 秒',.45,.05,1.)
SCHEMAS['backflip_native_ppo']['rebound_phase_version'] = field('接续阶段版本（v2含落地吸收）','legacy_v1',choices=['legacy_v1','landing_v2'])
SCHEMAS['backflip_native_ppo']['rebound_landing_max_seconds'] = field('v2落地吸收时限 秒',.25,.02,1.)
SCHEMAS['backflip_native_ppo']['rebound_min_air_rotation'] = field('v2纯腾空最小后翻转角 弧度',math.pi,.1,5.5)
SCHEMAS['backflip_native_ppo']['rebound_landing_upright_cos'] = field('v2落地确认朝上余弦',.9,.1,1.)
SCHEMAS['backflip_native_ppo']['rebound_target_rate_weight'] = field('v2映射目标速率惩罚权重',0.,0.,100.)
SCHEMAS['backflip_native_ppo']['rebound_target_rate_margin'] = field('v2映射目标速率惩罚起点/URDF限速',.8,.1,1.)
SCHEMAS['backflip_native_ppo']['rebound_absorption_upright_weight'] = field('中间落地直立奖励（仅吸收阶段）',0.,0.,100.)
SCHEMAS['backflip_native_ppo']['rebound_absorption_height_weight'] = field('中间落地高度奖励（不要求停稳）',0.,0.,100.)
for _key, _field in {
    'data': field('首翻保留数据目录（项目相对路径）',''),
    'manifest_sha256': field('首翻保留 manifest SHA256',''),
    'weight': field('首翻保留权重（0关闭）',0.,0.,100.),
    'learning_rate': field('首翻保留独立SGD学习率',.00001,.000000001,.001),
    'batch_size': field('首翻保留每轮采样数',128,1,4096),
}.items():
    SCHEMAS['backflip_native_ppo']['rebound_teacher_'+_key]=_field
SCHEMAS['sideflip_native_ppo']['boundary_terminal_hip_taper_frames']['help']='0保留有界学习补偿；2–50帧将末段髋残差衰减到0。参考归零不保证实测站姿归零，必须独立验收。'
for _rid in ('sideflip_native_ppo','backflip_native_ppo','frontflip_native_ppo'):
    SCHEMAS[_rid]['evaluation_seed_start']=field('独立四种子验收起点（自动评估时使用）',20000,0,2147483644)


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
    requires('target_rate_weight',o.get('boundary_residual'))
    requires('peak_endpoint_objective', o.get('fixstand_roundtrip'))
    requires('quiet_preparation', o.get('boundary_blend_frames'))
    requires('landing_constraints', o.get('peak_endpoint_objective'))
    requires('substep_objective', o.get('landing_constraints'))
    requires('stepping_gait', o.get('transfer_motors'))
    requires('alternating_rear_gait', o.get('stepping_gait') and o.get('fixstand_roundtrip'))
    requires('stationary_upright', o.get('transfer_motors') and o.get('fixstand_roundtrip') and not o.get('stepping_gait') and not o.get('alternating_rear_gait'))
    requires('in_place_stepping', o.get('transfer_motors') and o.get('fixstand_roundtrip') and o.get('stepping_gait') and o.get('alternating_rear_gait') and not o.get('stationary_upright'))
    requires('in_place_airtime',o.get('in_place_stepping'))
    requires('endpoint_residual', o.get('fixstand_roundtrip') and (o.get('stepping_gait') or o.get('stationary_upright')))
    requires('upright_reset_fraction', o.get('fixstand_roundtrip'))
    for k, high in [('boundary_blend_frames', 20), ('boundary_terminal_hip_taper_frames', 50)]:
        requires(k, o.get('boundary_residual') and 2 <= o.get(k, 0) <= high)
    requires('boundary_bridge_limit', o.get('boundary_residual') and .01 <= o.get('boundary_bridge_limit', 0) <= o.get('boundary_residual_limit', 0))
    for key in ('reset_std', 'initial_std', 'min_action_std', 'max_action_std'):
        requires(key, o.get(key, 0) >= .01)
    if rid in ('backflip_native_ppo', 'frontflip_native_ppo'):
        from skills.backflip.reward_options import parse_scales
        if parse_scales(o['reward_scales_json']) and not o['fixstand_roundtrip']:raise ValueError('Native reward overrides require FIXSTAND')
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
    if rid == 'backflip_native_ppo':
        if o['rebound_teacher_weight']:
            if o['rebound_phase_version']!='landing_v2' or not o['fixstand_roundtrip'] or not o['delayed_entry']:
                raise ValueError('Teacher retention requires delayed FIXSTAND landing_v2')
            from pathlib import PurePosixPath, PureWindowsPath
            value=o['rebound_teacher_data']
            if not value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or '..' in value.replace('\\','/').split('/'):
                raise ValueError('Teacher data must be a project relative directory')
            digest=o['rebound_teacher_manifest_sha256']
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Teacher manifest requires lowercase SHA256')
        elif o['rebound_teacher_data'] or o['rebound_teacher_manifest_sha256']:
            raise ValueError('Teacher data requires nonzero retention weight')
        if o['rebound_phase_version']=='landing_v2':
            if o['rebound_cycles']==1:raise ValueError('Landing v2 requires multiple cycles')
            # Studio runs without torch; retain the same cross-field bound here.
            if not .01 < o['rebound_landing_max_seconds'] <= o['rebound_max_seconds']:
                raise ValueError('Landing absorption must fit inside rebound budget')
        elif any(o[k]!=defaults(rid)[k] for k in ('rebound_landing_max_seconds','rebound_min_air_rotation',
                 'rebound_landing_upright_cos','rebound_target_rate_weight','rebound_target_rate_margin','rebound_absorption_upright_weight','rebound_absorption_height_weight')):
            raise ValueError('Landing/target-rate options require landing_v2')
        if o['rebound_cycles'] != 1 and not (o['fixstand_roundtrip'] and o['delayed_entry']):
            raise ValueError('Rebound training requires delayed FIXSTAND entry')
        if o['rebound_cycles'] == 1 and o['rebound_max_seconds'] != .45:
            raise ValueError('Rebound interval requires multiple cycles')
    if rid == 'upright_walk_ppo' and o['endpoint_residual_limit'] != .2 and not o['endpoint_residual']:
        raise ValueError('Endpoint limit requires endpoint residual')


def arguments(rid, options):
    result = []
    for key, f in SCHEMAS.get(rid, {}).items():
        value = options[key]
        if key.startswith('rebound_teacher_') and not options.get('rebound_teacher_weight'):continue
        if key.startswith('rebound_') and options.get('rebound_cycles',1)==1:continue
        if key.startswith('rebound_') and key not in ('rebound_cycles','rebound_max_seconds') and options.get('rebound_phase_version','legacy_v1')=='legacy_v1':continue
        if key == 'motion':
            continue  # resolved against the trusted project roots separately
        if key == 'evaluation_seed_start':continue  # evaluator option, never a trainer argument
        if key=='reward_scales_json' and value.strip()=='{}':continue
        if key == 'in_place_airtime_weight' and not options.get('in_place_airtime'):continue
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
