"""Audited native experiment semantics; no Isaac or UI dependency."""


def term(label,weight,category,unit,formula,description,risk,requires='always'):
    return dict(label=label,weight=weight,category=category,unit=unit,formula=formula,description=description,risk=risk,requires=requires)


REWARDS={
 'track_lin_vel_xy_exp':term('平面速度跟踪',1.5,'任务','无量纲','exp(-||v_xy-command_xy||² / σ²)','本体坐标速度指令跟踪；σ等参数以env.yaml为准。','过强可能追速度而忽略落脚。','locomotion'),
 'track_ang_vel_z_exp':term('偏航速度跟踪',.75,'任务','无量纲','exp(-(ω_z-command_z)² / σ²)','跟踪偏航角速度。','不是绝对航向误差。','locomotion'),
 'forward_progress':term('前进/目标方向进度',1.,'任务','m/s','clip(v_world_x,-1,1.2)','导航模式改用目标奖励模板；减速实验在近目标处衰减进度。','高权重可能导致冲过目标或冒险。'),
 'goal_motion':term('目标运动',5.,'任务','无量纲','近目标或朝目标以0.15–1.5m/s运动的指示值','arrival_aware开启后改为距离相关期望速度的指数跟踪。','两种模板公式不同，不能只比较权重。','navigation'),
 'goal_position':term('目标位置接近',100.,'任务','无量纲','1/(1+0.25*d²) * I(剩余时间<4s) /4','回合末段才生效。','大权重可能促进等待，需配合完整通过条件。','navigation'),
 'route_bonus':term('完整通过事件',500.,'任务','事件','I(route_complete)','使用既有路线距离、站稳等通过条件，不等于单纯前进。','不能用此奖励大小替代评估通过率。','navigation'),
 'termination':term('失败终止',-10.,'稳定','事件','I(terminated)','导航模板默认-100；奖励管理器仍按环境步长处理。','过强可能诱导原地不动。'),
 'lane_center':term('横向偏移',-.5,'任务','m²','(y_world-y_origin)²','wide_deep模板默认-2。','过强会妨碍绕障或横向落脚。'),
 'lin_vel_z_l2':term('竖直速度',-2.,'稳定','(m/s)²','v_body_z²','抑制机身上下振荡。','跳跃、高台冲跃任务需要谨慎调低。'),
 'ang_vel_xy_l2':term('横滚/俯仰角速度',-.05,'稳定','(rad/s)²','ω_body_x²+ω_body_y²','抑制身体旋转振荡。','与翻转、恢复技能目标冲突。'),
 'flat_orientation_l2':term('机身水平姿态',0.,'稳定','无量纲','g_projected_x²+g_projected_y²','基础模板存在，但默认权重0。','楼梯/高台不应强制始终水平，翻转任务不可盲用。'),
 'dof_torques_l2':term('关节力矩平方',-.0002,'能耗与平滑','(N·m)²','Σ τ_applied²','抑制大力矩。','不是电池能耗；过强会失去越障推力。'),
 'dof_acc_l2':term('关节加速度平方',-2.5e-7,'能耗与平滑','(rad/s²)²','Σ q̈²','抑制急剧关节加速。','数值量级很大，不能与其他权重直接比较。'),
 'action_rate_l2':term('动作变化率',-.01,'能耗与平滑','动作²','Σ(a_t-a_(t-1))²','抑制相邻策略输出抖动。','过强会降低反应速度和抬腿能力。'),
 'dof_pos_limits':term('软关节限位',-1.,'约束','rad','Σ relu(q_min-q)+relu(q-q_max)','约束软关节范围。','不是硬安全限位，不能代替底层控制检查。'),
 'undesired_contacts':term('大腿/小腿接触',-1.,'约束','接触数量','Σ I(历史峰值接触力范数>1N)','当前明确针对thigh/calf，不含所有机身碰撞。','接触物未分语义；恢复/攀爬任务可能有意接触。'),
 'feet_air_time':term('足部腾空时间',.2,'步态','s','Σ(last_air_time-0.2)*first_contact，移动指令门控','标准腾空奖励，导航模板移除此项。','过强可能形成不必要的大跨步或跳动。','locomotion'),
 'landing_support':term('落脚支撑裕度',1.,'落脚','比例','Σ(support_fraction-0.7)*first_contact，移动门控','足部附近9条理想探针在脚高附近的支撑比例。','仅训练评分真值，不直接指定落脚点。','locomotion'),
 'supported_contact':term('原支撑面落脚',1.,'落脚','比例','Σ(surface_fraction-0.7)*first_contact','按原支撑面高度评分，不把浅坑底当支撑；导航模板使用。','特权奖励不代表actor获得完整地图。','navigation'),
 'anchor_cost':term('原MoE动作保持',-2.,'保持','rad²','mean(((action-anchor)*0.25)²)*terrain_gate','平坦地形门控4，其他0.2；导航模板权重-0.2。','过强会阻止新增技能；过弱可能遗忘本体。'),
 'heading_cost':term('绝对航向误差',-2.,'任务','无量纲','1-cos(yaw)','仅wide_deep模板存在。','固定世界航向，不适用于任意转弯目标。','wide_deep'),
 'descending_foot_risk':term('下降低脚支撑风险',-2.,'落脚','比例','Σ I(-.05<z<.16, vz<-.05)*(1-support_fraction)','仅特权教师foothold实验。','阈值基于该地形设计，不是通用碰撞模型。','foothold'),
 'body_height_error':term('机身离地高度误差',-1.,'姿态目标','m²','(root_z-nearest_ground_z-target)²','奖励用训练真值；仅有效近地高度0–0.8m时计分。','不是运行时高度指令；深坑/腾跃期间可能不计分。','body_height'),
 'swing_clearance_error':term('活动摆腿离地间隙误差',-.5,'姿态目标','m²','mean(I(swing,moving,valid)*(foot_z-local_ground_z-target)²)','无接触、水平脚速>0.1m/s，脚高0.02–0.5m时评分。','不是步态峰值高度或落脚轨迹约束，不能保证不撞小腿。','swing_clearance'),
}


def reward_schema(config):
    c=config if isinstance(config,dict) else config.model_dump()
    nav=c.get('navigation',True);course=c.get('course','original')
    flags={'always':True,'navigation':nav,'locomotion':not nav,'wide_deep':course=='wide_deep',
        'foothold':bool(c.get('foothold_aware')),'body_height':c.get('body_height_target_m') is not None,
        'swing_clearance':c.get('swing_clearance_target_m') is not None}
    result={}
    for name,t in REWARDS.items():
        default=t['weight']
        if nav:default={'forward_progress':5.,'anchor_cost':-.2,'termination':-100.}.get(name,default)
        if name=='lane_center' and course=='wide_deep':default=-2.
        available=flags[t['requires']]
        result[name]={**t,'available':available,'template_weight':default if available else None,
            'effective_weight':c.get('rewards',{}).get(name,default) if available else None,
            'overridden':name in c.get('rewards',{}),'source':'moe_perceptive/task.py + mode configurators; runtime env.yaml is authoritative'}
        if c.get('arrival_aware') and name=='goal_motion':result[name]['formula']='exp(-||v_world_xy-desired_velocity(delta)||²/0.16)'
        if c.get('arrival_aware') and name=='forward_progress':result[name]['formula']='clip(v_world·goal_unit,-1,1.2)*min(distance/0.8,1)'
    return result


METHODS=[
 dict(id='moe_terrain_ppo',name='MoE感知适配',optimizer='PPO',status='native',mechanisms=['冻结本体的新增分支适配','分阶段解冻','可见深度/雷达高度记忆','特权教师/观测学生输入','可选空间注意力偏置实验'],note='学生输入训练不等于已实现教师蒸馏；空间偏置是实验，不是AME原算法。'),
 dict(id='pie_ppo',name='PIE',optimizer='PPO + 辅助估计损失',status='adapter_required',mechanisms=['双层隐式/显式估计','时序深度融合','重建/预测辅助任务与VAE KL'],note='Go2源权重推理已接通，原生训练尚未接通本任务队列。'),
 dict(id='wmp_ppo',name='WMP',optimizer='世界模型 + PPO',status='adapter_required',mechanisms=['观测/动作历史世界模型','潜变量导航与越障策略'],note='审计的源训练本体为Go1，不能直接换成Go2权重。'),
 dict(id='him_ppo',name='HIM',optimizer='PPO + 历史隐式估计',status='adapter_required',mechanisms=['历史状态表示','高台/恢复任务专用训练'],note='high/hop是不同任务；原树没有匹配的已训权重。'),
 dict(id='wtw_ppo',name='Walk These Ways',optimizer='PPO + 条件步态控制',status='adapter_required',mechanisms=['步态/相位/姿态条件化','适应模块'],note='高度/步态指令必须匹配其观测与训练分布，不能加到MoE输入冒充兼容。'),
 dict(id='bbc_tsc',name='NJU BBC/TSC',optimizer='模仿学习 + 分层强化学习/感知训练',status='adapter_required',mechanisms=['参考动作跟踪技能','分层任务策略','教师/学生感知迁移'],note='已有源技能推理不等于训练器和感知导航已经全部接通。'),
 dict(id='ame_reference',name='AME方法参考',optimizer='PPO + 感知表征/注意力结构',status='adapter_required',mechanisms=['地形表征融合','精准落脚训练'],note='原AME20cm过桩未复现；当前MoE窄桩配方是本项目实验，不是原AME原生训练器。'),
 dict(id='sea_reference',name='SEA-Nav方法参考',optimizer='安全强化学习 + 可微CBF',status='adapter_required',mechanisms=['危险样本重放','探索奖励','自适应安全屏障'],note='导航高层算法，不是关节运控优化器；射线合约和原生训练仍待适配。'),
]
