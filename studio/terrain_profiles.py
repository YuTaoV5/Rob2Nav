"""Web presets share the exact geometry and completion protocol of their evidence."""


def terrain_launch(skill):
    scene={'stairs':'stairs','platform':'platform','multiterrain':'mixed','obstacles':'hurdles'}[skill]
    args=['--scene',scene,'--locomotion','moe','--max-seconds','60','--episodes','1','--seed','12010']
    if skill in ('stairs','platform'):
        args+=['--no-jump','--roundtrip']
        if skill=='platform':args+=['--height','.20']
        note=('原始RoboGauge MoE：4级×8cm楼梯上、下行' if skill=='stairs' else '原始RoboGauge MoE：20cm高台上、下台')+'，终点站稳2秒；固定场景4回合通过，未获实机资格。'
    elif skill=='multiterrain':
        note='研究仿真：原始MoE+感知调度+关节跳跃原语；已启用跨沟触发，但落地仍失败，混合能力未验收。'
    else:
        args+=['--no-jump']
        note='原始MoE障碍场景；实际结果以本次运行记录为准。'
    return args,note
