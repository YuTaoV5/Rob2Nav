"""Simulation skill library. Scene launchability is distinct from hot-switch approval."""
import json
from pathlib import Path


def simulation_library(root):
    root=Path(root)
    rows=[
        ('backflip','后空翻','运行已发布的后翻候选，查看起跳、翻转和落地。','研究策略；完整验收未通过','docs/GO2_EDU_FLIP_DEPLOYMENT.md'),
        ('sideflip','侧空翻','运行已发布的侧翻候选；连续侧翻编排可在下方另行设置。','研究策略；完整验收未通过','docs/GO2_EDU_FLIP_DEPLOYMENT.md'),
        ('upright_walk','直立行走','从四足姿态尝试起立与直立步态，如实展示当前策略表现。','实验策略；直立步行未通过','skills/upright_walk/SIMULATION.md'),
        ('handshake','伸爪握手','拖动虚拟手掌，选择左爪或右爪，点击开始并查看真实接触。','交互仿真；独立接触控制器','go2_nav/paw_contact/DEPLOYMENT.md'),
        ('jump','原地跳高','执行蹲伏、蹬地、腾空与落地，显示实测高度和腾空时间。','关节动作原语；无训练权重','docs/UNIFIED_SKILL_LIBRARY.md'),
        ('stairs','走楼梯','在四级实体台阶中运行 MoE 行走与地形感知控制。','场景仿真；每次单独评估','skills/stairs/SIMULATION.md'),
        ('platform','登上高台','接近并攀上实体高台，检查抬腿与支撑过程。','场景仿真；每次单独评估','skills/platform/SIMULATION.md'),
        ('obstacles','连续越障','经过低台、窄障碍和落差，检查通过及恢复过程。','场景仿真；每次单独评估','docs/UNIFIED_SKILL_LIBRARY.md'),
        ('multiterrain','多地形穿越','在已有的水色风险区、沟壑和台阶组合中执行。','研究场景；不保证通过','skills/multiterrain/SIMULATION.md'),
    ]
    result=[]
    for ident,name,description,status,doc in rows:
        resources=[]
        if ident in ('backflip','sideflip'):
            try:
                manifest=json.loads((root/'studio/published_flips.json').read_text(encoding='utf-8'))[ident]
                resources=[root/p for p in manifest['sha256']]
            except (OSError,ValueError,KeyError):resources=[root/'missing-published-manifest']
        elif ident=='upright_walk':resources=[root/'artifacts/release_20261021/weights/upright_walk/representative_failed_baseline/exported/policy.pt']
        elif ident=='jump':resources=[root/'deployment/models/go2_isaac.xml']
        else:resources=[root/'vendor/RoboGauge/resources/robots/go2/go2.xml']
        available=all(p.is_file() for p in resources)
        result.append(dict(id=ident,name=name,description=description,status=status,available=available,
            blocker=None if available else '运行资源缺失，请先同步项目模型与资产',document=doc,
            execution_mode='interactive' if ident=='handshake' else 'isolated_scene',
            continuous_switch=False,hardware_qualified=False))
    return result
