# 技能权重来源与验收总表

核对日期：2026-10-21。按技能计数，不把一个检查点的PT、ONNX和TorchScript计为三个模型。以实际记录为准；“selected”不是质量认证。

## 统计口径

- 工作台7项训练技能中，**3项有在限定任务上全数通过的候选**：楼梯、历史保留高台、PPO导航。这只是指定场景/种子的任务验收，不表示通用鲁棒性、跨引擎或实机达标。
- **2项特技有动作演示通过记录但整体不达标**：侧翻、后翻。
- 多地形部分通过；直立行走不通过。
- **首尾FIXSTAND专项：当前0项通过**；侧翻第一轮0/4，其他正式训练/评估尚在推进。
- **实机资格：0项**。没有真实Go2验收记录。
- 7项都有服务器训练产物；其中楼梯/高台/多地形、导航明确依赖下载的MoE底层，不能称为全部从零自训。

## 工作台7项及对应版本

下表R表示 `artifacts/release_20261021/weights/`，C表示 `artifacts/capability_integration/`。

| 技能/版本 | 权重路径 | 来源与服务器训练 | 实际验收 | 当前结论 |
|---|---|---|---|---|
| 楼梯·历史保留 | C/retention_stairs_v2/model_199.pt | 下载MoE基座+服务器感知/融合训练 | 原生PhysX，stairs/flat/platform各32/32；AME两类窄支撑场景各难度均0/32 | 限定楼梯任务通过；非全地形 |
| 楼梯·手册复跑 | R/stairs/manual_reproduction/model_199.pt | 同一训练体系，新200次更新 | 指定楼梯32/32 | 限定任务通过 |
| 高台·历史保留 | C/retention_platform_v2/model_199.pt | 下载MoE基座+服务器感知/融合训练 | 原生PhysX，platform/flat/stairs各32/32；AME两类窄支撑均0/32 | 限定高台任务通过 |
| 高台·手册复跑 | R/platform/manual_reproduction/model_199.pt | 同体系新200次更新，权重与历史版不同 | 高台18/32 | 部分通过，不能沿用历史32/32 |
| 多地形·手册复跑 | R/multiterrain/manual_reproduction/model_199.pt | 下载MoE基座+服务器感知/融合训练 | flat31/32、stairs32/32、platform32/32，共95/96 | 部分通过；非完整鲁棒性通过 |
| PPO导航 | skills/navigation/policy/navigation.zip | 服务器实际训练100352步的PPO速度头；MoE底层冻结 | MuJoCo独立种子1300–1303/1400–1403共8/8；网页目标操作 | 限定两类布局任务通过；非任意地图保证 |
| 侧空翻·选定DC5200 | R/sideflip/selected/model_5200.pt；exported/policy.pt | 服务器5000次更新后电机包络适配；不是下载的成品策略 | MuJoCo独立3100–3107动作8/8；速度门限0/8，最高额定速度比1.08505 | 动作演示可用，整体不达标；不满足新首尾标准 |
| 侧空翻·其他旧版 | R/sideflip/native_5000/model_4999.pt；dc_final_unvalidated/model_5998.pt | 服务器原生训练/续训 | 原生和跨引擎记录分别保留；5998未独立验收 | 不因编号更大而优于5200 |
| 后空翻·4999 | R/backflip/selected/model_4999.pt；exported/policy.onnx | 服务器4096环境5000次更新，从项目训练实现复现 | 源MuJoCo完成约−6.151rad翻转及落稳；严格落地/关节限制失败，记录峰值77.96rad/s、越限0.5146rad | 动作演示可用，整体不达标；非标准首尾 |
| 直立行走·4999 | R/upright_walk/representative_failed_baseline/model_4999.pt | 服务器4096环境5000次更新 | 原生起立/前移伴滑移，没有真实换步；MuJoCo失败 | 不达标，无合格“最佳行走权重” |
| 直立·其他旧版 | R/upright_walk/dc_final_unvalidated/model_5998.pt；stepping_snapshot_unvalidated/model_5700.pt | 服务器DC适配/迈步重训；5700为中断快照 | 未完成完整独立能力验收 | 研究候选，不能推广 |

地形训练配方warmup20/fusion180/actor0，以已有V2感知检查点初始化，冻结137k MoE基座；更新的是上层模块。网页地形MuJoCo演示使用预训练MoE与地形控制，并非上述PhysX感知检查点迁移通过。

## 最新首尾FIXSTAND专项

服务器根目录 `/root/gpufree-data/Rob2Nav/`。以下截至本次在线查询；运行状态会变化，最终以completion.json和独立summary为准。

| 技能/版本 | 服务器输出目录 | 来源 | 当前结果 |
|---|---|---|---|
| 侧翻v1 | outputs/fixstand_finetune_20261021/sideflip | 从DC5200续训500更新 | model_5699完成；6100–6103联合验收0/4 |
| 侧翻v2 | outputs/fixstand_sideflip_residual_v2_20261021 | 从DC5200续训；首尾参考有界残差合约 | 正式500更新仍在运行；短程试验端点误差约0.16rad且身体触地，不通过 |
| 直立阶段版 | outputs/fixstand_finetune_20261021/upright | 从4999迁移，48→52维阶段观测 | 正式训练/评估未结束，不能沿用旧48维执行器 |
| 后翻阶段版 | outputs/fixstand_finetune_20261021/backflip | 从4999续训，强化标准站姿保持 | 编排任务运行中，尚无通过的正式评估 |

三项32环境2更新仅验证训练、导出、MuJoCo接口，全部未通过新目标，不能算合格权重。详细合约和验收阈值见[专项手册](../skills/FIXSTAND_ROUNDTRIP.md)。

## 下载权重与无训练权重功能

| 项目/功能 | 来源/文件 | 服务器是否训练它 | 证据与用途 |
|---|---|---|---|
| RoboGauge MoE137k | vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt | 未重训该基础文件 | 当前主要运动基座；支撑导航、地形演示、切换仿真；是下载权重中实际复用最多的一个，但不是独立实机认证 |
| RoboGauge CTS/MLP100k | 同目录go2_cts_max2_100k.pt | 否 | 兼容/对照；紧凑切换回归有失败，不列为当前推荐 |
| RoboGauge MoE124k、MoE8专家100k | 同目录go2_moe_cts_124k.pt、go2_moe_cts_8experts_2cmd_100k.pt | 否 | 已下载保留；没有当前完整匹配验收，不按文件存在计达标 |
| LeggedSkillDeploy跳跃/后翻/侧翻 | 外部固定提交9126dad；artifacts/capability_integration/lsd_*为推理记录 | 这些外部成品未在此完成本地重训 | diagnostic_only、未建立独立通过判据；不能称为效果好的已达标权重；许可不明不再分发其成品 |
| NJUAgility/PIE | vendor/NJUAgility/models/bbc2.pt、bbc3.pt；pie_source_inference记录 | 外部成品，不是本次服务器训练 | 有源推理展示，未完成原生复训/独立技能验收；不计达标 |
| SCAN导航 | 固定社区源码+本项目适配+MoE137k | SCAN规划器没有学习权重；本次没有新增SCAN训练 | 原静态场景2/2，另有固定动态任务记录；这是算法链验收，不能计新增模型 |
| 左右爪托手 | go2_nav/paw_contact控制器 | 无RL训练权重 | MuJoCo接触/抗扰/撤手等检查；真实相机识别及实机尚未验证 |
| FIXSTAND/键盘/热切换 | PD控制、速度输入及状态机 | 无独立神经网络权重 | 属于部署机制，不增加技能模型数量 |

## 审计入口

- [发布模型索引与SHA256](../artifacts/release_20261021/model_index.json)、[原始资源清单](../artifacts/release_20261021/manifest.json)。
- [历史保留楼梯评估](../artifacts/capability_integration/retention_stairs_v2/evaluation/stairs_0.50/summary.json)、[历史保留高台评估](../artifacts/capability_integration/retention_platform_v2/evaluation/platform_0.50/summary.json)。
- [手册复跑评估目录](../artifacts/release_20261021/evaluations/)、[导航训练来源](../skills/navigation/TRAINING.md)。
- [专项失败与运行快照](../artifacts/fixstand_20261021/README.md)、[最新交接](../handover.md)。

所有“限定通过”均只适用于匹配权重哈希、模型、执行器、场景和判据。当前没有一个特技同时通过动作、电机约束、标准首尾和实机验收。
