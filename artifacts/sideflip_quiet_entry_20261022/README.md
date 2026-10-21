# 侧翻准备段消抖诊断与v9启动证据

本目录记录固定v8权重的控制衔接探索和v9启动配置，不是新合格权重发布。

- `outputs/fixstand_failure_diagnostics_20261022`：v8侧翻和v6后翻的原合约逐物理步诊断及summary。诊断没有写入物理状态，也没有修改判据。
- `outputs/sideflip_entry_blend_ablation_20261022`：2/5/10/20帧连续衔接的单种子结果及派生控制元数据。网络文件沿用已发布v8的policy SHA，未重复存储；`diagnostic_control_override.trained_with_this_mapping=false`明确标注这些不是按新合约训练的模型。都发生额外接地，不晋级。
- `outputs/fixstand_sideflip_quiet_bridge_v9_20261022_smoke`：32环境2更新完成、实际YAML及合约；正式目录是2048环境1500更新启动配置，没有completion即没有训练完成证据。

本地及服务器52项离线测试通过，包含连续映射的训练/部署数值一致性、准备期惩罚不压制腾空动作、关节顺序和旧契约回归。后续独立种子8900–8903必须同时通过准备期、动作、首尾及URDF约束。现有网页v8发布保持不变。

根因、数值和下一步见 [准备段修复说明](../../docs/SIDEFLIP_QUIET_ENTRY.md)。原权重在 [已发布空翻资源](../fixstand_best_flips_20261022/README.md)。
