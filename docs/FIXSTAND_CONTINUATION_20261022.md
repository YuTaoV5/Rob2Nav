# FIXSTAND 续训与证据（2026-10-22）

目标为同一策略自主完成 **标准四足站立 → 动作 → 标准四足站立**，之后才进入统一部署的热切换验证。`successes` 是动作及首尾姿态判据，`hardware_constraints_pass` 另外检查 MuJoCo 中按 URDF 限定的关节速度比 ≤1 与硬关节限位超出 ≤0.001 rad。只有四个独立种子同时满足两者，才写作 `passed_simulation_only`；这个状态仍不等于 Go2 Edu 实机资格。

| 技能/服务器训练 | 独立评估 | 结论 |
| --- | --- | --- |
| 侧空翻 v4，`outputs/fixstand_sideflip_bridge_v4_20261021` | 8200–8203：0/4；完成约一周侧翻，结束姿态最大误差约 0.148–0.150 rad（门槛 0.12）；速度比约 1.095 | 首尾与速度均未达标。v4 是 v5 续训起点。 |
| 后空翻 v3，`outputs/fixstand_backflip_quiet_entry_v3_20261022` | 9500–9503：动作及首尾 4/4；速度比 1.656–1.679，硬限位无超出 | **未达标**。旧评估脚本曾仅按动作结果误写 `passed_simulation_only`，已改为 `not_accepted`；v4 用真实 URDF 速度上限续训。 |
| 直立行走 v2，`outputs/fixstand_upright_endpoint_v2_20261021` | 9100–9103：0/4；直立仅约 0.32–0.34 s，未完成后足承重步态周期 | 未达标。v3 从此权重续训。 |
| 侧空翻 v5，`outputs/fixstand_sideflip_terminal_v5_20261022` | 8400–8403：0/4；结束姿态误差约 0.267 rad，起始稳定比例约 0.69，速度比约 1.067 | 比 v4 退化；没有把最后检查点当成最佳模型。 |
| 侧空翻 v7，`outputs/fixstand_sideflip_v4_hip_taper_v7_20261022` | 8700–8703：0/4；结束误差约 0.255 rad，初始稳定比例约 0.64–0.67，速度比约 1.065–1.066 | 髋关节衰减续训仍退化。v8 改用低学习率和单关节峰值误差奖励。 |
| 后空翻 v4，`outputs/fixstand_backflip_urdf_speed_v4_20261022` | 历史播放9600–9603、9620–9623均0/4；后续核对发现训练配置4秒且未注入延迟入口，导出却标5秒带入口 | **`invalid_training_contract`，实验合约无效**；不能据此推断速度约束训练的效果。 |
| 后空翻 v5，`outputs/fixstand_backflip_speed_curriculum_v5_20261022` | 9900–9903均未通过；末端站姿通过，但动作/入口与峰值约束失败。与v4同样实际训练4秒、导出标5秒带入口 | **`invalid_training_contract`，实验合约无效**；保留失败记录，从正确v3父权重启动v6。 |
| 直立行走 v3，`outputs/fixstand_upright_tight_endpoint_v3_20261022` | 9700–9703：0/4；直立3.44–3.50 s，直立段位移约0.67 m，双后足完整承重步态周期各0；速度比1.068–1.110，硬限位超出0.125–0.136 rad | 直立保持有进展，但仍属未完成真实迈步及首尾站稳的候选，未晋级。 |

侧空翻 v4 的冻结权重做过**仅诊断、不声称训练**的髋关节末段残差衰减：8200–8203 的结束最大姿态误差降至约 0.130–0.131 rad，仍高于 0.12；速度比和限位超出不变。全关节衰减使小腿姿态更差，因此训练 v6 只衰减髋关节残差，其他关节保持承重控制。`reference_residual_v3` 是新动作合约，训练、MuJoCo 验收和统一部署采用同一相位映射；旧 v2 模型不能直接改名为 v3 并部署。

后空翻 v3 超速的根因之一是上游训练配置把所有12关节上限设成30 rad/s，而固定的Go2 URDF小腿上限为15.7 rad/s。后续训练逐关节读取真实上限，但v4/v5增加配置参数后，一处旧字符串替换没有命中，延迟入口训练未注入，metadata仍宣称已注入。该问题现已通过显式注入函数、运行时配置断言与导出器交叉校验修正；v4/v5作废，v6是修复后的首条正式续训。此前仅根据训练日志或MuJoCo失败解释v4/v5策略退化的推断不再成立。

服务器上并行训练和条件队列：

- `bash scripts/train_sideflip_fixstand_v5.sh`：2048 环境、1000 次更新，从 v4 `model_6198.pt` 续训，8400–8403 独立验收；**已完成但未达标**。
- `bash scripts/train_upright_fixstand_v3.sh`：2048 环境、1500 次更新，从 v2 `model_1499.pt` 续训；**已完成，四种子未达标**。
- `bash scripts/train_backflip_fixstand_v4.sh`：1024环境1500更新；**历史运行完成但训练/导出入口合约不一致，作废**。
- `bash scripts/train_sideflip_fixstand_v6.sh`：原条件队列在 v5 失败后短暂启动，因 v5 已明显退化而停止；保留部分检查点仅作中止记录，**没有独立验收**。
- `bash scripts/train_sideflip_fixstand_v7.sh`：从 v4 `model_6198.pt` 分支，1024 环境、1000 次更新；**已完成但未达标**。
- `bash scripts/train_upright_fixstand_v4.sh`：**已自动衔接训练**，从 v3 `model_2998.pt` 续训；25% 训练重置从平衡直立姿态及动作相位开始，75% 仍从标准四足站姿开始，以增加动作阶段的承重步态样本。已核对实际训练合约的75%/25%重置分布。独立验收依然 100% 从标准四足站姿开始，9800–9803 四种子。此前32环境、2次更新烟测仅证明环境接口可运行。
- `bash scripts/train_backflip_fixstand_v5.sh`：1024环境2000更新，课程确实重启；但**入口合约不一致，作废**，不可恢复该权重继续目标训练。
- `bash scripts/train_backflip_fixstand_v6.sh`：从正确的v3 `model_7496.pt` 重新分支，1024环境2000更新；固定学习率1e-4、重置课程、前100更新热身、1500更新逐渐收紧，10000–10003独立验收。32环境2更新烟测完成，并核对实际YAML为5秒episode、1秒入口延迟、0.1入口残差；烟测检查点成功按正确合约导出，正式训练已启动。
- `bash scripts/train_sideflip_fixstand_v8.sh`：从 v4 `model_6198.pt` 分支，固定学习率3e-5并重新建立优化器，探索标准差重设0.10，增加末段最大单关节误差、URDF速度裕度及关节限位奖励；1024环境1500更新，8800–8803验收。32环境2更新原生烟测已完成，正式训练已启动。奖励变化不放宽评估标准。

这些训练各有独立输出目录，不能把正在生成的检查点或 `completion.json` 当作合格证明。服务器当前没有经过这套新门槛验收的特技权重，因此 `deployment/skills/` 中的特技仿真与实机资格保持关闭；原 RoboGauge MoE 平地与楼梯仍共用一份 45 维权重。`stand → 特技 → stand` 及长任务热切换要等各模型完全通过独立种子后，再作统一运行器 MuJoCo 回归；实机还需单独设备资格记录。

直立 v3 的 `model_2100.pt` 仅做过 9700 单种子中途诊断：直立约 0.74 s、未完成承重迈步，首尾站稳仍未通过。这比旧 v2 的约 0.33 s 有进展，但不能预测最终检查点，更不能提前晋级。v4 的混合重置只发生于训练 episode 初始化，通过额外相位偏移让策略知道自己从动作段进入；正式 MuJoCo 和统一部署都不使用该偏移，仍由标准站姿开始。

## 执行器与动作历史一致性修正

后空翻导出现在从 `params/env.yaml` 固定控制合约：关节顺序、动作缩放/截断、PD增益、逐关节速度曲线、关节目标速度和驱动/制动力矩。MuJoCo播放读取该合约；旧播放默认所有关节速度30 rad/s、动作截断100，与本轮训练小腿15.7 rad/s、动作截断8不一致。v4的完整合约复评位于`outputs/backflip_v4_control_parity_v3_20261022`，仍失败，不能把接口修复当成功。

启动器现用`skills/backflip/launcher_patch.py`一次性生成基础配置与延迟入口，禁止依赖易失效的第二次文本替换；配置未生效即终止。导出器先检查实际训练YAML与入口合约一致，再写出模型，并记录`entry_training_contract_verified=true`。原v3的实际配置为5秒/1秒入口，校验有效；v4/v5实际4秒且无入口，现会被拒绝。实机与MuJoCo都不能使用这些错误标签的导出。49项本地与服务器测试通过；原生烟测/导出审计和作废记录见[`artifacts/fixstand_backflip_entry_audit_20261022/`](../artifacts/fixstand_backflip_entry_audit_20261022/)。

统一部署工作进程返回的动作历史已转换为SDK关节顺序，观测适配器再按技能关节顺序取回，避免非SDK顺序策略的历史被重新排列。后翻延迟入口的历史使用训练时经过入场缩放及8幅度截断的动作，侧翻/直立仍使用其原始策略历史；该修正会改变运行器哈希，需要重新完成部署回归与设备资格。47项离线测试通过。新的失败评估、控制合约和训练启动参数存于[`artifacts/fixstand_followup_20261022/`](../artifacts/fixstand_followup_20261022/)。

## 可复现文件

[`artifacts/fixstand_resume_20261022/`](../artifacts/fixstand_resume_20261022/) 内固定了早期三条续训**父检查点**、导出模型、训练环境/代理参数、动作合约、四种子评估和侧空翻参考 NPZ。它们是服务器本次训练得到的候选，而非外部下载的合格权重。直立v3的最终检查点`model_2998.pt`、导出、四种子评估、首种子诊断视频及v4启动参数位于[`artifacts/fixstand_upright_v3_20261022/`](../artifacts/fixstand_upright_v3_20261022/)，用于复现续训，不授予部署资格。当前活动任务是侧翻v8、后翻v6与直立v4；各目录`SHA256SUMS`固定已归档文件。历史证据文件保持原样，新审计结果明确覆盖v4/v5旧状态。

在固定上游版本、Isaac Sim/Isaac Lab 和依赖环境准备好后，恢复 `artifacts/fixstand_resume_20261022/outputs/*` 到服务器仓库同名 `outputs/*`，运行以上脚本；侧空翻需同时恢复 `outputs/fixstand_finetune_20261021/sideflip.npz` 及 JSON 合约。脚本中的根目录可通过 `ROB2NAV_ROOT` 设置，Go2 上游源码默认路径仍为 `/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip` 与 `unitree_rl_lab`。不要将冻结父权重的诊断重新解释为训练结果。配置、热切换按键和机载电脑连接见 [Go2 Edu 部署指南](GO2_EDU_HOT_SWITCH.md)。
