# FIXSTAND 续训与证据（2026-10-22）

目标为同一策略自主完成 **标准四足站立 → 动作 → 标准四足站立**，之后才进入统一部署的热切换验证。`successes` 是动作及首尾姿态判据，`hardware_constraints_pass` 另外检查 MuJoCo 中按 URDF 限定的关节速度比 ≤1 与硬关节限位超出 ≤0.001 rad。只有四个独立种子同时满足两者，才写作 `passed_simulation_only`；这个状态仍不等于 Go2 Edu 实机资格。

| 技能/服务器训练 | 独立评估 | 结论 |
| --- | --- | --- |
| 侧空翻 v4，`outputs/fixstand_sideflip_bridge_v4_20261021` | 8200–8203：0/4；完成约一周侧翻，结束姿态最大误差约 0.148–0.150 rad（门槛 0.12）；速度比约 1.095 | 首尾与速度均未达标。v4 是 v5 续训起点。 |
| 后空翻 v3，`outputs/fixstand_backflip_quiet_entry_v3_20261022` | 9500–9503：动作及首尾 4/4；速度比 1.656–1.679，硬限位无超出 | **未达标**。旧评估脚本曾仅按动作结果误写 `passed_simulation_only`，已改为 `not_accepted`；v4 用真实 URDF 速度上限续训。 |
| 直立行走 v2，`outputs/fixstand_upright_endpoint_v2_20261021` | 9100–9103：0/4；直立仅约 0.32–0.34 s，未完成后足承重步态周期 | 未达标。v3 从此权重续训。 |
| 侧空翻 v5，`outputs/fixstand_sideflip_terminal_v5_20261022` | 8400–8403：0/4；结束姿态误差约 0.267 rad，起始稳定比例约 0.69，速度比约 1.067 | 比 v4 退化；没有把最后检查点当成最佳模型。 |

侧空翻 v4 的冻结权重做过**仅诊断、不声称训练**的髋关节末段残差衰减：8200–8203 的结束最大姿态误差降至约 0.130–0.131 rad，仍高于 0.12；速度比和限位超出不变。全关节衰减使小腿姿态更差，因此训练 v6 只衰减髋关节残差，其他关节保持承重控制。`reference_residual_v3` 是新动作合约，训练、MuJoCo 验收和统一部署采用同一相位映射；旧 v2 模型不能直接改名为 v3 并部署。

后空翻 v3 超速的根因之一是上游训练配置把所有 12 关节上限设成 30 rad/s，而钉住的 Go2 URDF 小腿关节上限为 15.7 rad/s。v4 从 URDF 逐关节读取上限（髋/大腿 30.1，小腿 15.7 rad/s），将安全课程在续训初期加速生效，并以四个新种子重新验收。早期训练日志出现的速度比例下降只能说明训练约束已生效，不能代替最终 MuJoCo 结果。

服务器上并行训练和条件队列：

- `bash scripts/train_sideflip_fixstand_v5.sh`：2048 环境、1000 次更新，从 v4 `model_6198.pt` 续训，8400–8403 独立验收；**已完成但未达标**。
- `bash scripts/train_upright_fixstand_v3.sh`：2048 环境、1500 次更新，从 v2 `model_1499.pt` 续训，9700–9703 验收。
- `bash scripts/train_backflip_fixstand_v4.sh`：1024 环境、1500 次更新，从 v3 `model_7496.pt` 续训，9600–9603 验收。
- `bash scripts/train_sideflip_fixstand_v6.sh`：原条件队列在 v5 失败后短暂启动，因 v5 已明显退化而停止；保留部分检查点仅作中止记录，**没有独立验收**。
- `bash scripts/train_sideflip_fixstand_v7.sh`：改从 v4 `model_6198.pt` 分支，1024 环境、1000 次更新，以髋关节末段衰减合约续训，8700–8703 独立验收。
- `bash scripts/train_upright_fixstand_v4.sh`：**条件队列**。v3 若四种子未达标，从其最后检查点续训；25% 训练重置从平衡直立姿态及动作相位开始，75% 仍从标准四足站姿开始，以增加动作阶段的承重步态样本。独立验收依然 100% 从标准四足站姿开始，9800–9803 四种子。32 环境、2 次更新的原生 Isaac 接口烟测已生成 `completion.json`，只证明环境能运行。

这些训练各有独立输出目录，不能把正在生成的检查点或 `completion.json` 当作合格证明。服务器当前没有经过这套新门槛验收的特技权重，因此 `deployment/skills/` 中的特技仿真与实机资格保持关闭；原 RoboGauge MoE 平地与楼梯仍共用一份 45 维权重。`stand → 特技 → stand` 及长任务热切换要等各模型完全通过独立种子后，再作统一运行器 MuJoCo 回归；实机还需单独设备资格记录。

直立 v3 的 `model_2100.pt` 仅做过 9700 单种子中途诊断：直立约 0.74 s、未完成承重迈步，首尾站稳仍未通过。这比旧 v2 的约 0.33 s 有进展，但不能预测最终检查点，更不能提前晋级。v4 的混合重置只发生于训练 episode 初始化，通过额外相位偏移让策略知道自己从动作段进入；正式 MuJoCo 和统一部署都不使用该偏移，仍由标准站姿开始。

## 可复现文件

[`artifacts/fixstand_resume_20261022/`](../artifacts/fixstand_resume_20261022/) 内固定了三条续训**父检查点**、导出模型、训练环境/代理参数、动作合约、四种子评估和侧空翻参考 NPZ。它们是服务器本次训练得到的候选，而非外部下载的合格权重。当前运行的 v5/v6、v4 后翻、v3 直立的**最终**权重只有完成和验收后才应加入发布资源。文件哈希见该目录的 `SHA256SUMS`。

在固定上游版本、Isaac Sim/Isaac Lab 和依赖环境准备好后，恢复 `artifacts/fixstand_resume_20261022/outputs/*` 到服务器仓库同名 `outputs/*`，运行以上脚本；侧空翻需同时恢复 `outputs/fixstand_finetune_20261021/sideflip.npz` 及 JSON 合约。脚本中的根目录可通过 `ROB2NAV_ROOT` 设置，Go2 上游源码默认路径仍为 `/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip` 与 `unitree_rl_lab`。不要将冻结父权重的诊断重新解释为训练结果。配置、热切换按键和机载电脑连接见 [Go2 Edu 部署指南](GO2_EDU_HOT_SWITCH.md)。
