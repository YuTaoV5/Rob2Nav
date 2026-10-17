# AME 基础步态初始化对照

这是针对原版 Go2 从零 PPO 稀疏落脚停滞的独立实验，**不是原版从零复现，也不能把教师的性能算成 AME 的过桩性能**。原始两阶段任务继续单独保存。

## 动机与观测

原版第 1000、1500、2000、2200 轮检查点中，`stepping_stones` 和 `gaps` 两类各 64 个环境都保持 level 0。第 2200 轮踏石首回合评估为 0/32，环境 0 在约 1.4 秒、前进约 0.63 米后触发 `bad_orientation`。足端记录显示前足长时间支撑，机身继续前移后前倾；这是一条失败轨迹，不能据此概括全部训练步态。

已有 MLP 教师经关节排列、默认姿态和动作缩放转换，在相同 AME 原生控制器与粗糙地形上通过 32/32；同一教师在 Stage2 双列桩 level 0 上为 0/32。该对照支持继续检查落脚学习，并没有证明所有接口都无问题。

## 明确改变的部分

- 保持上游 Go2 AME 网络、2124 维 actor 观测、关节动作和限位。策略推理时没有额外 MLP、IK 或动作修正。
- 从原版第 2200 轮权重开始，在原 Stage1 `rough` 地形上做 300 轮动作模仿，512 环境、每轮 24 步。保留原八列地形覆盖范围，地形类别改为 rough。
- 外部教师来自本仓库 `artifacts/perceptive/checkpoints/teacher.pt`，输入是理想扫描和本体状态。其既有训练来源见原感知策略报告，不能描述成 AME 从零训练。
- 教师动作转换到上游关节目标、按原限位裁剪后作为标签。每个环境执行教师动作的概率从 1.0 线性下降到 0.5，其余执行 AME 均值动作。
- 对 AME actor 和地形编码器做均方误差训练：Adam 1e-4，每批数据 2 个 epoch、8 个 minibatch。critic 没有监督；共享编码器会变化，后续必须重新学习价值函数。
- 保留 PPO 探索标准差；模仿权重只可用于 PPO warm-start，不可当作 PPO 完整续训状态。后续 checkpoint 继承初始化标签。

## 命令

从仓库根目录执行，变量均需设为实际路径：

```bash
/root/isaacsim/python.sh -m critical_ame.initialize_gait --headless \
  --upstream "$AME_UPSTREAM" --checkpoint "$AME_STAGE1_2200" \
  --teacher-checkpoint artifacts/perceptive/checkpoints/teacher.pt \
  --teacher-io artifacts/perceptive/teacher_native/stairs_0.50/summary.json \
  --num-envs 512 --iterations 300 --output "$NEW_INITIALIZER_RUN"

python3 -m critical_ame.assess_gait --upstream "$AME_UPSTREAM" \
  --run "$NEW_INITIALIZER_RUN" --output "$NEW_ASSESSMENT"
```

评估采用固定地形 seed 42、重置 seed 801、XY ±3 cm、yaw ±0.04 rad，32 个环境首回合。检查 Stage1 rough / stepping_stones / stairs_up level 0，以及 Stage2 double_stakes / alternate_stakes level 0、5。是否继续 PPO 由实测结果决定，模仿损失下降本身不代表通过。

首次启动曾将训练地形缩成一列，导致 512 个机器人的初始复制网格部分位于网格外，射线缺失后按预期报错退出。修正为保留八列覆盖范围后重新创建运行目录；未补零、未覆盖失败日志。

## 初始化后的独立结果

300 轮完成，用时约 879 秒，最后 minibatch 平均 MSE 约 0.013。以下测试均完全由 AME 输出动作，没有运行外部教师：

| 原生地形 | 等级 | 完整通过 |
|---|---:|---:|
| rough | 0 | 32/32 |
| stairs_up | 0 | 11/32 |
| stepping_stones | 0 | 0/32 |
| double_stakes | 0、5 | 均 0/32 |
| alternate_stakes | 0、5 | 均 0/32 |

结果表明模仿误差下降不等于过桩；基础步态迁移仍没有解决稀疏支撑。另做 Stage1 原始全八类地形 200 轮 PPO 微调，seed 43、512 环境，仅从初始化权重 warm-start，不在运行时调用教师。它属于同一外部初始化实验，不能并入原版从零训练曲线。

这组直接微调用时约 867 秒，rough level 0 退化为 2/32，stairs_up 提高到 31/32，踏石与已测双列/交错桩仍为 0/32。因此没有采用为新的稳定策略。

进一步设置独立对照：重新从模仿初始化开始，先固定 actor、共享地图 CNN/注意力、actor 本体编码和探索标准差，只更新 critic MLP 与 critic 本体编码 100 轮。结束时逐值检查所有冻结状态不变；通过后恢复整个网络与优化器，继续 200 轮正常 PPO。冻结编码器是为了避免价值损失通过共享网络改变 actor。这是对“价值函数不适配影响稳定性”的检验，不是已确认的原因。

100 轮 critic 预热已完成且冻结检查通过。后续 PPO 只记录到总编号 196，未生成完成标记；服务器复查时该 worker 已不在运行，按中断处理，不推断退出原因，也不报告它完成了原计划 200 轮。最后实际保存的检查点与日志分别归档。新的主线见 `MOE_PERCEPTIVE.md`，与这条对照分开。
