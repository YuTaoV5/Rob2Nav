# 预训练 MoE＋地形感知＋逐步解冻

本实验以用户指定的 RoboGauge Go2 MoE 为基础，增加 AME 思路的地形注意力分支，并分阶段联合训练。目标是仿真中的提前抬腿、台阶/高台和精确落脚。**已启动训练，不代表已获得窄桩能力。** 任意目标导航、绕水坑和跳深坑不属于这批固定前进路线训练的完成声明。

## 已完成的起步验证

- 严格恢复现有 `go2_moe_cts_137k_0.6739.pt` 的所有 actor/历史编码器参数；原文件 SHA256 为 `2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead`。
- 4 路、64 步、两次局部重置的比较通过。服务器 CPU 重建网络相对单路原 JIT 的最大动作差约 `1.04e-5`，来自批量运算数值舍入；新增分支为零时，相对重建基础策略逐值相同。
- 新感知分支能够反向传播；所有地图未知时输出有限；冻结/逐步解冻开关通过检查。
- 5 cm 地图的未知掩码、世界坐标重投影、过期、单环境重置与重复观测更新时间通过测试。
- Isaac Lab 的原 MoE 平地探针完成 **16/16** 个首回合，随后同条件 32 环境基线也完成 **32/32**。到达目标后需要稳定停住 0.3 秒，不以瞬时支撑相位代替完整任务。
- 32 环境、4 轮短训练依次经过 value/fusion/actor/joint，完成有限值与冻结参数检查。短训练仅验证训练链路。

这些结果不证明窄桩通过，也不证明感知已经产生收益。

## 复用哪些权重

原部署文件仍保留完整的推理网络权重，因此可以按其实际结构重建普通 PyTorch 模块并严格加载；不需要将基础步态重新蒸馏为另一个网络。原训练 critic、teacher、优化器和 RNG 不在部署文件中，故新 critic 重新初始化，不能称为恢复作者的完整训练状态。

作者当前训练实现与本仓库旧 137k 导出结构有差别。以实际初始化文件及逐步等价测试为准，未将另一版本权重强行装入。作者公开完整检查点的下载受当前网络限制；该限制没有改变我们复用的 actor/编码器参数。

第三方来源与许可见 `vendor/moe_reference/PROVENANCE.md`、`THIRD_PARTY_NOTICES.md`。

## 网络与观测

actor 输入为 **2304 维**：5×45 本体观测历史，加 33×21 地图的高度、有效掩码、观测年龄三个通道。本体数据严格保持原模型的关节顺序、默认姿态、缩放与“上一时刻实际动作”。历史保存在环境观测中，PPO minibatch 不会错误地推进网络内部历史。

原 MoE 将历史编码成 32 维潜变量，共享 actor 输出 12 个关节动作；它不是八个独立的跳跃/爬桩技能。新增 CNN 保留 5 cm 空间分辨率，由本体状态与潜变量查询四头地形注意力。

动作形式为 `q_target = q_default + 0.25 * (base_action + terrain_residual)`。感知输出头零初始化，初始动作保持原模型。随后可解冻原 actor 和历史编码器，因此最终不限于小幅修正。始终保留一份冻结参考网络，作为低起伏地形的动作偏离正则。

critic 独立使用 actor 观测、理想高度及有效掩码、机身速度，共 **3693 维**。理想地形用于 critic 和训练奖励，不作为 actor 的可见地图填充值。critic 与 actor 不共享可被价值损失改变的地形 CNN。

## 仿真传感器

- 深度相机：64×32 射线，近似 87°×58° 视场，下俯 20°，相对 base 位移 `(0.25,0,0.06)` m，4 m 范围。
- 雷达：96×12 方向，水平 360°、垂直约 −7°～52°，相对 base 位移 `(0,0,0.16)` m，8 m 范围。
- 二者约 10 Hz 更新，返回点共同写入局部地图，保留最多 1 秒已见信息；使用理想位姿重投影。
- 训练加入约 5 mm 高度噪声、2% 返回丢失。未知保持未知；向上/遮挡射线未命中是正常现象，不进行补零修复。

这是简化的 D435i/Mid360 风格仿真射线及拟定外参，不是完整硬件扫描、误差模型或标定复现。平地探针环境 0 的首回合平均已知单元约 31%，并非全图可见，也未据此声明雷达单独带来收益；传感器消融仍需后续评估。

## 首批训练

服务器：Isaac Sim 5.1、Isaac Lab 2.3.2、RSL-RL 3.1.2、单张 L40。512 环境，每轮 24 步，计划 1000 轮，共约 1229 万次新交互。

| 迭代编号 | 更新的部分 |
|---|---|
| 0～49 | 只训练 critic，固定整个 actor 与探索标准差 |
| 50～349 | 感知编码器、融合头与探索标准差，固定原 MoE |
| 350～649 | 同时开放原共享 actor |
| 650～999 | 同时开放原 MoE 历史编码器 |

固定学习率：新参数 `2e-4`，原 actor `2e-5`，原历史编码器 `1e-5`。感知残差边界依阶段为 0.10/0.20/0.30 rad；最终原 actor 可联合变化。所有阶段都保留不可训练的原参考网络。

地形比例：平地 1/8、台阶 2/8、高台 1/8、AME 双列桩 2/8、交错桩 2/8。每个课程等级按真实完成/失败更新。奖励包含速度、前进、通道、落脚支撑余量、小腿接触、关节限位与动作偏离；支撑余量在首次接触时计算，避免惩罚脚正常经过空隙。

AME 桩使用先前已经校验的 SII 几何实现：40→20 cm 支撑面、8×8 m 地块、2×2 m 中央平台、−2 m 坑。最终评估固定 difficulty=1；间隙按原 5 cm 网格截断，实际值见 `NARROW_VALIDATION.md`，不把名义参数误写成实际碰撞尺寸。

控制器沿用原 MoE 的关节顺序、20/0.5 PD 和 0.25 动作缩放，配置原 MuJoCo 对应的关节范围及 23.7/45.43 Nm 力矩上限。两引擎物理仍需实测，不能把网络数值等价当成动力学等价。

任务沿初始 +x 前进 3.5 m；一个简单路线监督器随后将速度命令置零。它不选择落脚、不修改关节动作。完成要求两足以上支撑、直立、速度低于 0.25 m/s 持续 0.3 s，且无失败终止。

## 运行与检查

从仓库根目录执行；输出必须是未存在的新目录：

```bash
MOE=vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt
python -m moe_perceptive.check --checkpoint "$MOE" --output outputs/model_check.json
python -m unittest discover -s tests -p test_moe_perceptive_memory.py

/root/isaacsim/python.sh -m moe_perceptive.train --headless \
  --moe "$MOE" --num-envs 512 --iterations 1000 \
  --output /path/to/new_run
```

当前运行目录为 `/root/gpufree-data/MoE-runs/visible_moe_v1_seed42`。其 `manifest.json` 与 `source/` 保留启动时精确源码；`phases.jsonl`、`iterations.jsonl`、`status.json` 记录实际进度。checkpoint 原子写入，保存下一轮编号、优化器、RNG 与课程；续训使用 `--resume`，保持 seed、环境数、地形及阶段边界一致，但物理瞬时状态不会恢复。

当前控制流水线位于 `outputs/moe_perceptive/pilot1000`：原 MoE 基线与训练并行，训练完成后用同一组 9 个案例评估新权重，包括 20 cm 双列/交错桩。两组均保存首回合统计、轨迹与最窄场景视频。固定布局上多个重置条件不能作为多种布局泛化成绩。

流水线通过独立进程会话启动，SSH 断开不依赖终端存活。`pipeline1000.log` 和 `launch1000.json` 记录启动；成功启动不等于完成训练。`completed.json` 只标记命令完成，通过率以逐场景 `summary.json` 为准。


## 首轮完成与感知诊断

首轮 1000 轮已完成，完整结果见 `MOE_PERCEPTIVE_VALIDATION.md`。窄桩仍为 0/32，不将该权重作为成功模型。

对同一权重、同一种子执行感知记忆消融：

```bash
/root/isaacsim/python.sh -m moe_perceptive.evaluate --headless \
  --moe "$MOE" --checkpoint /path/to/model_999.pt \
  --kind ame_stakes --difficulty 0 --num-envs 32 \
  --memory-lifetime 1 --diagnostic --output outputs/diagnostic_memory1
# 在新目录改为 --memory-lifetime 4 重复运行；其输入分布不同于训练，不能当作重新训练结果。
python -m moe_perceptive.analyze_diagnostic \
  --trials outputs/diagnostic_memory1 outputs/diagnostic_memory4 \
  --output outputs/coverage_diagnostic
```

诊断记录动作执行前的观测，包含机器人 0 的四脚格有效性、周围 3×3 格覆盖率和完整三通道地图。灰色表示未知，不能用理想高度补齐。覆盖率是单机器人时序诊断，完整通过率仍统计全部 32 环境。诊断不改变关节动作。


第二轮控制命令（输出目录必须未存在）：

```bash
python -m moe_perceptive.memory_trial --moe "$MOE" \
  --initializer /root/gpufree-data/MoE-runs/visible_moe_v1_seed42/model_999.pt \
  --run /root/gpufree-data/MoE-runs/visible_moe_memory4_v2_seed42 \
  --output outputs/moe_perceptive/memory4_v2 --iterations 1000
```

控制器依次执行 2 轮原生短训练、512 环境联合训练、9 项评估。初始化只加载 policy/critic，优化器、课程、回合不恢复；4 秒记忆仍只保存实际观测，不补隐藏地形。状态与每阶段命令/返回码存入输出目录。
