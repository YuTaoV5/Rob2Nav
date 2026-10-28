# 使用 Rob2Nav MoE 行走模型辅助 SuperMimic

2026-10-26。用户明确指定使用工作台已有MoE行走权重作为教师。原V1、姿态V2和目标率V3没有使用教师；它们的结果不改写。尚未启动的V4纯奖励加权队列已记录为 `cancelled_for_moe_teacher`，不恢复。

## 1. 教师到底是哪一个

读取本仓库 `deployment/skills/walk_moe.json`，实际模型为：

```text
vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt
SHA256: 2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead
```

权重与部署合约已在Rob2Nav中保存，不重复复制一份新命名权重。适配器启动时强制校验SHA、观测类型和50Hz控制频率。

**该RoboGauge导出与上游 `MoeCtsDriver` 的go2_rl_lab导出格式不同。** 原TorchScript没有 `feature_dims`，不能仅传入 `--driver-policy` 后沿用原驱动器。

| 项目 | Rob2Nav MoE教师 | SuperMimic学生 |
| --- | --- | --- |
| 单帧观测 | 45 | 70（包含28槽命令） |
| 历史 | 5×45，归一化后按时间展开 | 每term 5历史，总350维 |
| 命令缩放 | vx/vy/wz为2、2、0.25 | 前三槽为物理速度，其余25零 |
| 动作 | 按FL/FR/RL/RR顺序，默认姿态+0.25×raw | 按环境实际关节顺序，默认姿态+逐关节scale×action |
| 控制频率 | 50Hz | 50Hz，4个200Hz物理子步 |

`rob2nav_moe_teacher.py` 复用模型内部normalizer、MoE encoder和actor，重建独立的N环境历史，逐关节将目标角映射到学生动作空间。没有把MoE的网络参数直接装入350维学生actor，也没有向学生增加教师latent、地形真值或线速度真值。

## 2. 已验证什么

`verify_moe_teacher.py` 用4个独立原始batch-1模型对照新批量实现，20帧中包含单环境历史重置，最大输出误差约2.83e-6，小于1e-5门槛。

纯教师在SuperMimic执行器下的8环境40秒八阶段评估：无跌倒、无非足接地，目标角映射误差约1.8e-7；但关节速度峰比1.078、位置越限约0.00946rad、目标变化率1.778。因此没有把教师所有输出都作为合格模仿标签。

测试保留SuperMimic执行器：髋/大腿Kp约17.692、Kd约1.126，小腿约39.807/2.534；MoE原生合约为20/0.5。两组增益都写入teacher_contract，测试结论仅描述这一适配工况，不静默改变PD来掩盖差异。

## 3. 教师如何参与训练

1. 初始25%的环境由MoE按**当前同一速度命令**驱动，给世界模型提供真实动力学数据。移除原上游追球/随机航点转向器。
2. 学生环境保留原探索噪声；教师环境使用干净MoE动作。修改执行动作时同时更新回放中同一动作张量，防止“执行的是A，世界模型拟合的是B”。
3. 真实采样照常进入世界模型，不删掉教师的失误片段。
4. 可选辅助模仿使用教师环境的学生观测和映射后目标角。仅对目标变化率≤0.8、转移后实际关节速度≤0.8限值、硬关节位置有效且未终止的帧做BC。这个筛选只作用于辅助标签，不改物理状态或评估报告。
5. BC为独立actor-only、无动量SGD，初始权重0.1、lr1e-4，关节目标误差按0.25rad归一化。世界模型、critic和教师权重不被BC更新。这是本项目明确记录的扩展，不宣称是原论文原有目标。
6. 600–1000轮逐渐撤去教师环境及BC，最后200轮完全由学生采样。最终独立评估与录像只加载学生actor。

32环境×24轮烟测实际完成：272条教师环境采样、269条合格辅助标签、20次非零教师辅助更新；命令误差0、执行动作对应误差0、映射误差小于1e-6；共享转移奖励误差0，学生命令历史填充零。

## 4. 新服务器逐条复现

先完成部署章节，安装最新overlay。以下变量沿用统一约定：

```bash
cd "$SUPERMIMIC_HOME"
export PROFILE="$ROB2NAV/deployment/skills/walk_moe.json"
export MOE="$ROB2NAV/vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt"
export PARENT="$ROB2NAV/skills/supermimic/checkpoints/velocity_gait_v3_model_800.pt"
export GAIT_CONFIG="$ROB2NAV/skills/supermimic/resources/gait_targetmax_v3.json"
sha256sum "$MOE"
"$VENV/bin/python" scripts/verify_moe_teacher.py \
  --policy "$MOE" --output reproduction/moe_teacher/forward_parity.json
"$VENV/bin/python" scripts/eval_moe_teacher.py \
  --teacher-profile "$PROFILE" --checkpoint "$PARENT" \
  --seed 34010 --num-envs 8 --stage-seconds 5 --initialize-hold-targets \
  --output reproduction/moe_teacher/teacher_strict.json \
  --gait-output reproduction/moe_teacher/teacher_gait.json
```

这里 `--checkpoint` 仅为同一学生环境合约，教师评估不执行其中actor；输出显式保存 `student_actor_used=false`、教师SHA和增益。

跑教师训练烟测：

```bash
"$VENV/bin/python" -u scripts/train_gait_moe.py \
  --teacher-profile "$PROFILE" --teacher-bc-weight 0.1 --teacher-bc-lr 0.0001 \
  --gait-config "$GAIT_CONFIG" --task Go2-Velocity-Gait-V2 \
  --driver-policy "$MOE" --driver-frac 0.25 --driver-fade 1000 2000 \
  --num-envs 32 --iterations 24 --warmup-iterations 2 \
  --critic-warmup-iterations 2 --actor-ramp-iterations 2 \
  --buffer-len 40 --imag-batch 32 --seed 34001 --log-interval 1 --save-interval 24 \
  --resume "$PARENT" --reset-critic true --continue-run false \
  --lr 0.00005 --actor-lr-scale 0.25 --run-name velocity_gait_moe_teacher_v1_smoke
```

检查实际 `teacher_contract.json`、`teacher_audit.json` 和 `gait_audit.json`，再填入真实烟测目录：

```bash
export SMOKE_RUN="$SUPERMIMIC_HOME/logs/supermimic/填入实际时间戳_velocity_gait_moe_teacher_v1_smoke"
"$VENV/bin/python" -u scripts/moe_training_pipeline.py \
  --source "$SUPERMIMIC_HOME" --parent "$PARENT" --config "$GAIT_CONFIG" \
  --teacher-profile "$PROFILE" --smoke-run "$SMOKE_RUN" \
  --teacher-evidence reproduction/moe_teacher --output reproduction/moe_teacher/formal_trial \
  --rob2nav "$ROB2NAV"
```

现有服务器已经启动唯一正式管线，不再执行第二份。正式配置4096环境、1200轮、seed34011，模型预热100、critic预热50、actor爬升100；保留8步世界模型/32步想象，lr5e-5、actor倍率0.25。重置critic、重建回放，父actor/world model为V3/model800。

管线校验源码/教师SHA与烟测后才训练；结束自动以34020–34023各128环境80秒验证学生，再做33000步态配对及34030原速录像。提供 `--rob2nav` 时自动以研究候选发布视频，需要ffmpeg与Node/npm。

## 5. 验收与下一步

读取 `reproduction/moe_teacher/formal_trial/state.json` 和日志。真实学生是否保持身高/DF/滑移改善、是否改进速度转向、是否消除目标突变，都以脱离教师的物理评估为准。不能用教师混合环境中的低跌倒数替代学生独立成绩。

检查点包含学生actor、critic、世界模型和teacher_training元数据；推理无需MoE。恢复教师辅助训练则仍需相同SHA教师和训练参数。当前未获Go2 EDU实机资格，也未开始盲楼梯训练；后续先按实际失败机制决定课程和教师比例，不无限追加相同条件。

若断点已经超过教师渐退终点1000轮，后续纯学生续训使用 `train_gait.py`、相同gait-config和 `Go2-Velocity-Gait-V2`，不传driver-policy；若重新开启教师阶段，应新建实验并明确新计数，不能把重新建立的物理环境/回放当作逐位恢复。

## 6. 本次正式执行记录

1200轮已完成：实际教师环境采样828736帧、BC合格801771帧、辅助更新848次；命令/执行对应误差0，映射误差约4.77e-7，共享奖励误差0。末轮world-model loss约0.1832、critic loss约0.0002104。

独立34020–34023四种子均无跌倒/非足接地，但目标变化率约1.133、1.234、1.290、1.090，仍未完整通过。教师辅助执行成功与学生能力达标分开记录。后续应对同一组种子做“仅教师采样/教师采样+BC”的单因素对照，或先分析后退切换的观测与动作历史敏感性，不能据不同种子数字直接推断BC一定改善或损害。

## 7. 教师仅采样的单因素对照

只将BC权重0.1改为0，保留同一个V3父模型、训练seed34011、教师采样/渐退、奖励、预算及独立评估种子34020–34023。教师仍驱动一部分环境帮助世界模型，学生不直接做BC。这用于检验辅助模仿与连续性的权衡，不预先宣称哪一组更好。

复制上方32×24烟测命令，把 `--teacher-bc-weight` 设为0、run_name设为 `velocity_gait_moe_dataonly_v1_smoke`，其余不变。审计应有真实教师帧、BC更新数严格为0、共享奖励与命令/执行检查通过。然后设置新的实际SMOKE_RUN：

```bash
export SMOKE_RUN="$SUPERMIMIC_HOME/logs/supermimic/填入实际时间戳_velocity_gait_moe_dataonly_v1_smoke"
"$VENV/bin/python" -u scripts/moe_training_pipeline.py \
  --source "$SUPERMIMIC_HOME" --parent "$PARENT" --config "$GAIT_CONFIG" \
  --teacher-profile "$PROFILE" --smoke-run "$SMOKE_RUN" \
  --teacher-evidence reproduction/moe_teacher \
  --output reproduction/moe_teacher_dataonly/formal_trial --rob2nav "$ROB2NAV" \
  --teacher-bc-weight 0 --run-name velocity_gait_moe_dataonly_v1_bounded_20261026 \
  --seed 34011 --eval-seed-start 34020 \
  --asset-id velocity_gait_moe_dataonly_v1_model1200 \
  --asset-title 'MoE教师仅采样 · SuperMimic学生第1200轮'
```

两个实验同名seed的报告位于不同trial目录，禁止混用或覆盖。现有服务器该对照的进程/state若已存在，先核对，不重复提交。

## 8. 同种子对照已完成

两组均完成4096环境、1200更新，评估时仅运行学生。实际配置审计确认训练参数仅run_name变化、教师合约仅BC权重0.1→0变化，同父模型、训练seed34011、评估34020–34023。

| 教师方式 | 四种子原始目标变化率/URDF限速 | 该单项通过 |
|---|---|---|
| 教师数据+BC 0.1 | 1.133 / 1.234 / 1.290 / 1.090 | 0/4 |
| 仅教师数据，BC 0 | 0.980 / 0.974 / 1.177 / 1.002 | 2/4 |

两组四种子均无跌倒或非足接地。2/4仅指目标变化率门槛，不是整项能力验收；所有模型仍capability_qualified=false。该单一训练种子的对照支持优先诊断仅教师数据候选，不能推断BC普遍有害。下一步定位34022的目标突变，检查指令阶段、动作历史与步态指标，禁止原条件盲目追加长训。完整原速人工复看与实机验证尚未完成。

新候选为 `skills/supermimic/checkpoints/velocity_gait_moe_dataonly_v1_model_1200.pt`，实际配置及独立报告见 `skills/supermimic/evidence/reports/moe_teacher_dataonly/`。网站 `/supermimic.html` 已发布对应原速回放。两个训练流水线均已结束，不重复启动。

## 9. 连续性有界优化：800轮完成

继承仅教师数据学生1200，保持步态与控制合约，仅目标速率权重1→4；纯学生训练4096环境800轮，seed34111。32×24前置烟测共享奖励误差0、actor梯度非零；正式奖励审计误差0，最终WM loss 0.16、critic loss 0.0006297。

新独立34120–34123目标变化率0.896/0.979/0.929/0.922，原34020–34023冻结配对0.919/0.928/0.903/0.959。八组均无跌倒、非足接地、位置越限或实际速度超限；新四种子物理门槛4/4通过。新旧种子均为每种子128环境80秒、200Hz逐物理步，保留原突变指令，没有输出滤波或门槛放宽。

配对前进身高0.2924m、滑移0.0464m/s，DF四腿0.557/0.531/0.499/0.550，倾角RMS1.55度。DF腿间差增大，后续仍需关注步态均衡与速度跟踪权衡。完整原速人工复看、统一部署适配和实机仍未完成，capability_qualified=false，不能直接晋级盲楼梯。

检查点 `skills/supermimic/checkpoints/velocity_moe_continuity_v2_model_800.pt`；配置 `skills/supermimic/resources/gait_moe_continuity_v2.json`；全部新旧种子报告在 `skills/supermimic/evidence/reports/moe_continuity_v2/`。可按第7节同样的环境变量设置，改用 `scripts/gait_refinement_pipeline.py --source "$SUPERMIMIC_HOME" --parent "$PARENT" --expected-parent-sha f54551a392112befb3b92985388f66f43fcd8f92c8bd5d43b1bf803b4674ba6f --config "$ROB2NAV/skills/supermimic/resources/gait_moe_continuity_v2.json" --smoke-run "$SMOKE_RUN" --output reproduction/moe_continuity_v2/formal_trial --run-name velocity_moe_continuity_weight4_v2_bounded_20261026 --iterations 800 --seed 34111 --eval-seed-start 34120`。PARENT应指向仅教师数据1200模型；SMOKE_RUN必须先用同父同配置完成32×24烟测，不能沿用其他配置审计。已有服务器任务已结束，不重复执行此复现示例。

## 10. 中间惩罚权重的同种子对照

权重2候选已完成4096环境800轮，与权重4实验使用同一MoE dataonly父模型、训练seed34111、训练预算与优化参数，仅目标变化率权重不同。32×24烟测及正式审计共享奖励误差0、命令填充为0。正式末轮WM loss0.1614、critic loss0.001229。

新独立34220–34223原始目标变化率为0.9913/1.0494/1.0679/0.9990，物理门槛2/4；两次失败发生在切换后退首帧的RR小腿指令。原34020–34023冻结配对为0.9953/0.9730/1.0050/0.9752，目标连续性单项3/4。全部报告保留，不将边缘超限四舍五入为通过。新四种子无跌倒、非足接地、关节越限或实际超速。

同seed33000、8环境配对稳态速度RMSE（剔除阶段首1秒）如下：

| 方案 | 前进 | 后退 | 左移 | 右移 | 停步 |
|---|---:|---:|---:|---:|---:|
| MoE仅数据父模型 | .04175 | .02341 | .03243 | .02644 | .00329 |
| 目标速率权重4 | .05090 | .05341 | .06994 | .04759 | .01229 |
| 目标速率权重2 | .04754 | .04033 | .05526 | .04549 | .00492 |

单位m/s。权重2部分恢复跟踪，但仍不及父模型，且连续性未全部通过。前进DF腿间差0.06575、倾角RMS1.72度、滑移0.04469m/s；不能称全面改善。下一步应诊断切换与常态动作约束的冲突，禁止继续盲目调整标量或放宽门槛。

研究候选 `skills/supermimic/checkpoints/velocity_moe_continuity_tradeoff_v3_model_800.pt`；配置 `skills/supermimic/resources/gait_moe_continuity_tradeoff_v3.json`；配置审计与报告分别见evidence/config中的tradeoff文件和evidence/reports/moe_continuity_tradeoff_v3。

复现沿用第9节命令，PARENT仍为MoE仅数据1200模型，先按权重2配置完成32×24烟测，再将config换为本节配置、smoke-run换为实际新目录、output设为reproduction/moe_continuity_tradeoff_v3/formal_trial、run-name设为velocity_moe_continuity_weight2_v3_bounded_20261026、seed仍34111、iterations仍800、eval-seed-start设34220。已有任务均结束，不重复提交。

新视频velocity_moe_continuity_tradeoff_v3_model800已发网站，HTTP206验证；概览只显示最新两个，更多页保留历史。完整原速人工复看、统一部署适配、实机未完成，capability_qualified=false。
