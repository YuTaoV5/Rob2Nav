# V4：真实深坑课程与失效诊断

2026-10-18。用户已批准停止平台期方案并执行新方案。**本页没有新的窄桩成功结论；V4仍在训练完整地图教师。**

## 已执行的停止与保全

旧 `nav_teacher_v3` 控制器及其训练进程组已通过 SIGINT 停止，不再自动启动后续块。最后一个完整原子检查点 `model_3500.pt` 经ZIP完整性检查后，复制到 `/root/gpufree-data/MoE-runs/nav_teacher_v3_stopped/`，保留源码、配置和SHA256。检查点后的未保存更新没有声称已保存。

[停止记录](artifacts/v4_redesign/v3_stop_for_redesign.json)、[保留权重](artifacts/v4_redesign/nav_teacher_v3_stopped/model_3500.pt)。第三轮前3000轮的三个完整gate均保留在 [v3_gates](artifacts/v4_redesign/v3_gates)。全部六组原桩始终为0/32；3000轮时平地/楼梯/高台各32/32。第四块未完成，不标记成完成4000轮。

## 先诊断，而非把猜测写成原因

固定第三轮 `model_2999.pt`，同一seed801、32环境、双列布局；本次新增执行前航向、足力和地图记录：

| 测试 | 完整通过 | 失败终止 | robot0平台边缘前航向偏差 | robot0低于平台0.1m的接触样本 |
|---|---:|---:|---:|---:|
| 原深坑，difficulty0.5 | 0/32 | 32/32 | 约63.1° | 0 |
| 浅坑诊断，difficulty0.5 | 2/32 | 30/32 | 约27.1° | 0 |
| 原深坑，difficulty1 | 0/32 | 32/32 | 约40.1° | 0 |

航向采样取robot0首次机身x>0.75m的诊断帧，不能作为所有32环境的统计。接触计数同样只来自robot0，不证明其他回合没有坑底支撑。**现有结果不支持简单断言“所有失败就是依赖坑底”**；浅坑本身也多数失败。可以确认旧策略存在较明显的航向与横向偏移，且深浅地形输入会改变行为。

![同一权重的足端首次接触与机身路径](artifacts/v4_redesign/v4_failure_analysis/foot_contacts.png)

[完整诊断统计](artifacts/v4_redesign/v4_failure_analysis/summary.json)、[原始回合/动作前地图](artifacts/v4_redesign/v4_diagnosis)。浅坑2/32不能计为原场景成功。

原生地图检查将robot0初始姿态的高度扫描XY与预期33×21网格逐点比较，再与解析原AME高度场比较；排除距碰撞盒边界2mm以内的格。两组各632个高度样本全部正确，最大XY误差约3.8微米，最大高度误差约2.6–3.8微米。**此次初始姿态检查未发现转置或偏移错误**，不代表已覆盖所有运动姿态和传感器误差。

[中等难度对齐记录](artifacts/v4_redesign/v4_diagnosis/deep05/map_alignment.json)、[最终难度对齐记录](artifacts/v4_redesign/v4_diagnosis/deep1/map_alignment.json)。

## 新课程的实际实现

- `moe_perceptive/sparse_course.py`新增 `wide_deep`：所有等级坑底始终−2m，支撑面从1.0m逐步缩到0.2m，起始间隙约5cm，保留5cm网格量化。
- 中间布局明确是训练前置任务。difficulty=1直接调用原AME生成器；测试已验证高度场逐格相同、碰撞盒列表相同。
- 原严格suite默认仍是original，不带wide_deep，绝不将宽桩成绩混为20cm原桩成绩。
- 每个环境在当前等级重新累计成功EMA与尝试数，完整到达后才晋级；晋级要求至少5次尝试且成功EMA>0.75，切换等级后清零，避免沿用易级成绩快速跳级。
- 针对诊断的偏航与横移：增加 `1-cos(yaw)` 惩罚（权重−2），通道中心惩罚从−0.5改为−2。没有在线修改关节动作或加入IK。
- 从第一轮MoE感知权重初始化，使用新的优化器与课程；不沿用第三轮已偏航/失效的策略。仍保留冻结的原MoE参考和明确物理限制。

四轮32环境原生短训练已完成，验证配置、PPO、课程更新与检查点可运行；两个几何测试已通过，确认每级真实坑及最终原几何一致。这些是接口和几何检查，不是越障技能证据。

[原生短训练完成记录](artifacts/v4_redesign/wide_deep_v4_smoke/completed.json)。

## 当前运行与自动评估

控制器 `moe_perceptive/deep_trial.py`；输出 `outputs/moe_perceptive/wide_deep_v4`；训练前缀 `/root/gpufree-data/MoE-runs/wide_deep_v4_chunkN`。

512环境，500轮一块，初始最多5块。每块结束：

1. 对原始九项场景完整评估，录制20cm双列/交错桩视频；仍是完整地图教师成绩。
2. 另测宽桩起始课程的双列/交错完整通过率，并做原生地图对齐检查，明确标为precursor。
3. 记录严格通过数、课程等级和precursor成绩；若连续两个gate均无实质改善，即使停在较高级也转入诊断，不再只判断是否卡在level0。

[启动记录](artifacts/v4_redesign/launch_wide_deep_v4.json)、[启动时源码/参数](artifacts/v4_redesign/wide_deep_v4_chunk1/manifest.json)。当前公开目录仅包含V4启动快照，不是最终权重或最终评估。

```bash
python3 -m moe_perceptive.deep_trial \
  --moe vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt \
  --initializer /root/gpufree-data/MoE-runs/visible_moe_v1_seed42/model_999.pt \
  --smoke /root/gpufree-data/MoE-runs/wide_deep_v4_smoke \
  --output outputs/moe_perceptive/wide_deep_v4 \
  --run-prefix /root/gpufree-data/MoE-runs/wide_deep_v4 \
  --chunks 5 --chunk-iterations 500 --num-envs 512
```

以上是已启动命令，接手后先检查进程，不重复运行。所有输出必须使用未存在的新目录。

## 后续仍未完成

教师真实深坑通过后，才继续可见D435i/Mid360学生蒸馏和PPO、安装/记忆误差验证、独立种子、MuJoCo以及目标点导航整合。当前阶段没有新增实机或学生成功结论。

归档包含旧三次完整gate、停止权重、匹配诊断、V4原生短训练及启动元数据。见 [文件校验清单](artifacts/v4_redesign/archive_manifest.json)。

## 2026-10-18 17:57 更新：V4 已触发平台期停训

控制器实际状态为 `needs_diagnosis`，完成四块共2000轮，保留 `wide_deep_v4_chunk4/model_1999.pt`，没有启动第五块。四次原场景严格评估的六组桩均为0/32。

| 块 | 平地 | 楼梯 | 高台 | 宽双列前置 | 宽交错前置 |
|---|---:|---:|---:|---:|---:|
| 1 | 32/32 | 32/32 | 27/32 | 3/32 | 17/32 |
| 2 | 32/32 | 32/32 | 31/32 | 7/32 | 18/32 |
| 3 | 32/32 | 29/32 | 28/32 | 5/32 | 22/32 |
| 4 | 32/32 | 32/32 | 25/32 | 4/32 | 6/32 |

第四块的独立视频/诊断重放为宽双列2/32、宽交错6/32；双列与普通评估有差异，不应假称严格确定性重放。两次初始地图对齐各632个非边界样本全部正确。robot0在平台边缘前偏航约−16.5°；双列最终机身位置约(1.84,−0.10,0.09)m，交错约(3.61,0.57,−0.13)m。仅这两个robot0轨迹均无足端低于平台0.1m的接触样本，不足以解释全部环境失败。

下一步先分析支撑几何、脚位、奖励与课程晋级。现有checkpoint只保存地形等级，未保存本级EMA/尝试次数，分块恢复会重新积累晋级资格；这是需要处理的恢复语义问题，但尚不能断言是失败根因。当前没有继续训练，也没有进入学生阶段。

服务器证据：`outputs/moe_perceptive/wide_deep_v4/`（四次gate与前置评估），`outputs/moe_perceptive/v4_plateau_diagnostic/`（视频、动作前地图、足力、analysis.json与contact_analysis.json）。这些新增完整V4结果尚未发布；公开的v4_redesign仍主要是V3停止、诊断和V4启动记录。

## V4完整评估归档（2026-10-18 18:12）

四次严格评估、八组宽桩前置评估、十段完整robot0视频及第二/第四块权重已收集至 [artifacts/v4_completed](artifacts/v4_completed)。[归档清单](artifacts/v4_completed/archive_manifest.json)记录1569个文件的SHA256与大小；服务器同时核对1296条源码引用及十段视频帧数，本地再次校验全部1569个文件。它们是失败与退化证据，不是成功演示。

复核命令：`python -m moe_perceptive.verify --root artifacts/v4_completed --manifest-name archive_manifest.json`（需numpy/torch及imageio/ffmpeg环境）。完整地图教师仍未通过原窄桩。

## 2026-10-18 18:32：arrival_v5_probe 有界探针

非零课程恢复原生验证通过：64环境，EMA注入0.37、次数7，恢复后两轮所有值保持一致；普通两轮+恢复两轮也完成。证据服务器 outputs/moe_perceptive/v4_resume_smoke/sentinel_check.json。

新增可选 --arrival-aware（旧V4默认奖励不变）：目标速度在距目标0.8m内逐渐降低，最大0.6m/s；进度改为朝目标投影速度并在近处衰减。原成功标准未放宽。旧奖励固定支付全局前进，存在接近目标仍获推进奖励的冲突；robot0交错诊断在x>3.3m的33帧平均水平速度约0.76m/s，不能外推全部失败。

已启动 moe_perceptive.arrival_trial，控制输出 outputs/moe_perceptive/arrival_v5_probe，训练 /root/gpufree-data/MoE-runs/arrival_v5_probe，512环境仅200轮，从V4 model1999初始化、全新优化器与课程。这是探索探针，并非严格隔离变量消融。完成后自动执行原九项严格suite、两类宽桩视频/落脚诊断，然后停止，不自动加轮数。仍为特权教师，不能计为原窄桩成功。

## 2026-10-18 18:42：探针失败，进入数值诊断

arrival_v5_probe在第54轮报 `normal expects all elements of std >= 0.0`，只有model0，未完成200轮。启动器异常返回0使控制器误进入缺失model199的评估；已停止进程组并写invalid_evaluation.json，所有这些下游尝试无效，不计通过率。arrival_trial新增完成标记、失败标记、指定权重和评估输出验证。不能把零退出码视为成功。

新增 --numeric-debug：观察/rollout有限值检查与autograd异常追踪，不做NaN替换或强行续训。已从相同初始权重启动512环境最多60轮的数值复现，输出 /root/gpufree-data/MoE-runs/arrival_numeric_debug，日志 outputs/moe_perceptive/arrival_numeric_debug/driver.log。这只是故障定位，不是新一轮成功训练。当前没有有效V5通过率。

## 2026-10-18 18:47：数值复现未复现原异常

numeric-debug完整跑完60轮，有限值检查与autograd异常检测均未触发；这不证明原异常已修复。已从model59保存的优化器、课程、随机状态恢复，追加最多140轮至总200轮；活动路径 /root/gpufree-data/MoE-runs/arrival_debug_cont140，控制日志 outputs/moe_perceptive/arrival_debug_cont140/driver.log。不再使用失败arrival_v5_probe的评估。下一步仅在completed标记存在、failed不存在、model199确实存在后运行严格suite与宽桩诊断。
